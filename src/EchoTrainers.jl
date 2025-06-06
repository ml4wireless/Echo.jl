module EchoTrainers
export EchoTrainer, train!

using Accessors
using Crayons
using Flux
using Logging
using Optimisers
using Printf
using ProgressMeter
using Statistics
using StatsBase: sample
using TensorBoardLogger

using ..Agents
using ..DataUtils: get_random_data_sb
using ..Evaluators
using ..FluxUtils
using ..LookupTableUtils
using ..ModulationModels
using ..Protocols
using ..Simulators
using ..Schedules: step!

# using PrettyPrinting
using Infiltrator

import Base.show


"""
    EchoTrainer(agents, protocol)

EchoTrainer holds state necessary to simulate interactions and train Echo agents.
"""
struct EchoTrainer
    agents::Vector{Agent}
    # Store simulator_class instead of simulator to allow sampling of agents
    simulator_class::Type
    evaluator_class::Type
    update_fn!::Function
    roundtrip::Bool
end

function EchoTrainer(agents, protocol)
    roundtrip = protocol ∈ [ESP, EPP]
    shared_or_grad = protocol ∈ [GP, ESP]
    if roundtrip
        simulator = shared_or_grad ? SharedPreambleSimulator : PrivatePreambleSimulator
        update_fn! = shared_or_grad ? update_shared_preamble! : update_private_preamble!
    else
        simulator = shared_or_grad ? GradientPassingSimulator : LossPassingSimulator
        update_fn! = shared_or_grad ? update_gradient_passing! : update_loss_passing!
    end
    EchoTrainer(agents, simulator, Evaluator, update_fn!, roundtrip)
end

Base.show(io::IO, et::EchoTrainer) = print(io,
    Crayon(bold=true, foreground=:light_magenta), "EchoTrainer", Crayons(reset=true),
    "(agents = $(et.agents), sim = $(et.simulator_class), rt = $(et.roundtrip))"
)


"""
    apply_model_updates!(optims, models, grads; update_log_std = true)

Apply optimiser updates to models based on grads.

Uses in-place version of update! which is more efficient, but mutates the
original model, and returns a new model which is the only copy guaranteed
to be correct.

# Returns
- New optimiser states `newoptims`: Array of updated optimiser state, rule trees
- New optimised models `newmodels`: Array of updated models which must be
                                    assigned back to any higher level containers
"""
function apply_model_updates!(optims, models, grads; update_log_std::Bool=true)
    if grads === nothing
        @debug "`grads` is nothing, not updating"
        return optims, models
    end
    newoptims, newmodels = [], []
    for (o, m, g) in zip(optims, models, grads)
        if m.mod !== nothing && !update_log_std && hasfield(typeof(m.mod), :log_std)
            Optimisers.freeze!(o.mod.log_std)
        end
        newo, newm = Optimisers.update!(o, m, g)
        push!(newoptims, newo)
        push!(newmodels, newm)
    end
    newoptims, newmodels
end


"""
Returns a new Simulator with the agents replaced and remaining params untouched.

Required to produce correct gradients with new explicit mode differentiation.
"""
function replace_simulator_agents(sim::S, agents::Tuple{Agent, Agent}) where {S <: Simulator}
    typeof(sim)(agents[1], agents[2], sim.bits_per_symbol, sim.len_preamble, sim.cuda)
end


"""
Calculate current non-exploration BER across SNRs

Compares first agent to every other agent to avoid combinatorial growth of BER checks
"""
function validate(trainer::EchoTrainer, len_preamble, verbose::Bool)
    final_ber_array = []
    for i in 2:length(trainer.agents)
        evaluator = trainer.evaluator_class(trainer.simulator_class, trainer.agents[1], trainer.agents[i], trainer.roundtrip)
        t0 = time()
        ber_array = evaluate(evaluator, len_preamble=len_preamble)
        push!(final_ber_array, ber_array)
        if verbose
            println("$(time() - t0) seconds to evaluate")
            println("Mean 1->$i BER ", mean(ber_array, dims=2)[3])
        end
    end
    final_ber_array = cat(final_ber_array..., dims=3)
    val_dict = Dict(:ber => final_ber_array,
                    :kwargs => (; (Symbol("agent$i") => get_kwargs(trainer.agents[i]) for i in 1:length(trainer.agents))...)
                    )
    val_dict
end


"""
Gradient Passing update logic
"""
function update_gradient_passing!(simulator, eavesdroppers, SNR_db,
                                  optims, optim_models, verbose)
    loss = 0
    local res
    grads = Flux.gradient(optim_models) do agents
        simulator = replace_simulator_agents(simulator, agents)
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble)
        loss = loss_crossentropy(logits=res.d2_logits, target=target)
        loss
    end
    if verbose
        println("loss: $(loss)")
    end
    # Pass grads[1] because we pass in agents as first arg to gradient()
    optims, optim_models = apply_model_updates!(optims, optim_models, grads[1], update_log_std=false)
    (; optims, optim_models, loss)
end


"""
Loss Passing update logic
"""
function update_loss_passing!(simulator, eavesdroppers, SNR_db,
                              optims, optim_models, verbose)
    m1_loss = d2_loss = 0
    local res
    grads = Flux.gradient(optim_models) do agents
        simulator = replace_simulator_agents(simulator, agents)
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble)
        if isneural(agents[2].demod)
            # Train demod
            d2_loss = loss(agents[2].demod, logits=res.d2_logits, target=target)
        end
        if isneural(agents[1].mod)
            # Train mod
            m1_loss = loss(agents[1].mod, symbols=res.preamble, received_symbols=res.d2_symbs, actions=res.m1_actions)
        end
        loss_sum = m1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), d2 loss: $(d2_loss)")
    end
    # Update the partner models based on this simulation
    a1, a2 = optim_models
    o1, o2 = optims
    pm_loss = 0
    if a1.use_prtnr_model
        a1, o1, pm_loss = update_partner_model!(a1, o1, res, verbose)
    end  # Only agent 1 (mod)'s partner model can be updated for LP
    optim_models = (a1, a2)
    optims = (o1, o2)
    # Update the mod and demod models
    # Pass grads[1] because we pass in agents as first arg to gradient()
    optims, optim_models = apply_model_updates!(optims, optim_models, grads[1])
    (; optims, optim_models, m1_loss, d2_loss, pm_loss)
end


"""
Shared preamble update logic
"""
function update_shared_preamble!(simulator, eavesdroppers, SNR_db,
                                 optims, optim_models, verbose)
    d1_loss = d2_loss = m1_loss = m2_loss = 0
    local res
    grads = Flux.gradient(optim_models) do agents
        simulator = replace_simulator_agents(simulator, agents)
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble1)
        if isneural(agents[1].demod)
            # Train demod of a1 agent
            d1_loss = loss(agents[1].demod, logits=res.d1_logits, target=target)
        end
        if isneural(agents[2].demod)
            # Train demod of a2 agent
            d2_loss = loss(agents[2].demod, logits=res.d2_logits, target=target)
        end
        if isneural(agents[1].mod)
            # Train mod of a1 agent
            m1_loss = loss(agents[1].mod, symbols=res.preamble1, received_symbols=res.d1_rt_symbs, actions=res.m1_actions)
        end
        if isneural(agents[2].mod)
            # Train mod of a2 agent
            m2_loss = loss(agents[2].mod, symbols=res.preamble2, received_symbols=res.d2_rt_symbs, actions=res.m2_actions)
        end
        loss_sum = m1_loss + m2_loss + d1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), m2 loss: $(m2_loss), d1 loss: $(d1_loss), d2 loss: $(d2_loss)")
    end
    # Update the partner models based on this simulation
    a1, a2 = optim_models
    o1, o2 = optims
    pm1_loss = pm2_loss = 0
    if a1.use_prtnr_model
        a1, o1, pm1_loss = update_partner_model!(a1, o1, res, 1, verbose)
    end
    if a2.use_prtnr_model
        a2, o2, pm2_loss = update_partner_model!(a2, o2, res, 2, verbose)
    end
    optim_models = (a1, a2)
    optims = (o1, o2)
    # Update the mod and demod models
    # Pass grads[1] because we pass in agents as first arg to gradient()
    optims, optim_models = apply_model_updates!(optims, optim_models, grads[1])
    (; optims, optim_models, m1_loss, m2_loss, d1_loss, d2_loss, pm1_loss, pm2_loss)
end


"""
Private preamble update logic
"""
function update_private_preamble!(simulator, eavesdroppers, SNR_db,
                                  optims, optim_models, verbose)
    d1_loss = d2_loss = m1_loss = m2_loss = 0
    local res
    grads = Flux.gradient(optim_models) do agents
        simulator = replace_simulator_agents(simulator, agents)
        res = simulate(simulator, SNR_db, explore=true)
        target1 = symbols_to_onehot(res.preamble1)
        target2 = symbols_to_onehot(res.preamble2)
        if isneural(agents[1].demod)
            # Train demod of a1 agent
            d1_loss = loss(agents[1].demod, logits=res.d1_rt_logits, target=target1)
        end
        if isneural(agents[2].demod)
            # Train demod of a2 agent
            d2_loss = loss(agents[2].demod, logits=res.d2_rt_logits, target=target2)
        end
        if isneural(agents[1].mod)
            # Train mod of a1 agent
            m1_loss = loss(agents[1].mod, symbols=res.preamble1, received_symbols=res.d1_rt_symbs, actions=res.m1_actions)
        end
        if isneural(agents[2].mod)
            # Train mod of a2 agent
            m2_loss = loss(agents[2].mod, symbols=res.preamble2, received_symbols=res.d2_rt_symbs, actions=res.m2_actions)
        end
        loss_sum = m1_loss + m2_loss + d1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), m2 loss: $(m2_loss), d1 loss: $(d1_loss), d2 loss: $(d2_loss)")
    end
    # Update the partner models based on this simulation
    a1, a2 = optim_models
    o1, o2 = optims
    pm1_loss = pm2_loss = 0
    if a1.use_prtnr_model
        a1, o1, pm1_loss = update_partner_model!(a1, o1, res, 1, verbose)
    end
    if a2.use_prtnr_model
        a2, o2, pm2_loss = update_partner_model!(a2, o2, res, 2, verbose)
    end
    optim_models = (a1, a2)
    optims = (o1, o2)
    # Update the mod and demod models
    # Pass grads[1] because we pass in agents as first arg to gradient()
    optims, optim_models = apply_model_updates!(optims, optim_models, grads[1])
    (; optims, optim_models, m1_loss, m2_loss, d1_loss, d2_loss, pm1_loss, pm2_loss)
end


#############################################################
# Self-play and partner model update logic
#############################################################
"""
Partner model update with cross-entropy loss (KL-divergence for hard decision feedback)
"""
function update_partner_model!(agent, opt, res::RTResults, agent_idx::Int, verbose=false)
    """
    Update agent with partner model via SGD(lr=η) for round-trip, agent number `agent_idx`"
    """
    loss = 0
    # Run forward pass and update
    grads = Flux.gradient(agent.prtnr_model) do prtnr_model
        if agent_idx == 1
            target = symbols_to_onehot(res.d1_rt_symbs)
            logits = prtnr_model(res.m1_actions)
        else
            target = symbols_to_onehot(res.d2_rt_symbs)
            logits = prtnr_model(res.m2_actions)
        end
        loss = loss_crossentropy(logits=logits, target=target)
        loss
    end
    if verbose
        println("Partner model loss: $(loss)")
    end
    newopt, newprtnr_model = Optimisers.update!(opt.prtnr_model, agent.prtnr_model, grads[1])
    agent = Agent(agent, prtnr_model=newprtnr_model)
    opt = @set opt.prtnr_model = newopt
    (; agent, opt, loss)
end

function update_partner_model!(agent, opt, res::HTResults, verbose=false)
    """
    Update agent with partner model via SGD(lr=η) for half-trip"
    """
    loss = 0
    # Run forward pass and update
    grads = Flux.gradient(agent.prtnr_model) do prtnr_model
        target = symbols_to_onehot(res.d2_symbs)
        logits = prtnr_model(res.m1_actions)
        loss = loss_crossentropy(logits=logits, target=target)
        loss
    end
    if verbose
        println("Partner model loss: $(loss)")
    end
    newopt, newprtnr_model = Optimisers.update!(opt.prtnr_model, agent.prtnr_model, grads[1])
    agent = Agent(agent, prtnr_model=newprtnr_model)
    opt = @set opt.prtnr_model = newopt
    (; agent, opt, loss)
end


function update_self_play!(agent, train_args, bps, SNR_db, iter=1)
    """
    Update agent with gradient-passing self-play via SGD(lr=η)"
    """
    a1 = Agent(agent.mod, nothing)
    a2 = Agent(nothing, agent.demod)
    η = train_args.self_play.lr * train_args.self_play.lr_decay ^ (iter - 1)
    o1 = Optimisers.setup(Optimisers.Descent(η), a1)
    o2 = Optimisers.setup(Optimisers.Descent(η), a2)
    # Run simulation and update
    simulator = GradientPassingSimulator(a1, a2, bps, train_args.len_preamble, iscuda(agent))
    for _ in 1:train_args.self_play.rounds
        newoptims, newa12, losses... = update_gradient_passing!(
            simulator, Vector{Agent}(), SNR_db,
            (o1, o2), (a1, a2), train_args.verbose
        )
        a1, a2 = newa12
        o1, o2 = newoptims
    end
    Agent(agent, mod=a1.mod, demod=a2.demod)
end


function update_from_partner_model!(agent, train_args, iter=1)
    """
    Update agent with gradient-passing and partner model via SGD(lr=η)"
    """
    newmod = agent.mod
    η = train_args.partner_modeling.lr
    o = Optimisers.setup(Optimisers.Descent(η), agent.mod)
    cuda = iscuda(agent)
    bps = agent.mod.bits_per_symbol
    len_preamble = train_args.len_preamble
    # Run forward pass and update
    for _ in 1:train_args.partner_modeling.rounds
        preamble = get_random_data_sb(len_preamble, bps, cuda=cuda)
        target = symbols_to_onehot(preamble)
        grads = Flux.gradient(newmod) do mod
            actions = modulate(mod, preamble, explore=false)
            logits = agent.prtnr_model(actions)
            loss_crossentropy(logits=logits, target=target)
        end
        o, newmod = Optimisers.update!(o, newmod, grads[1])
    end
    # Return agent with new mod
    Agent(agent, mod=newmod)
end


#############################################################
# Helpers for train!()
#############################################################
_ensemble_train_bers(bers) = mean(bers, dims=3)[:, 5]
_safe_diversity(a) = isneural(a.mod) ? loss_diversity(constellation(a.mod)) : 0f0
_diversity_losses(agents) = [_safe_diversity(a) for a in agents]

"""
    flatten(x::Tuple)
    flatten(x::NamedTuple)
`flatten` a nested tuple or named tuple. The result will be a tuple consisting of the
leaves.
EXAMPLE:
```
julia> x = randnt(3,2)
(y = (f = :f, b = :b, t = :t), w = (m = :m, f = :f), s = (m = :m, v = :v, q = :q))
julia> flatten(x)
(:f, :b, :t, :m, :f, :m, :v, :q)
```

From [NestedTuples.jl](https://github.com/cscherrer/NestedTuples.jl/blob/master/src/leaves.jl)
"""
flatten(x, y...) = (flatten(x)..., flatten(y...)...)
flatten(x::Tuple) = flatten(x...)
flatten(x::NamedTuple) = flatten(values(x)...)
flatten(x) = (x,)
flatten() = ()

function log_learning_rate(logger, lrs, iter)
    mulrs = filter(x -> x isa Number, flatten(lrs.mod.μ))
    stdlr = lrs.mod.log_std
    dlrs = Float32[]
    if isa(lrs.demod, NamedTuple) && haskey(lrs.demod, :net)
        dlrs = filter(x -> x isa Number, flatten(lrs.demod.net))
    end
    # if iter == 1
    #     mlayout = Dict("LearningRate" => Dict("Learning Rate" => (tb_multiline, vcat(["Mod/log_std"], ["Mod/μ.$i" for i in 1:length(mulrs)]))))
    #     dlayout = Dict("LearningRate" => Dict("Learning Rate" => (tb_multiline, ["Demod/net.$i" for i in 1:length(dlrs)])))
    #     log_custom_scalar(logger, mlayout)
    #     log_custom_scalar(logger, dlayout)
    # end
    for (i, lr) in enumerate(mulrs)
        log_value(logger, "LearningRate/Mod/μ.$i", lr, step=iter)
    end
    log_value(logger, "LearningRate/Mod/log_std", stdlr, step=iter)
    for (i, lr) in enumerate(dlrs)
        log_value(logger, "LearningRate/Demod/net.$i", lr, step=iter)
    end
end

function log_losses(logger, losses, iter)
    # @info "loss" m1=losses.m1_loss m2=losses.m2_loss d1=losses.d1_loss d2=losses.d2_loss
    for (k, v) in pairs(losses)
        name = string(k)
        log_value(logger, "loss/$name", v, step=iter)
    end
end

#############################################################
# Top-level training function
#############################################################

"""
    train!(trainer, train_args, logdir = "./logs")

Run Echo training loop with `trainer`.

Returns a `Channel` immediately and runs as a separate task, pushing intermediate
results into the channel as they are produced. This allows checkpoint saving logic
to be handled outside the function.

# Parameters
- Trainer (`trainer`): EchoTrainer holding training state.
- Training args (`train_args`): NamedTuple of `train_kwargs` from YAML experiment config.
- Log directory (`logdir`): Location to write tensorboard logs for this run
"""
function train!(trainer::EchoTrainer, train_args, logdir="./tensorboard_logs")
    Channel() do channel
        bps = train_args.bits_per_symbol
        nagents = length(trainer.agents)
        if trainer.roundtrip
            SNR_db = get_optimal_SNR_for_BER_roundtrip(train_args.target_ber, bps)
        else
            SNR_db = get_optimal_SNR_for_BER(train_args.target_ber, bps)
        end
        @info "Training at $(SNR_db) dB SNR"
        # Per-model and separate mu & log_std optimizers for individual learning rates
        optims = get_optimisers(trainer.agents, get_optimiser_type(train_args.optimiser))
        schedules = get_schedules(optims, get_schedule_type(train_args.schedule.type), train_args.schedule)
        # Setup progress logging
        is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
        p = Progress(train_args.num_iterations_train, dt=0.25, output=stderr, enabled=!is_logging(stderr))
        generate_showvalues(iter, losses, results, info) = () -> [
            (:iter, iter),
            (:losses, losses),
            (:bers, join([@sprintf("%0.4f", x) for x in _ensemble_train_bers(results[maximum(keys(results))][:ber])], ", ")),
            (:info, info)
        ]
        info = ""
        t0 = time()
        results = Dict{Int, Any}()
        # Setup TensorBoard logging
        lg = TBLogger(logdir, min_level=Logging.Info)
        with_logger(lg) do
            # Run training loop
            for iter in 1:train_args.num_iterations_train
                if iter == 1
                    val_dict = validate(trainer, train_args.len_preamble_eval, train_args.verbose)
                    val_dict[:t_elapsed] = time() - t0
                    results[0] = val_dict
                end
                # Select participating agents
                if train_args.protocol ∈ [GP, LP] && nagents == 2
                    idx1 = 1; idx2 = 2
                else
                    idx1, idx2 = sample(1:nagents, 2, replace=false)
                end
                a1, a2 = trainer.agents[idx1], trainer.agents[idx2]
                o1, o2 = optims[idx1], optims[idx2]
                eavesdroppers = trainer.agents[ones(Bool, nagents) .&& (1:nagents .!= idx1) .&& (1:nagents .!= idx2)]
                # TODO: still necessary?
                if train_args.protocol ∈ [GP, LP]
                    sim_models = filter(isneural, [a1.mod, a2.demod])
                else
                    sim_models = filter(isneural, vcat([a.mod for a in (a1, a2)], [a.demod for a in (a1, a2)]))
                end
                # Run simulation and update
                simulator = trainer.simulator_class(a1, a2, bps, train_args.len_preamble, iscuda(trainer.agents[1]))
                newoptims, newa12, losses... = trainer.update_fn!(
                    simulator, eavesdroppers, SNR_db,
                    (o1, o2), (a1, a2), train_args.verbose
                )
                # Run self-play if enabled
                a1, a2 = newa12
                if a1.self_play
                    a1 = update_self_play!(a1, train_args, bps, SNR_db, iter)
                end
                if a2.self_play
                    a2 = update_self_play!(a2, train_args, bps, SNR_db, iter)
                end
                newa12 = (a1, a2)
                # Run partner model updates if enabled
                a1, a2 = newa12
                if a1.use_prtnr_model && train_args.partner_modeling.warmup < iter
                    a1 = update_from_partner_model!(a1, train_args, iter)
                end
                if a2.use_prtnr_model && train_args.partner_modeling.warmup < iter
                    a2 = update_from_partner_model!(a2, train_args, iter)
                end
                newa12 = (a1, a2)
                # Replace updated optimisers and agents in lists
                optims[idx1] = newoptims[1]; optims[idx2] = newoptims[2]
                trainer.agents[idx1] = newa12[1]; trainer.agents[idx2] = newa12[2]
                # Log losses and learning rates
                log_losses(lg, losses, iter)
                log_learning_rate(lg, get_true_lr(schedules[1], optims[1], iter), iter)
                # Handle learning rate scheduling and restarts
                optims = [step!(opt, sch, a, iter) for (opt, sch, a) in zip(optims, schedules, trainer.agents)]
                # Validate
                if iter % train_args.stats_every_train == 0 || iter == train_args.num_iterations_train
                    val_dict = validate(trainer, train_args.len_preamble_eval, train_args.verbose)
                    val_dict[:t_elapsed] = time() - t0
                    val_dict[:losses] = losses
                    results[iter] = val_dict
                    valbers = _ensemble_train_bers(val_dict[:ber])
                    @info "ber" ht1=valbers[1] ht2=valbers[2] rt=valbers[3] log_step_increment=0
                    divlosses = _diversity_losses(trainer.agents)
                    @info "loss" m1_div=divlosses[1] m2_div=divlosses[2] log_step_increment=0
                end
                if iter % train_args.checkpoint_every_train == 0
                    put!(channel, results)
                    info = "Checkpointed @$(iter)"
                end
                ProgressMeter.next!(p, showvalues=generate_showvalues(iter, losses, results, info))
            end
        end
        # Print summary statistics
        elapsed = round(time() - t0, digits=1)
        fber = round.(_ensemble_train_bers(results[train_args.num_iterations_train][:ber]), sigdigits=3)
        fber = "HT1: $(fber[1]) | HT2: $(fber[2]) | RT: $(fber[3])"
        println("Finished in $elapsed seconds with final BERs $fber")
        # Send final results to controlling task
        put!(channel, results)
    end
end


end