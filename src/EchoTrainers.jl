module EchoTrainers
export EchoTrainer, train!

using Flux
using Optimisers
using ProgressMeter
using Statistics
using StatsBase: sample
using Printf

using ..Agents
using ..ModulationModels
using ..Simulators
using ..Evaluators
using ..LookupTableUtils
using ..FluxUtils
using ..Protocols


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


"""
Initialize optimisers for neural models.

Returns (optims, optimised_models) tuple
- `optims`: list of optimisers for every trained model
- `optimised_models`: list of optimised models in matching order
"""
function get_optimisers(agents, optimiser=Optimisers.Adam)
    optims = []
    optim_models = []
    for a in agents
        if isneural(a.mod)
            push!(optim_models, a.mod)
            rule = optimiser(a.mod.lr_dict.mu)
            state = Optimisers.setup(rule, a.mod)
            # Separate η for log_std
            Optimisers.adjust!(state[:log_std], a.mod.lr_dict.std)
            push!(optims, state)
        end
        if isneural(a.demod)
            push!(optim_models, a.demod)
            rule = optimiser(a.demod.lr)
            state = Optimisers.setup(rule, a.demod)
            push!(optims, state)
        end
    end
    (;optims=optims, optimised_models=optim_models)
end


"""
Apply optimiser updates to models based on grads.

Uses in-place version of update! which is more efficient, but mutates the
original model, and returns a new model which is the only copy guaranteed
to be correct.

Returns:
- New optimiser states `newoptims`: Array of updated optimiser state, rule trees
- New optimised models `newmodels`: Array of updated models which must be
                                    assigned back to any higher level containers
"""
function apply_model_updates!(optims, models, grads; update_log_std::Bool=true)
    newoptims, newmodels = [], []
    for (o, m, g) in zip(optims, models, grads)
        if !update_log_std && hasfield(m, :log_std)
            Optimisers.freeze!(o[:log_std])
        end
        newo, newm = Optimisers.update!(o, m, g)
        push!(newoptims, newo)
        push!(newmodels, newm)
    end
    newoptims, newmodels
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
    a1 = simulator.agent1
    a2 = simulator.agent2
    loss = 0
    local res
    grads = Flux.gradient(all_params) do
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble)
        loss = loss_crossentropy(logits=res.d2_logits, target=target)
    end
    if verbose
        println("loss: $(loss)")
    end
    optims, optim_models = apply_model_updates!(optims, optim_models, grads, update_log_std=false)
    (; optims, optim_models, loss)
end


"""
Loss Passing update logic
"""
function update_loss_passing!(simulator, eavesdroppers, SNR_db,
                              optims, optim_models, verbose)
    a1 = simulator.agent1
    a2 = simulator.agent2
    m1_loss = d2_loss = 0
    local res
    grads = Flux.gradient(all_params) do
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble)
        if isneural(a2.demod)
            # Train demod
            d2_loss = loss(a2.demod, logits=res.d2_logits, target=target)
        end
        if isneural(a1.mod)
            # Train mod
            m1_loss = loss(a1.mod, symbols=res.preamble, received_symbols=res.d2_symbs, actions=res.m1_actions)
        end
        loss_sum = m1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), d2 loss: $(d2_loss)")
    end
    optims, optim_models = apply_model_updates!(optims, optim_models, grads)
    (; optims, optim_models, m1_loss, d2_loss)
end


"""
Shared preamble update logic
"""
function update_shared_preamble!(simulator, eavesdroppers, SNR_db,
                                 optims, optim_models, verbose)
    a1 = simulator.agent1
    a2 = simulator.agent2
    d1_loss = d2_loss = m1_loss = m2_loss = 0
    local res
    grads = Flux.gradient(all_params) do
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble1)
        if isneural(a1.demod)
            # Train demod of a1 agent
            d1_loss = loss(a1.demod, logits=res.d1_logits, target=target)
        end
        if isneural(a2.demod)
            # Train demod of a2 agent
            d2_loss = loss(a2.demod, logits=res.d2_logits, target=target)
        end
        if isneural(a1.mod)
            # Train mod of a1 agent
            m1_loss = loss(a1.mod, symbols=res.preamble1, received_symbols=res.d1_rt_symbs, actions=res.m1_actions)
        end
        if isneural(a2.mod)
            # Train mod of a2 agent
            m2_loss = loss(a2.mod, symbols=res.preamble2, received_symbols=res.d2_rt_symbs, actions=res.m2_actions)
        end
        loss_sum = m1_loss + m2_loss + d1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), m2 loss: $(m2_loss), d1 loss: $(d1_loss), d2 loss: $(d2_loss)")
    end
    optims, optim_models = apply_model_updates!(optims, optim_models, grads)
    (; optims, optim_models, m1_loss, m2_loss, d1_loss, d2_loss)
end


"""
Private preamble update logic
"""
function update_private_preamble!(simulator, eavesdroppers, SNR_db,
                                  optims, optim_models, verbose)
    a1 = simulator.agent1
    a2 = simulator.agent2
    d1_loss = d2_loss = m1_loss = m2_loss = 0
    local res
    grads = Flux.gradient(all_params) do
        res = simulate(simulator, SNR_db, explore=true)
        target1 = symbols_to_onehot(res.preamble1)
        target2 = symbols_to_onehot(res.preamble2)
        if isneural(a1.demod)
            # Train demod of a1 agent
            d1_loss = loss(a1.demod, logits=res.d1_rt_logits, target=target1)
        end
        if isneural(a2.demod)
            # Train demod of a2 agent
            d2_loss = loss(a2.demod, logits=res.d2_rt_logits, target=target2)
        end
        if isneural(a1.mod)
            # Train mod of a1 agent
            m1_loss = loss(a1.mod, symbols=res.preamble1, received_symbols=res.d1_rt_symbs, actions=res.m1_actions)
        end
        if isneural(a2.mod)
            # Train mod of a2 agent
            m2_loss = loss(a2.mod, symbols=res.preamble2, received_symbols=res.d2_rt_symbs, actions=res.m2_actions)
        end
        loss_sum = m1_loss + m2_loss + d1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), m2 loss: $(m2_loss), d1 loss: $(d1_loss), d2 loss: $(d2_loss)")
    end
    optims, optim_models = apply_model_updates!(optims, optim_models, grads)
    (; optims, optim_models, m1_loss, m2_loss, d1_loss, d2_loss)
end


_ensemble_train_bers(bers) = mean(bers, dims=3)[:, 5]

"""
Run Echo training loop with trainer
"""
function train!(trainer::EchoTrainer, train_args)
    Channel() do channel
        bps = train_args.bits_per_symbol
        nagents = length(trainer.agents)
        if trainer.roundtrip
            SNR_db = get_optimal_SNR_for_BER_roundtrip(train_args.target_ber, bps)
        else
            SNR_db = get_optimal_SNR_for_BER(train_args.target_ber, bps)
        end
        println("Training at $(SNR_db) dB SNR")
        # Per-model and separate mu & log_std optimizers for individual learning rates
        optims, optim_models = get_optimisers(trainer.agents, get_optimiser_type(train_args.optimiser))
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
            newoptims, newa12, losses = trainer.update_fn!(
                simulator, eavesdroppers, SNR_db,
                (o1, o2), (a1, a2), train_args.verbose
            )
            optims[idx1] = newoptims[1]; optims[idx2] = newoptims[2]
            trainer.agents[idx1] = newa12[1]; trainer.agents[idx2] = newa12[2]
            # Validate
            if iter % train_args.stats_every_train == 0 || iter == train_args.num_iterations_train
                val_dict = validate(trainer, train_args.len_preamble_eval, train_args.verbose)
                val_dict[:t_elapsed] = time() - t0
                val_dict[:losses] = losses
                results[iter] = val_dict
            end
            if iter % train_args.checkpoint_every_train == 0
                put!(channel, results)
                info = "Checkpointed @$(iter)"
            end
            ProgressMeter.next!(p, showvalues=generate_showvalues(iter, losses, results, info))
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