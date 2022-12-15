module EchoTrainers
export EchoTrainer, train!

using Flux
using Flux.Optimise
using ProgressMeter
using Statistics
using Printf

using ..Agents
using ..ModulationModels
using ..Simulators
using ..Evaluators
using ..LookupTableUtils
using ..FluxUtils


struct EchoTrainer
    tx_agent::Union{MixedAgent, ClassicAgent, NeuralAgent}
    rx_agent::Union{MixedAgent, ClassicAgent, NeuralAgent}
    # Store simulator_class instead of simulator to allow sampling of agents
    simulator_class::Type
    evaluator_class::Type
    update_fn!::Function
    roundtrip::Bool
end

function EchoTrainer(tx_agent, rx_agent; shared_or_grad::Bool, roundtrip::Bool)
    if roundtrip
        simulator = shared_or_grad ? SharedPreambleSimulator : PrivatePreambleSimulator
        update_fn! = shared_or_grad ? update_shared_preamble! : update_private_preamble!
    else
        simulator = shared_or_grad ? GradientPassingSimulator : LossPassingSimulator
        update_fn! = shared_or_grad ? update_gradient_passing! : update_loss_passing!
    end
    EchoTrainer(tx_agent, rx_agent, simulator, Evaluator, update_fn!, roundtrip)
end


"""
Collect non-classic models with optimisers and both individual & collective parameters
"""
function get_optimisers_params(agents, optimiser=Flux.Adam)
    optims = IdDict()
    indiv_params = IdDict()
    all_params, _ = multi_agent_params(agents)
    optim_models = []
    for a in agents
        if isneural(a.mod)
            push!(optim_models, a.mod)
            optims[a.mod.μ] = optimiser(a.mod.lr_dict.mu)
            indiv_params[a.mod.μ] = Flux.params(a.mod.μ)
            optims[a.mod.log_std] = optimiser(a.mod.lr_dict.std)
            indiv_params[a.mod.log_std] = Flux.params(a.mod.log_std)
        end
        if isneural(a.demod)
            push!(optim_models, a.demod)
            optims[a.demod] = optimiser(a.demod.lr)
            indiv_params[a.demod] = Flux.params(a.demod)
        end
    end
    (;optims=optims, optimised_models=optim_models, all_params=all_params, indiv_params=indiv_params)
end


"""
Apply optimiser updates to models based on grads
"""
function apply_model_updates!(grads, optims, models, indiv_params; update_log_std::Bool=true)
    for m in models
        if ismod(m)
            # Update μ parameters
            ps = indiv_params[m.μ]
            opt = optims[m.μ]
            for p in ps
                Flux.update!(opt, p, grads[p])
            end
            # Update log_std parameters
            if update_log_std
                ps = indiv_params[m.log_std]
                opt = optims[m.log_std]
                for p in ps
                    Flux.update!(opt, p, grads[p])
                end
            end
        else
            # Update demod parameters
            ps = indiv_params[m]
            opt = optims[m]
            for p in ps
                Flux.update!(opt, p, grads[p])
            end
        end
    end
end


"""
Calculate current non-exploration BER across SNRs
"""
function validate(trainer::EchoTrainer, len_preamble, verbose::Bool)
    evaluator = trainer.evaluator_class(trainer.simulator_class, trainer.tx_agent, trainer.rx_agent, trainer.roundtrip)
    t0 = time()
    ber_array = evaluate(evaluator, len_preamble=len_preamble)
    if verbose
        println("$(time() - t0) seconds to evaluate")
        println("Mean BER ", mean(ber_array, dims=2)[3])
    end
    val_dict = Dict(:ber => ber_array,
                    :kwargs => (; tx=get_kwargs(trainer.tx_agent),
                                  rx=get_kwargs(trainer.rx_agent)))
end


"""
Gradient Passing update logic
"""
function update_gradient_passing!(simulator, SNR_db,
                                  optims, models, all_params, indiv_params,
                                  verbose)
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
    apply_model_updates!(grads, optims, models, indiv_params, update_log_std=false)
    (; loss = loss,)
end


"""
Loss Passing update logic
"""
function update_loss_passing!(simulator, SNR_db,
                              optims, models, all_params, indiv_params,
                              verbose)
    a1 = simulator.agent1
    a2 = simulator.agent2
    m1_loss = d2_loss = 0
    local res
    grads = Flux.gradient(all_params) do
        res = simulate(simulator, SNR_db, explore=true)
        target = symbols_to_onehot(res.preamble)
        if !isclassic(a2.demod)
            # Train demod
            d2_loss = loss(a2.demod, logits=res.d2_logits, target=target)
        end
        if !isclassic(a1.mod)
            # Train mod
            m1_loss = loss(a1.mod, symbols=res.preamble, received_symbols=res.d2_symbs, actions=res.m1_actions)
        end
        loss_sum = m1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), d2 loss: $(d2_loss)")
    end
    apply_model_updates!(grads, optims, models, indiv_params)
    (; m1_loss = m1_loss, d2_loss = d2_loss)
end


"""
Shared preamble update logic
"""
function update_shared_preamble!(simulator, SNR_db,
                                 optims, models, all_params, indiv_params,
                                 verbose)
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
    apply_model_updates!(grads, optims, models, indiv_params)
    (; m1_loss = m1_loss, m2_loss = m2_loss, d1_loss = d1_loss, d2_loss = d2_loss)
end


"""
Private preamble update logic
"""
function update_private_preamble!(simulator, SNR_db,
                                 optims, models, all_params, indiv_params,
                                 verbose)
    a1 = simulator.agent1
    a2 = simulator.agent2
    d1_loss = d2_loss = m1_loss = m2_loss = 0
    local res
    grads = Flux.gradient(all_params) do
        res = simulate(simulator, SNR_db, explore=true)
        target1 = symbols_to_onehot(res.preamble1)
        target2 = symbols_to_onehot(res.preamble2)
        if !isclassic(a1.demod)
            # Train demod of a1 agent
            d1_loss = loss(a1.demod, logits=res.d1_rt_logits, target=target1)
        end
        if !isclassic(a2.demod)
            # Train demod of a2 agent
            d2_loss = loss(a2.demod, logits=res.d2_rt_logits, target=target2)
        end
        if !isclassic(a1.mod)
            # Train mod of a1 agent
            m1_loss = loss(a1.mod, symbols=res.preamble1, received_symbols=res.d1_rt_symbs, actions=res.m1_actions)
        end
        if !isclassic(a2.mod)
            # Train mod of a2 agent
            m2_loss = loss(a2.mod, symbols=res.preamble2, received_symbols=res.d2_rt_symbs, actions=res.m2_actions)
        end
        loss_sum = m1_loss + m2_loss + d1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    if verbose
        println("m1 loss: $(m1_loss), m2 loss: $(m2_loss), d1 loss: $(d1_loss), d2 loss: $(d2_loss)")
    end
    apply_model_updates!(grads, optims, models, indiv_params)
    (; m1_loss = m1_loss, m2_loss = m2_loss, d1_loss = d1_loss, d2_loss = d2_loss)
end


"""
Run Echo training loop with trainer
"""
function train!(trainer::EchoTrainer, train_args)
    Channel() do channel
        bps = train_args.bits_per_symbol
        if trainer.roundtrip
            SNR_db = get_optimal_SNR_for_BER_roundtrip(train_args.target_ber, bps)
        else
            SNR_db = get_optimal_SNR_for_BER(train_args.target_ber, bps)
        end
        println("Training at $(SNR_db) dB SNR")
        # Per-model and separate mu & log_std optimizers for individual learning rates
        simulator = trainer.simulator_class(trainer.tx_agent, trainer.rx_agent, bps, train_args.len_preamble, iscuda(trainer.tx_agent))
        optims, optim_models, all_params, indiv_params = get_optimisers_params(
            [trainer.tx_agent, trainer.rx_agent], get_optimiser_type(train_args.optimiser)
        )
        # Run training loop
        is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
        p = Progress(train_args.num_iterations_train, dt=0.25, output=stderr, enabled=!is_logging(stderr))
        generate_showvalues(iter, losses, results, info) = () -> [
            (:iter, iter),
            (:losses, losses),
            (:bers, join([@sprintf("%0.4f", x) for x in results[maximum(keys(results))][:ber][:, 5]], ", ")),
            (:info, info)
        ]
        info = ""
        t0 = time()
        results = Dict{Int, Any}()
        for iter in 1:train_args.num_iterations_train
            if iter == 1
                val_dict = validate(trainer, train_args.len_preamble_eval, train_args.verbose)
                val_dict[:t_elapsed] = time() - t0
                results[0] = val_dict
            end
            losses = trainer.update_fn!(
                simulator, SNR_db,
                optims, optim_models, all_params, indiv_params,
                train_args.verbose
            )
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
        fber = round.(results[train_args.num_iterations_train][:ber][:, 5], sigdigits=3)
        fber = "HT1: $(fber[1]) | HT2: $(fber[2]) | RT: $(fber[3])"
        println("Finished in $elapsed seconds with final BERs $fber")
        # Send final results to controlling task
        put!(channel, results)
    end
end


end