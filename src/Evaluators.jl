module Evaluators
export Evaluator, MetaEvaluator, evaluate

using ..LookupTableUtils
using ..Agents: iscuda
using ..Simulators


struct Evaluator
    simulator_class
    agent1
    agent2
    roundtrip::Bool
end

struct MetaEvaluator
    simulator_class
    agent1_sampler
    agent2_sampler
end


function evaluate(ev::Evaluator; len_preamble)
    if ev.roundtrip
        test_SNRs = get_test_SNR_dbs_roundtrip(ev.agent1.bits_per_symbol)
    else
        test_SNRs = get_test_SNR_dbs(ev.agent1.bits_per_symbol)
    end
    simulator = ev.simulator_class(ev.agent1, ev.agent2, ev.agent1.bits_per_symbol, len_preamble, iscuda(ev.agent1))
    ber_array = Array{Float32}(undef, (3, length(test_SNRs)))
    for (i, snr) in enumerate(test_SNRs)
        results = compute_ber_htrt(simulator, snr)
        ber_array[:, i] .= results
    end
    ber_array
end


"""
Shared preamble update logic
"""
function update_shared_preamble_sgd!(a1, a2, len_preamble, SNR_db,)
    bits_per_symbol = a1.bits_per_symbol
    local d1_loss, d2_loss, m1_loss, m2_loss
    grads = Flux.gradient(all_params) do
        res = simulate_round_trip(a1, a2, bits_per_symbol, len_preamble, SNR_db)
        target = symbols_to_onehot(res.preamble)
        if !isclassic(a1.demod)
            # Train demod of a1 agent
            d1_loss = loss(a1.demod, logits=res.d1_logits, target=target)
        end
        if !isclassic(a2.demod)
            # Train demod of a2 agent
            d2_loss = loss(a2.demod, logits=res.d2_logits, target=target)
        end
        if !isclassic(a1.mod)
            # Train mod of a1 agent
            m1_loss = loss(a1.mod, symbols=res.preamble, received_symbols=res.d1_rt_symbs, actions=res.m1_actions)
        end
        if !isclassic(a2.mod)
            # Train mod of a2 agent
            m2_loss = loss(a2.mod, symbols=res.preamble, received_symbols=res.d2_rt_symbs, actions=res.m2_actions)
        end
        loss_sum  = m1_loss + m2_loss + d1_loss + d2_loss
        loss_sum  # Need to sum all losses to force backprop through all models
    end
    for m in [a1.mod, a1.demod, a2.mod, a2.demod]
        if isclassic(m)
            continue
        end
        if ismod(m)
            # Update μ parameters
            ps = Flux.params(m.μ)
            opt = Descent(m.lr_dict.mu)
            for p in ps
                Flux.update!(opt, p, grads[p])
            end
            # Update log_std parameters
            ps = Flux.params(m.log_std)
            opt = Descent(m.lr_dict.std)
            for p in ps
                Flux.update!(opt, p, grads[p])
            end
        else
            # Update demod parameters
            ps = Flux.params(m)
            opt = Descent(m.lr)
            for p in ps
                Flux.update!(opt, p, grads[p])
            end
        end
    end
    (; m1_loss => m1_loss, m2_loss => m2_loss, d1_loss => d1_loss, d2_loss => d2_loss)
end

function simulate_inner_loop(sim::SharedPreambleSimulator, num_iterations, len_preamble, SNR_db, stats_every::Integer=10)
    ber_array = Array{Float32, 2}(undef, (4, Int(num_iterations / stats_every) + 1))
    bers = compute_ber_htrt(sim, SNR_db)
    ber_array[:, 1] = Float32[0, bers...]

    for i in 1:num_iterations
        update_shared_preamble_sgd!(sim.agent1, sim.agent2, len_preamble, SNR_db)
        if i % stats_every == 0 || i == num_iterations
            bers = compute_ber_htrt(sim, SNR_db)
            ber_array[:, i + 1] = Float32[i, bers...]
        end
    end
    ber_array
end

function evaluate(ev::MetaEvaluator; num_iterations, len_preamble, SNR_db, num_trials=10, stats_every=1)
    ber_array = Array{Float32}(undef, (4, Int(num_iterations / stats_every) + 1, num_trials))
    for i in 1:num_trials
        agent1 = rand(ev.agent1_sampler)
        agent2 = rand(ev.agent2_sampler)
        simulator = ev.simulator_class(agent1, agent2, agent1.bits_per_symbol, 10000)
        results = simulate_inner_loop(simulator, num_iterations, len_preamble, SNR_db, stats_every)
        ber_array[:, :, i] = results
    end
    ber_array
end

end