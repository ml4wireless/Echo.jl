module MetaTrainersSharedPreamble
export MetaTrainerSharedPreambleMod, train!

using Statistics
using Flux
using Flux.Optimise
using Printf
using Accessors

using ..Agents
using ..ModulationModels
using ..Simulators
using ..Evaluators
using ..LookupTableUtils


struct MetaTrainerSharedPreambleMod
    mod::NeuralMod
    tx_agent_sampler::AgentSampler
    rx_agent_sampler::Union{AgentSampler, ClassicAgentSampler}
    simulator_class
    evaluator_class
end

function MetaTrainerSharedPreambleMod(mod, tx_agent_sampler, rx_agent_sampler)
    MetaTrainerSharedPreambleMod(mod, tx_agent_sampler, rx_agent_sampler, SharedPreambleRoundTripSimulator, MetaEvaluator)
end


"""
Run validation during meta-training
"""
function validate(trainer::MetaTrainerSharedPreambleMod,
                  meta_iter, mod_kwargs, num_inner_iterations,
                  len_preamble, SNR_db, num_trials_eval, stats_every_eval,
                  verbose=false, plot_initial=false, t_elapsed=nothing)
    sampler_kwargs = get_kwargs(trainer.tx_agent_sampler)
    println(sampler_kwargs)
    sampler_kwargs = @set sampler_kwargs.mod_class = NeuralMod
    sampler_kwargs = @set sampler_kwargs.mod_kwargs = mod_kwargs
    sampler1 = AgentSampler(;sampler_kwargs...)
    sampler2 = trainer.rx_agent_sampler
    evaluator = trainer.evaluator_class(trainer.simulator_class, sampler1, sampler2)

    t0 = time()
    ber_array = evaluate(evaluator, num_iterations=num_inner_iterations, len_preamble=len_preamble, SNR_db=SNR_db, num_trials=num_trials_eval, stats_every=stats_every_eval)
    println("$(time() - t0) seconds to evaluate")
    if verbose
        elapsed_str = ""
        if t_elapsed !== nothing
            elapsed_str = @sprintf(", %.1f seconds elapsed", t_elapsed)
        end
        println(@sprintf("Meta iter %d%s", meta_iter, elapsed_str))
        println("Mean final BER ", mean(ber_array, dims=3)[4, end])
    end

    val_dict = Dict(:ber => ber_array, :mod_kwargs => mod_kwargs)
    if t_elapsed !== nothing
        val_dict[:t_elapsed] = t_elapsed
    end
    val_dict
end


"""
Run meta-training loop with trainer

### Returns a Channel object which outputs checkpoints pushed by the training coroutine.
"""
function train!(trainer::MetaTrainerSharedPreambleMod;
               bits_per_symbol::Integer,
               num_meta_iterations::Integer,
               target_ber::Real=1f-2,
               meta_lr_dict::NamedTuple,
               len_preamble::Integer=64,
               num_inner_tasks::Integer=10,
               num_inner_iterations::Integer=10,
               first_order::Bool=false,
               inner_step_meta_grad_weighting="last",
               stats_every_train::Integer=100,
               num_trials_eval::Integer=5,
               num_inner_iterations_eval::Integer=100,
               stats_every_eval::Integer=5,
               checkpoint_every_train::Integer=-1,
               verbose::Bool=false,
               plot_initial_eval::Bool=false)
    # Channel() do channel
        SNR_db = get_optimal_SNR_for_BER_roundtrip(target_ber, bits_per_symbol)
        meta_opt = ADAM()
        t0 = time()
        results = Dict()
        for meta_iter in 1:num_meta_iterations
            if meta_iter == 1
                mod_kwargs = deepcopy(get_kwargs(trainer.mod, include_weights=true))
                val_dict = validate(trainer, meta_iter, mod_kwargs, num_inner_iterations_eval, len_preamble, SNR_db, num_trials_eval, stats_every_eval, verbose, plot_initial_eval, time() - t0)
                results[0] = deepcopy(val_dict)
            end
            # put!(channel, results)
            break
        end
    # end
end


end