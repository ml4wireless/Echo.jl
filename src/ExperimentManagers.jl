# Experiment Management
module ExperimentManagers
export ExperimentConfig
export run_experiment, run_experiment_meta
export get_mod, get_demod, get_agent, reconstruct_agents

using Accessors
using BSON
using Crayons
using CUDA: @allowscalar
using Flux: gpu
using Random

using ..ModulationModels
using ..Agents
using ..MetaTrainersSharedPreamble
using ..EchoTrainers
using ..Protocols
using ..ResultsUtils: loadresults

using Infiltrator

import Base.show


agent_type_map = Dict(
    "neuraldemod" => NeuralDemod,
    "neuralmod" => NeuralMod,
    "gnnmod" => GNNMod,
    "classicdemod" => ClassicDemod,
    "classicmod" => ClassicMod,
    "clusterdemod" => ClusteringDemod,
    "clusteringdemod" => ClusteringDemod,
    "nonemod" => nothing,
    "nonedemod" => nothing,
)

"""
    ExperimentConfig(config, results_dir = "./results"; cuda = false)

ExperimentConfig manages configuration for Echo experiments.

# Parameters
- Run configuration (`config`): NamedTuple loaded from a YAML experiment definition file with `loadconfig`.
- Output directory (`results_dir`): Top level directory to save run results in, further organized by `experiment_type`.
- Use GPU (`cuda`): Whether to run experiment on GPU.
"""
struct ExperimentConfig
    config::NamedTuple
    results_dir::String
    logging_dir::String
    bps::Int
    cuda::Bool
end

ExperimentConfig(config::NamedTuple, results_dir::String="./results";
                 cuda::Bool=false, logging_dir::String="./tensorboard_logs") = (
    ExperimentConfig(config, results_dir, logging_dir, config.train_kwargs.bits_per_symbol, cuda)
)

Base.show(io::IO, cfg::ExperimentConfig) = print(io,
    Crayon(bold=true, foreground=:light_blue), "ExperimentConfig", Crayon(reset=true),
    "($(cfg.config.experiment_type), $(cfg.config.experiment_id), $(cfg.results_dir))",
)


"""
    run_experiment(expmt; save_results = true, rorerun = false)

Run training for shared preamble experiment

# Parameters
- Experiment config (`expmt`): ExperimentConfig defining run parameters.
- Save results (`save_results`): Whether to save results to disk as well as return from this function.
- No rerun (`norerun`): If true, skip running this experiment if a results file already exists.
"""
function run_experiment(expmt::ExperimentConfig; save_results::Bool=true, norerun::Bool=false)
    @info "Running experiment $(expmt.config.experiment_type)->$(expmt.config.experiment_id)"
    savefile = joinpath(expmt.results_dir, expmt.config.experiment_type, expmt.config.experiment_id * ".bson")
    mkpath(dirname(savefile), mode=0o755)
    @info "Saving results to $(savefile)"
    if norerun && isfile(savefile)
        oldcfg = loadresults(savefile).config
        if oldcfg == expmt.config
            @info "Results file already exists with same config, skipping this experiment" savefile
            return nothing, (nothing, nothing)
        else
            @warn "Overwriting existing results file with results from new config" savefile
        end
    end
    seed = expmt.config.seed
    Random.seed!(seed)
    # Set parameters based on protocol
    protocol = expmt.config.train_kwargs.protocol
    agents = Agent[]
    for agent_type in expmt.config.agent_types
        mclass = agent_type_map[agent_type.mod * "mod"]
        dclass = agent_type_map[agent_type.demod * "demod"]
        pmclass = agent_type_map["neuraldemod"]
        mkwargs = agent_type.alt_kwargs ? expmt.config.neural_mod_2_kwargs : expmt.config.neural_mod_kwargs
        dkwargs = agent_type.alt_kwargs ? expmt.config.neural_demod_2_kwargs : expmt.config.neural_demod_kwargs
        # Self-play can only be enabled for neural agents
        self_play = agent_type.mod == "neural" || agent_type.demod == "neural" ? agent_type.self_play : false
        if self_play && (mclass === nothing || dclass === nothing)
            @error "Self-play can only be run for agents with both mod ($(mclass !== nothing)) and demod ($(dclass !== nothing)), continuing with self-play disabled."
            self_play = false
        end
        # Partner modeling can only be enabled for neural mods
        use_prtnr_model = agent_type.mod == "neural" ? agent_type.use_prtnr_model : false
        pmkwargs = use_prtnr_model ? expmt.config.partner_model_kwargs : (;)
        pmclass = use_prtnr_model ? pmclass : nothing
        sampler = AgentSampler(
            mod_class=mclass, demod_class=dclass,
            mod_kwargs=mclass === nothing ? (;) : mkwargs,
            demod_kwargs=dclass === nothing ? (;) : dkwargs;
            prtnr_model_class=pmclass, prtnr_model_kwargs=pmkwargs,
            self_play=self_play,
            use_prtnr_model=use_prtnr_model,
            expmt.config.classic_agent_sampler_kwargs...,
        )
        for _ in 1:agent_type.count
            push!(agents, rand(sampler))
        end
    end
    if expmt.cuda
        for i in eachindex(agents)
            @allowscalar agents[i] = gpu(agents[i])
        end
    end
    trainer = EchoTrainer(agents, protocol)
    # TensorBoard logger
    logdir = joinpath(expmt.logging_dir, expmt.config.experiment_type, expmt.config.experiment_id)
    @info "Writing tensorboard logs to $logdir"
    # Run training
    channel = EchoTrainers.train!(trainer, expmt.config.train_kwargs, logdir)
    results = nothing
    for msg in channel
        results = msg
        last_iter = maximum(collect(keys(results)))
        if save_results
            bson(savefile, config=expmt.config, results=results)
            # Load with BSON.load(savefile, Flux)
        end
    end
    results, trainer.agents
end


"""
Run meta-training for one mod, shared preamble
"""
function run_experiment_meta(expmt::ExperimentConfig, save_results::Bool=true, results_dir::String="../data")
    println("Running experiment $(expmt.config.experiment_type)->$(expmt.config.experiment_id)")

    seed = expmt.config.seed
    Random.seed!(seed)
    mod = NeuralMod(;expmt.config.neural_mod_kwargs...)
    rx_agent_sampler = AgentSampler(
        mod_class=agent_type_map[expmt.config.agent_types.rx_mod * "mod"],
        demod_class=agent_type_map[expmt.config.agent_types.rx_demod * "demod"],
        mod_kwargs=expmt.config.neural_mod_2_kwargs,
        demod_kwargs=expmt.config.neural_demod_kwargs;
        expmt.config.classic_agent_sampler_kwargs...
    )
    tx_agent_sampler = AgentSampler(
        mod_class=nothing,
        demod_class=agent_type_map[expmt.config.agent_types.tx_demod * "demod"],
        demod_kwargs=expmt.config.neural_demod_kwargs;
        expmt.config.classic_agent_sampler_kwargs...
    )
    trainer = MetaTrainerSharedPreambleMod(mod, tx_agent_sampler, rx_agent_sampler)
    results = train(trainer; expmt.config.train_kwargs...)
    # channel = train(trainer; expmt.config.train_kwargs...)
    # results = nothing
    # while true
    #     try
    #         results = take!(channel)
    #     catch e
    #         if isa(e, InvalidStateException) && e.state != :closed
    #             println("Channel exception: $(e)")
    #         else
    #             rethrow()
    #         end
    #         break
    #     end
    if save_results
        savefile = joinpath(results_dir, expmt.config.experiment_type, expmt.config.experiment_id * ".bson")
        mkpath(dirname(savefile))
        last_iter = maximum(collect(keys(results)))
        println("Saving checkpoint @$(last_iter) to $(savefile)...")
        bson(savefile, config=expmt.config, results=results)
    end
    # end
    println("Finished experiment $(expmt.config.experiment_id)")
    println("=" ^ 40 * "\n")
    results
end


function get_mod(expmt::ExperimentConfig, index::Integer = 1)
    agent_type = expmt.config.agent_types[index]
    mclass = agent_type_map[agent_type.mod * "mod"]
    if mclass === nothing
        return nothing
    end
    mkwargs = agent_type.alt_kwargs ? expmt.config.neural_mod_2_kwargs : expmt.config.neural_mod_kwargs
    mclass(;mkwargs...)
end

function get_demod(expmt::ExperimentConfig, index::Integer = 1)
    agent_type = expmt.config.agent_types[index]
    dclass = agent_type_map[agent_type.demod * "demod"]
    if dclass === nothing
        return nothing
    end
    dkwargs = agent_type.alt_kwargs ? expmt.config.neural_demod_2_kwargs : expmt.config.neural_demod_kwargs
    dclass(;dkwargs...)
end

function get_agent(expmt::ExperimentConfig, index::Integer = 1)
    agent_type = expmt.config.agent_types[index]
    mclass = agent_type_map[agent_type.mod * "mod"]
    dclass = agent_type_map[agent_type.demod * "demod"]
    mkwargs = agent_type.alt_kwargs ? expmt.config.neural_mod_2_kwargs : expmt.config.neural_mod_kwargs
    dkwargs = agent_type.alt_kwargs ? expmt.config.neural_demod_2_kwargs : expmt.config.neural_demod_kwargs
    sampler = AgentSampler(
        mod_class=mclass, demod_class=dclass,
        mod_kwargs=mclass === nothing ? (;) : mkwargs,
        demod_kwargs=dclass === nothing ? (;) : dkwargs;
        expmt.config.classic_agent_sampler_kwargs...
    )
    rand(sampler)
end

function reconstruct_agents(results, iter=0)
    kwargs = results[iter][:kwargs]
    agents = Agent[]
    for agent in values(kwargs)
        push!(agents, Agent(mod=agent.mod, demod=agent.demod))
    end
    agents
end

end