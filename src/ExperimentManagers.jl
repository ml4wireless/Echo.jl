# Experiment Management
module ExperimentManagers
export ExperimentConfig
export run_experiment, run_experiment_meta
export get_mod, get_demod, get_opp_mod, get_opp_demod, reconstruct_agents

using BSON
using Random
using Accessors

using ..ModulationModels
using ..Agents
using ..MetaTrainersSharedPreamble
using ..EchoTrainers
using ..Protocols
using ..ResultsUtils: loadresults

import Base.show


agent_type_map = Dict(
    "neuraldemod" => NeuralDemod,
    "neuralmod" => NeuralMod,
    "classicdemod" => ClassicDemod,
    "classicmod" => ClassicMod
)


struct ExperimentConfig
    config::NamedTuple
    results_dir::String
    bps::Int
end

ExperimentConfig(config::NamedTuple, results_dir::String="./results") = ExperimentConfig(config, results_dir, config.train_kwargs.bits_per_symbol)

Base.show(io::IO, cfg::ExperimentConfig) = print(io, "ExperimentConfig($(cfg.config.experiment_type), $(cfg.config.experiment_id), $(cfg.results_dir))")


"""
Run training for shared preamble experiment
"""
function run_experiment(expmt::ExperimentConfig; save_results::Bool=true, norerun::Bool=false)
    println("Running experiment $(expmt.config.experiment_type)->$(expmt.config.experiment_id)")
    savefile = joinpath(expmt.results_dir, expmt.config.experiment_type, expmt.config.experiment_id * ".bson")
    mkpath(dirname(savefile))
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
    esp_or_epp = protocol ∈ [ESP, EPP]
    gp_or_esp = protocol ∈ [GP, ESP]
    gp_or_lp = protocol ∈ [GP, LP]
    tx_agent_sampler = AgentSampler(
        mod_class=agent_type_map[expmt.config.agent_types.tx_mod * "mod"],
        demod_class=gp_or_lp ? nothing : agent_type_map[expmt.config.agent_types.tx_demod * "demod"],
        mod_kwargs=expmt.config.neural_mod_kwargs,
        demod_kwargs=gp_or_lp ? (;) : expmt.config.neural_demod_kwargs;
        expmt.config.classic_agent_sampler_kwargs...
    )
    rx_agent_sampler = AgentSampler(
        mod_class=gp_or_lp ? nothing : agent_type_map[expmt.config.agent_types.rx_mod * "mod"],
        demod_class=agent_type_map[expmt.config.agent_types.rx_demod * "demod"],
        mod_kwargs=gp_or_lp ? (;) : expmt.config.neural_mod_2_kwargs,
        demod_kwargs=expmt.config.neural_demod_2_kwargs;
        expmt.config.classic_agent_sampler_kwargs...
    )
    trainer = EchoTrainer(rand(tx_agent_sampler), rand(rx_agent_sampler), shared_or_grad=gp_or_esp, roundtrip=esp_or_epp)
    channel = EchoTrainers.train!(trainer, expmt.config.train_kwargs,)
    results = nothing
    for msg in channel
        results = msg
        last_iter = maximum(collect(keys(results)))
        if save_results
            # println("Saving checkpoint @$(last_iter) to $(savefile)...")
            bson(savefile, config=expmt.config, results=results)
            # Load with BSON.load(savefile, Flux)
        else
            # println("Iteration $(last_iter) finished...")
        end
    end
    results, (trainer.tx_agent, trainer.rx_agent)
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


get_mod(expmt::ExperimentConfig) = NeuralMod(;expmt.config.neural_mod_kwargs...)

function get_opp_mod(expmt::ExperimentConfig)
    if expmt.config.agent_types.rx_mod == "neural"
        mod = NeuralMod(;expmt.config.neural_mod_2_kwargs...)
    else
        sampler = ClassicAgentSampler(;expmt.config.classic_agent_sampler_kwargs...)
        mod = rand(sampler).mod
    end
    mod
end

function get_demod(expmt::ExperimentConfig)
    if expmt.config.agent_types.tx_demod == "neural"
        demod = NeuralDemod(;expmt.config.neural_demod_kwargs...)
    else
        sampler = ClassicAgentSampler(expmt.config.classic_agent_sampler_kwargs...)
        demod = rand(sampler).demod
    end
    demod
end

function get_opp_demod(expmt::ExperimentConfig)
    if expmt.config.agent_types.rx_demod == "neural"
        demod = NeuralDemod(;expmt.config.neural_demod_kwargs...)
    else
        sampler = ClassicAgentSampler(;expmt.config.classic_agent_sampler_kwargs...)
        demod = rand(sampler).demod
    end
    demod
end

function reconstruct_agents(config, results, iter)
    kwargs = results[0][:kwargs]
    m1_type = agent_type_map[config.agent_types.tx_mod * "mod"]
    m2_type = agent_type_map[config.agent_types.rx_mod * "mod"]
    d1_type = agent_type_map[config.agent_types.tx_demod * "demod"]
    d2_type = agent_type_map[config.agent_types.rx_demod * "demod"]
    m1 = m1_type(;kwargs.tx.mod...)
    d1 = d1_type(;kwargs.tx.demod...)
    m2 = m2_type(;kwargs.rx.mod...)
    d2 = d2_type(;kwargs.rx.demod...)
    a1 = Agent(m1, d1)
    a2 = Agent(m2, d2)
    a1, a2
end

end