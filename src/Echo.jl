# Top Level Echo Module
module Echo
include("Protocols.jl")
include("ConfigUtils.jl")
include("DataUtils.jl")
include("FluxUtils.jl")
include("ModulationUtils.jl")
include("LookupTableUtils.jl")
include("ModulationModels.jl")
include("Agents.jl")
include("Simulators.jl")
include("Evaluators.jl")
include("EchoTrainers.jl")
include("MetaTrainersSharedPreamble.jl")
include("ResultsUtils.jl")
include("ExperimentManagers.jl")
include("VisualizationUtils.jl")

using Reexport: @reexport
@reexport begin
    using .Protocols
    using .ConfigUtils
    using .DataUtils
    using .FluxUtils
    using .ModulationUtils
    using .LookupTableUtils
    using .ResultsUtils

    using .ModulationModels
    using .Simulators
    using .Evaluators
    using .Agents

    using .EchoTrainers
    using .ExperimentManagers
    using .VisualizationUtils
end

precompile(run_experiment, (ExperimentConfig,))
precompile(EchoTrainers.train!, (EchoTrainer, NamedTuple))
precompile(simulate_half_trip, (NeuralAgent, NeuralAgent, Int, Int, Float32, Bool))
precompile(simulate_half_trip, (NeuralAgent, ClassicAgent, Int, Int, Float32, Bool))
precompile(simulate_round_trip, (NeuralAgent, NeuralAgent, Int, Int, Float32, Bool, Bool, Bool))
precompile(simulate_round_trip, (NeuralAgent, ClassicAgent, Int, Int, Float32, Bool, Bool, Bool))
end
