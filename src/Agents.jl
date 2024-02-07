module Agents
export Agent, ClassicAgent, NeuralAgent, MixedAgent
export ClassicAgentSampler, AgentSampler
export get_kwargs, iscuda

using ..ModulationModels
import .ModulationModels.get_kwargs
import .ModulationModels.iscuda

using Crayons
using Random
import Random.rand
using Flux
using Flux: @functor
import Flux.trainable
import Base.show

using Infiltrator



MaybeCMod = Union{ClassicMod, Nothing}
MaybeNMod = Union{NeuralMod, Nothing}
MaybeMod = Union{NeuralMod, ClassicMod, Nothing}
MaybeCDMod = Union{ClassicDemod, Nothing}
MaybeNDMod = Union{NeuralDemod, Nothing}
MaybeDMod = Union{NeuralDemod, ClassicDemod, ClusteringDemod, Nothing}

get_kwargs(::Nothing; include_weights=false) = (;)
kwarg_types(args) = tuple(pairzip(typeof.(keys(args)), typeof.(values(args)))...)

abstract type Agent end

##############
# Neural agents

struct NeuralAgent <: Agent
    bits_per_symbol::Integer
    mod::MaybeNMod
    demod::MaybeNDMod
    prtnr_model::MaybeNDMod
    self_play::Bool
    use_prtnr_model::Bool
end

@functor NeuralAgent (mod, demod, prtnr_model)

get_kwargs(a::NeuralAgent) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :mod => get_kwargs(a.mod, include_weights=true),
    :demod => get_kwargs(a.demod, include_weights=true),
    :prtnr_model => get_kwargs(a.prtnr_model, include_weights=true),
    :self_play => a.self_play,
    :use_prtnr_model => a.use_prtnr_model,
)


##############
# Mixed agents

struct MixedAgent <: Agent
    bits_per_symbol::Integer
    rotation_deg::Float32
    avg_power::Float32
    mod::MaybeMod
    demod::MaybeDMod
    prtnr_model::MaybeNDMod
    self_play::Bool
    use_prtnr_model::Bool
end

@functor MixedAgent (mod, demod, prtnr_model)

get_kwargs(a::MixedAgent) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :rotation_deg => a.rotation_deg,
    :avg_power => a.avg_power,
    :mod => get_kwargs(a.mod, include_weights=true),
    :demod => get_kwargs(a.demod, include_weights=true),
    :prtnr_model => get_kwargs(a.prtnr_model, include_weights=true),
    :self_play => a.self_play,
    :use_prtnr_model => a.use_prtnr_model,
)


################
# Classic agents

struct ClassicAgent <: Agent
    bits_per_symbol::Integer
    rotation_deg::Float32
    avg_power::Float32
    mod::MaybeCMod
    demod::MaybeCDMod
    self_play::Bool  # Always false
    use_prtnr_model::Bool  # Always false
end

@functor ClassicAgent (mod, demod)

get_kwargs(a::ClassicAgent) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :rotation_deg => a.rotation_deg,
    :avg_power => a.avg_power,
    :mod => get_kwargs(a.mod),
    :demod => get_kwargs(a.demod),
    :self_play => a.self_play,
    :use_prtnr_model => a.use_prtnr_model,
)


################
# Agent Sampling

struct AgentSampler{M <: Modulator, D <: Demodulator}
    bits_per_symbol::Integer
    mod_class::Union{Type{M}, Nothing}
    demod_class::Union{Type{D}, Nothing}
    prtnr_model_class::Union{Type{D}, Nothing}
    mod_kwargs::NamedTuple
    demod_kwargs::NamedTuple
    prtnr_model_kwargs::NamedTuple
    min_rotation_deg::Float32
    max_rotation_deg::Float32
    avg_power::Float32
    self_play::Bool
    use_prtnr_model::Bool
end


get_kwargs(a::AgentSampler) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :mod_class => a.mod_class,
    :demod_class => a.demod_class,
    :prtnr_model_class => a.prtnr_model_class,
    :mod_kwargs => a.mod_kwargs,
    :demod_kwargs => a.demod_kwargs,
    :prtnr_model_kwargs => a.prtnr_model_kwargs,
    :min_rotation_deg => a.min_rotation_deg,
    :max_rotation_deg => a.max_rotation_deg,
    :avg_power => a.avg_power,
    :self_play => a.self_play,
    :use_prtnr_model => a.use_prtnr_model,
)

function AgentSampler(;bits_per_symbol, mod_class, demod_class, prtnr_model_class, mod_kwargs=NamedTuple(), demod_kwargs=NamedTuple(),
                       prtnr_model_kwargs=NamedTuple(), min_rotation_deg=0f0, max_rotation_deg=Float32(2pi), avg_power=1f0,
                       self_play=false, use_prtnr_model=false)
    MC = mod_class === nothing ? Modulator : mod_class
    DC = demod_class === nothing ? Demodulator : demod_class
    AgentSampler{MC, DC}(bits_per_symbol, mod_class, demod_class, prtnr_model_class, mod_kwargs, demod_kwargs, prtnr_model_kwargs,
                         min_rotation_deg, max_rotation_deg, avg_power, self_play, use_prtnr_model)
end

function Random.rand(rng::AbstractRNG, s::AgentSampler)
    # Sample classic's rotation
    rotation_deg = rand(rng) * (s.max_rotation_deg - s.min_rotation_deg) + s.min_rotation_deg
    # Sample mod, demod
    if s.mod_class === nothing
        mod = nothing
    elseif s.mod_class == ClassicMod
        mod = s.mod_class(bits_per_symbol=s.bits_per_symbol, rotation_deg=rotation_deg, avg_power=s.avg_power)
    else
        mod = s.mod_class(;s.mod_kwargs...)
    end
    if s.demod_class === nothing
        demod = nothing
    elseif s.demod_class == ClassicDemod
        demod = s.demod_class(bits_per_symbol=s.bits_per_symbol, rotation_deg=rotation_deg, avg_power=s.avg_power)
    elseif s.demod_class == ClusteringDemod
        demod = s.demod_class(bits_per_symbol=s.bits_per_symbol)
    else
        demod = s.demod_class(bits_per_symbol=s.bits_per_symbol; s.demod_kwargs...)
    end
    # Sample partner model
    if s.prtnr_model_class === nothing || s.use_prtnr_model == false
        prtnr_model = nothing
    else
        prtnr_model = s.prtnr_model_class(bits_per_symbol=s.bits_per_symbol; s.prtnr_model_kwargs...)
    end

    # Specialize returned agent type
    return Agent(mod, demod, prtnr_model, s.self_play, s.use_prtnr_model)
end


struct ClassicAgentSampler
    bits_per_symbol::Integer
    min_rotation_deg::Float32
    max_rotation_deg::Float32
    avg_power::Float32
end

get_kwargs(c::ClassicAgentSampler) = (;
    :bits_per_symbol => c.bits_per_symbol,
    :min_rotation_deg => c.min_rotation_deg,
    :max_rotation_deg => c.max_rotation_deg,
    :avg_power => c.avg_power,
)

function ClassicAgentSampler(;bits_per_symbol, min_rotation_deg=0f0, max_rotation_deg=Float32(2pi), avg_power=1f0)
     ClassicAgentSampler(bits_per_symbol, min_rotation_deg, max_rotation_deg, avg_power)
end

function Random.rand(rng::AbstractRNG, s::ClassicAgentSampler)
    rotation_deg = Float32(rand(rng) * (s.max_rotation_deg - s.min_rotation_deg) + s.min_rotation_deg)
    mod = ClassicMod(bits_per_symbol=s.bits_per_symbol, rotation_deg=rotation_deg, avg_power=s.avg_power)
    demod = ClassicDemod(bits_per_symbol=s.bits_per_symbol, rotation_deg=rotation_deg, avg_power=s.avg_power)
    ClassicAgent(s.bits_per_symbol, rotation_deg, s.avg_power, mod, demod, false, false)
end

##############################
# High-level Agent constructor

"""Return specialized Agent type based on mod & demod types"""
function Agent(mod::MaybeMod, demod::MaybeDMod, prtnr_model::MaybeNDMod=nothing, self_play=false, use_prtnr_model=false)
    # Get global bps
    # @infiltrate
    bps = mod === nothing ? demod.bits_per_symbol : mod.bits_per_symbol
    # Get classic rotation
    rotation_deg = isclassic(mod) ? mod.rotation[1] * 180 / pi : 0
    rotation_deg = isclassic(demod) ? demod.rotation[1] * 180 / pi : rotation_deg
    # Get mod/demod average power
    if mod !== nothing
        avg_power = mod.avg_power
    elseif isclassic(demod)
        avg_power = demod.avg_power
    else
        avg_power = 1f0
    end
    # Create Agent struct
    if isa(mod, MaybeNMod) && isa(demod, MaybeNDMod)
        agent = NeuralAgent(bps, mod, demod, prtnr_model, self_play, use_prtnr_model)
    elseif isa(mod, MaybeCMod) && isa(demod, MaybeCDMod)
        agent = ClassicAgent(bps, rotation_deg, avg_power, mod, demod, false, false)
    else
        agent = MixedAgent(bps, rotation_deg, avg_power, mod, demod, prtnr_model, self_play, use_prtnr_model)
    end
    agent
end

"""Agent constructor from kwargs"""
function Agent(;mod::Union{NamedTuple, Nothing}, demod::Union{NamedTuple, Nothing},
                prtnr_model::Union{NamedTuple, Nothing}=nothing, self_play=false, use_prtnr_model=false, kwargs...)
    if mod === nothing || length(mod) == 0
        mod = nothing
    end
    if demod === nothing || length(demod) == 0
        demod = nothing
    end
    if prtnr_model === nothing || length(prtnr_model) == 0 || use_prtnr_model == false
        prtnr_model = nothing
    end
    newmod = mod === nothing ? mod : Modulator(; mod...)
    newdemod = demod === nothing ? demod : Demodulator(; demod...)
    newprtnr_model = prtnr_model === nothing ? prtnr_model : Demodulator(; prtnr_model...)
    Agent(newmod, newdemod, newprtnr_model, self_play, use_prtnr_model)
end

"""Copy constructor with optional component replacements"""
function Agent(a::Agent; mod=nothing, demod=nothing, prtnr_model=nothing)
    Agent(mod === nothing ? a.mod : mod, demod === nothing ? a.demod : demod,
          prtnr_model === nothing ? a.prtnr_model : prtnr_model, a.self_play, a.use_prtnr_model)
end

"""
Trainable fields for Agent
Partner models are trained separately from mod and demod, so not included here.
"""
function Flux.trainable(a::Agent)
    if isneural(a.mod) && isneural(a.demod)
        return (;mod=a.mod, demod=a.demod)
    elseif isneural(a.mod)
        return (;mod=a.mod)
    elseif isneural(a.demod)
        return (;demod=a.demod)
    else
        return NamedTuple()
    end
end

Base.show(io::IO, a::Agent) = print(io, Crayon(bold=true, foreground=:green), typeof(a),
                                    Crayon(reset=true), "(", a.mod, ", ", a.demod, ", ",
                                    a.prtnr_model, a.self_play ? ", SP" : "",
                                    a.use_prtnr_model ? ", PM" : "", ")")

iscuda(a::Agent) = iscuda(a.mod) || iscuda(a.demod)


end