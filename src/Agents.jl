module Agents
export Agent, ClassicAgent, NeuralAgent, MixedAgent
export ClassicAgentSampler, AgentSampler
export get_kwargs, iscuda

using ..ModulationModels
import .ModulationModels.get_kwargs
import .ModulationModels.iscuda

using Random
import Random.rand
using Flux: @functor
import Flux.trainable

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
end

@functor NeuralAgent (mod, demod)

get_kwargs(a::NeuralAgent) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :mod => get_kwargs(a.mod, include_weights=true),
    :demod => get_kwargs(a.demod, include_weights=true),
)


##############
# Mixed agents

struct MixedAgent <: Agent
    bits_per_symbol::Integer
    rotation_deg::Float32
    avg_power::Float32
    mod::MaybeMod
    demod::MaybeDMod
end

@functor MixedAgent (mod, demod)

get_kwargs(a::MixedAgent) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :rotation_deg => a.rotation_deg,
    :avg_power => a.avg_power,
    :mod => get_kwargs(a.mod, include_weights=true),
    :demod => get_kwargs(a.demod, include_weights=true),
)


################
# Classic agents

struct ClassicAgent <: Agent
    bits_per_symbol::Integer
    rotation_deg::Float32
    avg_power::Float32
    mod::MaybeCMod
    demod::MaybeCDMod
end

@functor ClassicAgent (mod, demod)

get_kwargs(a::ClassicAgent) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :rotation_deg => a.rotation_deg,
    :avg_power => a.avg_power,
    :mod => get_kwargs(a.mod),
    :demod => get_kwargs(a.demod),
)


################
# Agent Sampling

struct AgentSampler{M <: Modulator, D <: Demodulator}
    bits_per_symbol::Integer
    mod_class::Union{Type{M}, Nothing}
    demod_class::Union{Type{D}, Nothing}
    mod_kwargs::NamedTuple
    demod_kwargs::NamedTuple
    min_rotation_deg::Float32
    max_rotation_deg::Float32
    avg_power::Float32
end


get_kwargs(a::AgentSampler) = (;
    :bits_per_symbol => a.bits_per_symbol,
    :mod_class => a.mod_class,
    :demod_class => a.demod_class,
    :mod_kwargs => a.mod_kwargs,
    :demod_kwargs => a.demod_kwargs,
    :min_rotation_deg => a.min_rotation_deg,
    :max_rotation_deg => a.max_rotation_deg,
    :avg_power => a.avg_power,
)

function AgentSampler(;bits_per_symbol, mod_class, demod_class, mod_kwargs=NamedTuple(), demod_kwargs=NamedTuple(), min_rotation_deg=0f0, max_rotation_deg=Float32(2pi), avg_power=1f0)
    MC = mod_class === nothing ? Modulator : mod_class
    DC = demod_class === nothing ? Demodulator : demod_class
    AgentSampler{MC, DC}(bits_per_symbol, mod_class, demod_class, mod_kwargs, demod_kwargs, min_rotation_deg, max_rotation_deg, avg_power)
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

    # Specialize returned agent type
    return Agent(mod, demod)
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
    ClassicAgent(s.bits_per_symbol, rotation_deg, s.avg_power, mod, demod)
end

##############################
# High-level Agent constructor

"""Return specialized Agent type based on mod & demod types"""
function Agent(mod::MaybeMod, demod::MaybeDMod)
    # Get global bps
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
        agent = NeuralAgent(bps, mod, demod)
    elseif isa(mod, MaybeCMod) && isa(demod, MaybeCDMod)
        agent = ClassicAgent(bps, rotation_deg, avg_power, mod, demod)
    else
        agent = MixedAgent(bps, rotation_deg, avg_power, mod, demod)
    end
    agent
end

"""Agent constructor from kwargs"""
function Agent(;mod::Union{NamedTuple, Nothing}, demod::Union{NamedTuple, Nothing}, kwargs...)
    if length(mod) == 0
        mod = nothing
    end
    if length(demod) == 0
        demod = nothing
    end
    newmod = mod === nothing ? mod : Modulator(; mod...)
    newdemod = demod === nothing ? demod : Demodulator(; demod...)
    Agent(newmod, newdemod)
end

"""Trainable fields for Agent"""
function Flux.trainable(a::Agent)
    if a.mod === nothing && a.demod === nothing
        return ()
    elseif a.mod === nothing
        return (; demod=a.demod)
    elseif a.demod === nothing
        return (; mod=a.mod)
    else
        return (; mod=a.mod, demod=a.demod)
    end
end


iscuda(a::Agent) = iscuda(a.mod)


end