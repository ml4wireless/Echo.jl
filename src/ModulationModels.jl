module ModulationModels
export ClassicMod, ClassicDemod
export NeuralMod, NeuralDemod
export modulate, demodulate, isclassic, ismod, iscuda, loss
export get_kwargs, Modulator, Demodulator


push!(LOAD_PATH, "./")

using CUDA: CuArray
using Distributions
using Flux
using Flux: unsqueeze
using ChainRules: @ignore_derivatives, ignore_derivatives
using Printf
using Random
using Statistics

using ..ModulationUtils
using ..DataUtils
using ..FluxUtils

import Random.rand
import Distributions.logpdf

# Helpers

# rotation_matrix(r) = Float32[[cos(r) sin(r)]
                            #  [-sin(r) cos(r)]]
rotation_matrix(r) = hcat([cos(r); -sin(r)], [sin(r); cos(r)])

plain_print_list(l) = "[" * join(l, ",") * "]"

##################################################################################
# Abstract Types
##################################################################################
abstract type Modulator end
abstract type Demodulator end

##################################################################################
# Classic Modulator
##################################################################################
struct ClassicMod{V <: AbstractVector{Float32}, M <: AbstractMatrix{Float32}} <: Modulator
    bits_per_symbol::Int
    # This is ALWAYS a 1-element vector.
    # Has to be mutable and held by reference for autodiff
    rotation::V
    avg_power::Float32
    symbol_map::M
end

# Only the rotation field of a ClassicMod struct is trainable
# but symbol map should also be sent to the desired device
Flux.@functor ClassicMod (rotation, symbol_map)
Flux.trainable(m::ClassicMod) = (m.rotation,)

Base.show(io::IO, m::ClassicMod) = @printf(io, "%s(bps=%d, rot=%.1f)",
    typeof(m), m.bits_per_symbol, 180/pi*m.rotation[1])

function ClassicMod(;bits_per_symbol::Integer, rotation_deg::Real=0., avg_power::Real=1.)
    symbol_map = get_symbol_map(bits_per_symbol)
    # symbol_map = rotation_matrix(rotation) * symbol_map
    curr_avg_power = mean(sum(abs2.(symbol_map), dims=1))
    normalization_factor = sqrt((relu(curr_avg_power - avg_power) + avg_power) / avg_power)
    return ClassicMod(bits_per_symbol, Float32[convert(Float32, pi / 180 * rotation_deg)], avg_power,
                      symbol_map .* normalization_factor)
end


get_kwargs(m::ClassicMod; include_weights::Bool=false) = (;
    :bits_per_symbol => m.bits_per_symbol,
    :rotation_deg => m.rotation[1] * 180 / pi,
    :avg_power => m.avg_power
)


"""
Normalize avg power of constellation to avg_power
Inputs:
mod: ClassicMod for settings
means: constellation points to normalize

Outputs:
normalized means
"""
function normalize_constellation(mod::ClassicMod, means)
    avg_power = mean(sum(abs2.(mod.symbol_map), dims=1))
    if mod.avg_power > 0.
        norm_factor = sqrt((relu(avg_power - mod.avg_power) + mod.avg_power) / mod.avg_power)
    else
        norm_factor = one(means[1])  # Automatically match type of means elements
    end
    means = means ./ norm_factor
    means
end


"""
Modulate a bps x N symbol message to cartesian coordinates
"""
function (m::ClassicMod)(message)
    inds = symbols_to_integers(message)
    rmap = rotation_matrix(m.rotation[1]) * m.symbol_map
    norm_map = normalize_constellation(m, rmap)
    return norm_map[:, inds .+ 1]
end


"""
Modulate a bps x N symbol message to cartesian coordinates
"""
function modulate(m::ClassicMod, symbols; explore=False)
    m(symbols)
end


##################################################################################
# Classic Demodulator
##################################################################################
struct ClassicDemod{V <: AbstractVector{Float32}, M <: AbstractMatrix{Float32}} <: Demodulator
    bits_per_symbol::Int
    # This is ALWAYS a 1-element vector.
    # Has to be mutable and held by reference for autodiff
    rotation::V
    avg_power::Float32
    symbol_map::M
end

# Only the rotation field of a ClassicDemod struct is trainable
# but symbol map should also be sent to the desired device
Flux.@functor ClassicDemod (rotation, symbol_map)
Flux.trainable(d::ClassicDemod) = (d.rotation,)

Base.show(io::IO, d::ClassicDemod) = @printf(io, "%s(bps=%d, rot=%.1f)",
    typeof(d), d.bits_per_symbol, 180/pi*d.rotation[1])

function ClassicDemod(;bits_per_symbol::Integer, rotation_deg::Real=0f0, avg_power::Real=1.)
    symbol_map = get_symbol_map(bits_per_symbol)
    # symbol_map = rotation_matrix(rotation) * symbol_map
    curr_avg_power = mean(sum(abs2.(symbol_map), dims=1))
    normalization_factor = sqrt((relu(curr_avg_power - avg_power) + avg_power) / avg_power)
    return ClassicDemod(bits_per_symbol, [convert(Float32, pi / 180 * rotation_deg)], avg_power,
                        symbol_map .* normalization_factor)
end


get_kwargs(d::ClassicDemod; include_weights::Bool=false) = (;
    :bits_per_symbol => d.bits_per_symbol,
    :rotation_deg => d.rotation[1] * 180 / pi,
    :avg_power => d.avg_power
)


"""
Inputs:
iq: shape [2,n] with modulated symbols

Output:
logits: negative squared error w.r.t. constellation points
"""
function (d::ClassicDemod)(iq)
    rmatrix = rotation_matrix(d.rotation[1])
    if iscuda(d)
        rmatrix = gpu(rmatrix)
    end
    rmap = rmatrix * d.symbol_map
    dist = sum(abs2.(unsqueeze(iq, 2) .- rmap), dims=1)
    dist = dropdims(dist, dims=1)
    logits = -dist
    # labels_si_g = getindex.(argmax(dist, dims=1), 1)
    logits
end


"""
Inputs:
iq: shape [2,n] with modulated symbols

Output:
symbols_si: vector of integer symbols, or logits if soft=true
"""
function demodulate(d::ClassicDemod, iq; soft=false)
    logits = d(iq)
    if soft
        return logits
    end
    logits_to_symbols_si(logits)
end


##################################################################################
# Neural Modulator
##################################################################################
mutable struct NeuralModPolicy{A <: AbstractArray{Normal{Float32}}}
    symb_policies::A
end

Flux.@functor NeuralModPolicy (symb_policies,)
NeuralModPolicy() = NeuralModPolicy(Matrix{Normal{Float32}}(undef, 0, 0))
Random.rand(rng::AbstractRNG, policy::NeuralModPolicy) = rand.(rng, policy.symb_policies)
Distributions.logpdf(policy::NeuralModPolicy, x) = logpdf.(policy.symb_policies, x)
Base.size(p::NeuralModPolicy) = size(p.symb_policies)


struct NeuralMod{A <: AbstractVector{Float32}, U <: AbstractMatrix{UInt16}} <: Modulator
    bits_per_symbol::Int
    hidden_layers::Vector{<:Integer}
    restrict_energy::Int
    activation_fn_hidden::Function
    activation_fn_output::Function
    avg_power::Float32
    μ::Flux.Chain
    log_std::A
    policy::NeuralModPolicy
    log_std_dict::NamedTuple  # min, max, initial
    lr_dict::NamedTuple  # mu, std
    all_unique_symbols::U
    lambda_prob::Float32  # For numerical stability while computing pg loss
end

# Only the μ and log_std fields of a NeuralMod struct are trainable
Flux.@functor NeuralMod (μ, log_std, policy, all_unique_symbols)
Flux.trainable(m::NeuralMod) = (m.μ, m.log_std)

Base.show(io::IO, m::NeuralMod) = @printf(io, "%s(bps=%d, hl=%s, actv_fn=%s, re=%d)",
    typeof(m), m.bits_per_symbol, plain_print_list(m.hidden_layers),
    String(Symbol(m.activation_fn_hidden)), m.restrict_energy)

function NeuralMod(;bits_per_symbol,
                   hidden_layers,
                   restrict_energy=1, activation_fn_hidden=tanh,
                   activation_fn_output=identity, avg_power=1f0,
                   log_std_dict=(; initial=-1f0, max=1f0, min=-3f0),
                   lr_dict=(; mu=5f-1, std=1f-3),
                   lambda_prob=1f-6,
                   weights=nothing)
    if isa(activation_fn_hidden, String)
        activation_fn_hidden = getfield(Flux, Symbol(activation_fn_hidden))
    end
    hidden_layers = Vector{Int}(hidden_layers)
    layer_dims = vcat([bits_per_symbol], hidden_layers, [2])
    layers = []
    for i in 1:length(layer_dims)-2
        push!(layers, Dense(layer_dims[i], layer_dims[i+1], activation_fn_hidden))
    end
    push!(layers, Dense(layer_dims[end-1], layer_dims[end], activation_fn_output))
    μ = Chain(layers...)
    if weights !== nothing
        Flux.loadparams!(μ, weights)
    end
    log_std = Vector{Float32}([log_std_dict.initial, log_std_dict.initial])
    all_unique_symbols = integers_to_symbols(0:(2^bits_per_symbol-1), bits_per_symbol)
    NeuralMod(bits_per_symbol, hidden_layers,
              restrict_energy, activation_fn_hidden,
              activation_fn_output, avg_power,
              μ, log_std, NeuralModPolicy(),
              log_std_dict, lr_dict,
              all_unique_symbols, lambda_prob,
              )
end


get_kwargs(m::NeuralMod; include_weights=false) = (;
    :bits_per_symbol => m.bits_per_symbol,
    :hidden_layers => m.hidden_layers,
    :restrict_energy => m.restrict_energy,
    :activation_fn_hidden => String(Symbol(m.activation_fn_hidden)),
    :avg_power => m.avg_power,
    :log_std_dict => m.log_std_dict,
    :lr_dict => cpu(m.lr_dict),
    :lambda_prob => m.lambda_prob,
    # TODO: change to new recommended method: deepcopy(m.μ), loadmodel!(m.μ, prevμ)
    :weights => include_weights ? deepcopy(Flux.params(cpu(m.μ))) : nothing,
)


"""
Normalize avg power of constellation to avg_power
Inputs:
mod: NeuralMod for settings
means: constellation points to normalize

Outputs:
normalized means
"""
function normalize_constellation(mod::NeuralMod, means)
    avg_power = mean(sum(abs2.(mod.μ(mod.all_unique_symbols)), dims=1))
    if mod.avg_power > 0.
        norm_factor = sqrt((relu(avg_power - mod.avg_power) + mod.avg_power) / mod.avg_power)
    else
        norm_factor = one(means[1])  # Automatically match type of means elements
    end
    means = means ./ norm_factor
    means
end


"""
Normalize avg power of transmitted symbols to avg_power
Inputs:
mod: NeuralMod for settings
means: constellation points to normalize

Outputs:
normalized means
"""
function normalize_symbols(mod::NeuralMod, means)
    avg_power = mean(sum(abs2.(means), dims=1))
    if mod.avg_power > 0.
        norm_factor = sqrt((relu(avg_power - mod.avg_power) + mod.avg_power) / mod.avg_power)
    else
        norm_factor = one(means[1])  # Automatically match type of means elements
    end
    means = means ./ norm_factor
    means
end


function center_means(means)
    center = mean(means, dims=2)
    return means .- center
end


"""
Center means, then normalize avg power of constellation to avg_power
Inputs:
mod: NeuralMod for settings
means: constellation points to normalize

Outputs:
centered then normalized means
"""
function center_and_normalize_constellation(mod::NeuralMod, means)
    centered_constellation = center_means(mod.μ(mod.all_unique_symbols))
    avg_power = mean(sum(abs2.(centered_constellation), dims=1))
    if mod.avg_power > 0.
        norm_factor = sqrt((relu(avg_power - mod.avg_power) + mod.avg_power) / mod.avg_power)
    else
        norm_factor = one(means[1])  # Automatically match type of means elements
    end
    means = center_means(means) ./ norm_factor
    means
end


"""
Modulate a bps x N symbol message to cartesian coordinates
"""
function (m::NeuralMod)(symbols::AbstractArray{Float32})
    means = m.μ(symbols)
    if m.restrict_energy == 1
        means = normalize_constellation(m, means)
    elseif m.restrict_energy == 2
        means = normalize_symbols(m, means)
    elseif m.restrict_energy == 3
        means = center_and_normalize_constellation(m, means)
    end
    means
end


"""
Modulate a bps x N symbol message to cartesian coordinates, with policy exploration
"""
modulate(m::NeuralMod, symbols::AbstractArray{UInt16}; explore::Bool=false) = modulate(m, Float32.(symbols), explore=explore)
function modulate(m::NeuralMod, symbols::AbstractArray{Float32}; explore::Bool=false)
    means = m(2 .* symbols .- 1)
    if explore
        log_std = clamp.(m.log_std, m.log_std_dict.min, m.log_std_dict.max)
        m.policy.symb_policies = Normal.(means, exp.(reshape(log_std, (:, 1))))
        cartesian_points = ignore_derivatives(rand(m.policy))
    else
        cartesian_points = means
    end
    cartesian_points
end


"""
Modulator policy gradient loss

symbols: 2 x n: original symbols
received_symbols: 2 x n:f guesses of symbols after demodulation
actions: 2 x n: cartesian I-Q constellation space after modulation

reward is negative of bit errors between symbols and received symbols
"""
function loss(mod::NeuralMod; symbols, received_symbols, actions)
    reward = ignore_derivatives() do
        -sum.(eachcol(Float32.(xor.(symbols, received_symbols))))
    end
    loss = loss_vanilla_pg(mod=mod, reward=reward, actions=actions)
end


##################################################################################
# Neural Demdulator
##################################################################################
struct NeuralDemod <: Demodulator
    bits_per_symbol::Int
    hidden_layers::Vector{<:Integer}
    activation_fn_hidden::Function
    activation_fn_output::Function
    net::Flux.Chain
    lr::Float32
    all_unique_symbols::Array{Float32, 2}
end

# Only the net field of a NeuralDemod struct is trainable
Flux.@functor NeuralDemod (net,)

Base.show(io::IO, d::NeuralDemod) = @printf(io, "%s(bps=%d, hl=%s, actv_fn=%s)",
    typeof(d), d.bits_per_symbol, plain_print_list(d.hidden_layers),
    String(Symbol(d.activation_fn_hidden)))

function NeuralDemod(;bits_per_symbol::Integer, hidden_layers::Vector{Int},
                     activation_fn_hidden::Union{Function,String}=tanh,
                     activation_fn_output::Function=identity,
                     lr::Real=1f-1, weights=nothing)
    if isa(activation_fn_hidden, String)
        activation_fn_hidden = getfield(Flux, Symbol(activation_fn_hidden))
    end
    hidden_layers = Vector{Int}(hidden_layers)
    layer_dims = vcat([2], hidden_layers, [2^bits_per_symbol])
    layers = []
    for i in 1:length(layer_dims)-2
        push!(layers, Dense(layer_dims[i], layer_dims[i+1], activation_fn_hidden))
    end
    push!(layers, Dense(layer_dims[end-1], layer_dims[end], activation_fn_output))
    net = Chain(layers...)
    if weights !== nothing
        Flux.loadparams!(net, weights)
    end
    all_unique_symbols = integers_to_symbols(0:(2^bits_per_symbol-1), bits_per_symbol)
    NeuralDemod(bits_per_symbol, hidden_layers, activation_fn_hidden,
                activation_fn_output, net, Float32(lr), all_unique_symbols)
end


get_kwargs(d::NeuralDemod; include_weights=false) = (;
    :bits_per_symbol => d.bits_per_symbol,
    :hidden_layers => d.hidden_layers,
    :activation_fn_hidden => String(Symbol(d.activation_fn_hidden)),
    :lr => d.lr,
    # TODO: change to new recommended method: deepcopy(d.net), loadmodel!(d.net, prevnet)
    :weights => include_weights ? deepcopy(Flux.params(cpu(d.net))) : nothing,
)


"""
Inputs:
iq: shape [2,n] with modulated symbols

Output:
logits: shape [2^bits_per_symbol, n]
"""
function (d::NeuralDemod)(iq)
    logits = d.net(iq)
    logits
end


"""
Inputs:
iq: shape [2,n] with modulated symbols

Output:
symbols_si: vector of integer symbols, or logits if soft=true
"""
function demodulate(d::NeuralDemod, iq; soft=false)
    logits = d(iq)
    if soft
        return logits
    end
    logits_to_symbols_si(logits)
end

"""
Update the demodulator using cross-entropy loss

logits: 2^bps x n: demodulated logits
targets: 2^bps x n: one-hot array of correct bit labels
take_step: bool: whether to apply SGD update to model

reward is negative of bit errors between symbols and received symbols
"""
function loss(demod::NeuralDemod; logits, target)
    loss = loss_crossentropy(;logits, target)
end


##################################################################################
# Generic Mod/Demod constructors
##################################################################################
function Modulator(;kwargs...)
    if :hidden_layers in keys(kwargs)
        mod = NeuralMod(;kwargs...)
    else
        mod = ClassicMod(;kwargs...)
    end
    mod
end


function Demodulator(;kwargs...)
    if :hidden_layers in keys(kwargs)
        demod = NeuralDemod(;kwargs...)
    else
        demod = ClassicDemod(;kwargs...)
    end
    demod
end


##################################################################################
# Extras
##################################################################################
isclassic(::ClassicMod) = true
isclassic(::ClassicDemod) = true
isclassic(::Any) = false

ismod(::ClassicMod) = true
ismod(::NeuralMod) = true
ismod(::Any) = false

iscuda(m::ClassicMod) = isa(m.rotation, CuArray)
iscuda(m::NeuralMod) = isa(m.log_std, CuArray)
iscuda(d::ClassicDemod) = isa(d.rotation, CuArray)
iscuda(d::NeuralDemod) = isa(d.net[1].weight, CuArray)

end
