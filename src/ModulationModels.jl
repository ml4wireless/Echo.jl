module ModulationModels
export ClassicMod, ClassicDemod
export NeuralMod, NeuralDemod
export GNNMod
export ClusteringDemod
export modulate, demodulate, constellation, isclassic, isclustering, isneural, ismod, iscuda, loss
export get_kwargs, Modulator, Demodulator


push!(LOAD_PATH, "./")

using ChainRules: @ignore_derivatives, ignore_derivatives
using Clustering: kmeans, assignments
using Combinatorics: permutations
using CUDA: CuArray, Mem.DeviceBuffer as DeviceBuffer, randn as curandn
using Crayons
using Distributions
using Flux
using Flux: unsqueeze
using IterTools: product
using Logging
using Printf
using Random
using Statistics

# using InteractiveUtils: @code_warntype

using ..ModulationUtils
using ..DataUtils
using ..FluxUtils

using Infiltrator

import Random.rand
import Distributions.logpdf


CV32 = CuArray{Float32, 1, DeviceBuffer}
CM32 = CuArray{Float32, 2, DeviceBuffer}
V32 = Vector{Float32}
M32 = Matrix{Float32}

# Helpers
slogger = Logging.SimpleLogger(Logging.Error)

# rotation_matrix(r) = Float32[[cos(r) sin(r)]
                            #  [-sin(r) cos(r)]]
rotation_matrix(r) = hcat([cos(r); -sin(r)], [sin(r); cos(r)])

plain_print_list(l) = "[" * join(l, ",") * "]"

function color_type(t, main_color::Symbol = :blue)
    typestr = string(typeof(t))
    tmain, tparam = match(r"(\w+)(\{.+\})*", typestr).captures
    if tparam === nothing
        tparam = ""
    end
    (Crayon(foreground=main_color, bold=true), tmain,
     Crayon(foreground=:light_gray, bold=false, italics=true), tparam,
     Crayon(reset=true))
end

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
    trainable::Bool
end

# Only the rotation field of a ClassicMod struct is trainable
# but only symbol map should be sent to the desired device
Flux.@functor ClassicMod (symbol_map,)
Flux.trainable(m::ClassicMod) = m.trainable ? (; rotation=m.rotation,) : (;)

function Base.show(io::IO, m::ClassicMod)
    argstr = @sprintf("(bps=%d, rot=%.1f)", m.bits_per_symbol, 180/pi*m.rotation[1])
    print(io, color_type(m)..., argstr)
end

function ClassicMod(;bits_per_symbol::Integer, rotation_deg::Real=0., avg_power::Real=1., trainable::Bool=false)
    symbol_map = get_symbol_map(bits_per_symbol)
    # symbol_map = rotation_matrix(rotation) * symbol_map
    curr_avg_power = mean(sum(abs2.(symbol_map), dims=1))
    normalization_factor = Float32(sqrt((relu(curr_avg_power - avg_power) + avg_power) / avg_power))
    return ClassicMod(bits_per_symbol, Float32[Float32( pi / 180 * rotation_deg)], Float32(avg_power),
                      symbol_map .* normalization_factor, trainable)
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
    rmatrix = rotation_matrix(m.rotation[1])
    if iscuda(m)
        rmatrix = gpu(rmatrix)
    end
    rmap = rmatrix * m.symbol_map
    norm_map = normalize_constellation(m, rmap)
    return norm_map[:, inds .+ 1]
end


"""
Return the constellation points of the modulator
"""
constellation(mod::ClassicMod) = mod(mod.all_unique_symbols)


"""
Modulate a bps x N symbol message to cartesian coordinates
"""
function modulate(m::ClassicMod, symbols; explore=false)
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
    trainable::Bool
end

# Only the rotation field of a ClassicDemod struct is trainable
# but only the symbol map should be sent to the desired device
Flux.@functor ClassicDemod (symbol_map,)
Flux.trainable(d::ClassicDemod) = d.trainable ? (; rotation=d.rotation,) : (;)

function Base.show(io::IO, d::ClassicDemod)
    argstr = @sprintf("(bps=%d, rot=%.1f)", d.bits_per_symbol, 180/pi*d.rotation[1])
    print(io, color_type(d)..., argstr)
end

function ClassicDemod(;bits_per_symbol::Integer, rotation_deg::Real=0f0, avg_power::Real=1., trainable::Bool=false)
    symbol_map = get_symbol_map(bits_per_symbol)
    # symbol_map = rotation_matrix(rotation) * symbol_map
    curr_avg_power = mean(sum(abs2.(symbol_map), dims=1))
    normalization_factor = Float32(sqrt((relu(curr_avg_power - avg_power) + avg_power) / avg_power))
    return ClassicDemod(bits_per_symbol, [Float32(pi / 180 * rotation_deg)], Float32(avg_power),
                        symbol_map .* normalization_factor, trainable)
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
    iq = unsqueeze(iq, 2)
    # Logits are negative squared error w.r.t. constellation points
    logits = @. -abs2(iq[1, :, :] - rmap[1, :]) - abs2(iq[2, :, :] - rmap[2, :])
    # dist = sum(abs2.(unsqueeze(iq, 2) .- rmap), dims=1)
    # dist = dropdims(dist, dims=1)
    # logits = -dist
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
# Clustering Demodulator
##################################################################################
struct ClusteringDemod <: Demodulator
    bits_per_symbol::Int
    centers::Matrix{Float32}
end

Base.show(io::IO, d::ClusteringDemod) = print(io, color_type(d)..., "(bps=$(d.bits_per_symbol))")


function ClusteringDemod(;bits_per_symbol::Integer, centers::Union{Nothing, Matrix{Float32}}=nothing)
    if centers === nothing
        centers = rand(Float32, (2, 2 ^ bits_per_symbol))
    end
    return ClusteringDemod(bits_per_symbol, centers)
end


get_kwargs(d::ClusteringDemod; include_weights::Bool=false) = (;
    :bits_per_symbol => d.bits_per_symbol,
    :centers => deepcopy(d.centers),
)


"""
Inputs:
iq: shape [2,n] with modulated symbols

Output:
logits: cluster centers, cluster assignments
"""
function (d::ClusteringDemod)(iq)
    local R
    # Disable derivatives and warnings about clustering cost increases
    @ignore_derivatives begin
        Logging.with_logger(slogger) do
            R = kmeans(iq, 2 ^ d.bits_per_symbol; display=:none)
        end
    end
    centers = R.centers
    symbs = assignments(R)
    centers, symbs
end


_get_logits(iq, centers, bps) = -reshape(sum((reshape(iq, (1, 2, :)) .- reshape(centers', (:, 2, 1))) .^ 2, dims=2), (2 ^ bps, :))

"""
Inputs:
iq: shape [2,n] with modulated symbols
soft: return logits, otherwise return hard decisions
preamble_si: vector of integer symbols, required to update cluster centers

Output:
symbols_si: vector of integer symbols, or logits if soft=true
"""
function demodulate(d::ClusteringDemod, iq; soft=false, preamble_si=nothing)
    # If we have a shared preamble, try to find the best center -> symbol mapping
    # Otherwise, reuse old centers
    if preamble_si !== nothing
        @ignore_derivatives begin
            cpreamble_si = cpu(preamble_si)
            ciq = cpu(iq)
            centers, assignments = d(ciq)
            # Majority vote on symbol decision per cluster
            for a in 1:2 ^ d.bits_per_symbol
                true_symbols = cpreamble_si[assignments .== a]
                symb_choices = modes(true_symbols)
                d.centers[:, rand(symb_choices) + 1] .= centers[:, a]
            end
        end
    end
    # If soft, return distance to each center per sample
    centers = isa(iq, CuArray) ? gpu(d.centers) : d.centers
    logits = _get_logits(iq, centers, d.bits_per_symbol)
    if soft
        return logits
    end
    # Else return assignment per sample
    logits_to_symbols_si(logits)
end


##################################################################################
# Neural Modulator
##################################################################################
mutable struct NeuralModPolicy{M <: AbstractMatrix{Float32}, V <: AbstractVector{Float32}}
    means::M
    stds::V
end

Flux.@functor NeuralModPolicy (means, stds)
NeuralModPolicy() = NeuralModPolicy(M32(undef, 0, 0), V32(undef, 0))
Random.rand(rng::AbstractRNG, policy::NeuralModPolicy{M32, V32}) = randn(rng, Float32, size(policy)) .* policy.stds .+ policy.means
Random.rand(policy::NeuralModPolicy{CM32, CV32}) = curandn(Float32, size(policy)) .* policy.stds .+ policy.means

const log2π::Float32 = Float32(log(2π))
function normallogpdf(x, μ, σ)
    @. -(((x - μ) / σ)^2 + log2π) / 2 - log(σ)
end
logpdf(policy::NeuralModPolicy, x) = normallogpdf(x, policy.means, policy.stds)
Base.size(p::NeuralModPolicy) = size(p.means)


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
    λdiversity::Float32  # Weighting for diversity loss
end

# Only the μ and log_std fields of a NeuralMod struct are trainable
Flux.@functor NeuralMod (μ, log_std, policy, all_unique_symbols)
Flux.trainable(m::NeuralMod) = (; μ=m.μ, log_std=m.log_std)

function Base.show(io::IO, m::NeuralMod)
    argstr = @sprintf("(bps=%d, hl=%s, 𝑓=%s, re=%d, λμ=%g, λσ=%g, λdiv=%g)",
                      m.bits_per_symbol, plain_print_list(m.hidden_layers),
                      string(Symbol(m.activation_fn_hidden)), m.restrict_energy,
                      m.lr_dict.mu, m.lr_dict.std, m.λdiversity)
    print(io, color_type(m)..., argstr)
end

function NeuralMod(;bits_per_symbol,
                   hidden_layers,
                   restrict_energy=1, activation_fn_hidden=elu,
                   activation_fn_output=identity, avg_power=1f0,
                   log_std_dict=(; initial=-1f0, max=1f0, min=-3f0),
                   lr_dict=(; mu=5f-1, std=1f-3),
                   lambda_prob=1f-6,
                   lambda_diversity=0f0,
                   log_std=nothing, weights=nothing)
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
        if isa(weights, Flux.Params)
            Flux.loadparams!(μ, weights)
        else
            Flux.loadmodel!(μ, weights)
        end
    end
    if log_std === nothing
        log_std = Vector{Float32}([log_std_dict.initial, log_std_dict.initial])
    end
    all_unique_symbols = get_all_unique_symbols(bits_per_symbol)
    NeuralMod(bits_per_symbol, hidden_layers,
              restrict_energy, activation_fn_hidden,
              activation_fn_output, avg_power,
              μ, log_std, NeuralModPolicy(),
              log_std_dict, lr_dict,
              all_unique_symbols, lambda_prob,
              Float32(lambda_diversity),
              )
end


get_kwargs(m::NeuralMod; include_weights=false) = (;
    :bits_per_symbol => m.bits_per_symbol,
    :hidden_layers => m.hidden_layers,
    :restrict_energy => m.restrict_energy,
    :activation_fn_hidden => String(Symbol(m.activation_fn_hidden)),
    :avg_power => m.avg_power,
    :log_std_dict => m.log_std_dict,
    :lr_dict => m.lr_dict,
    :lambda_prob => m.lambda_prob,
    :lambda_diversity => m.λdiversity,
    :log_std => include_weights ? deepcopy(cpu(m.log_std)) : nothing,
    # TODO: change to new recommended method: deepcopy(m.μ), loadmodel!(m.μ, prevμ)
    :weights => include_weights ? deepcopy(cpu(m.μ)) : nothing,
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
    avg_power = mean(sum(abs2.(mod.μ(2 .* Float32.(mod.all_unique_symbols) .- 1)), dims=1))
    if mod.avg_power > 0f0
        norm_factor = sqrt((relu(avg_power - mod.avg_power) + mod.avg_power) / mod.avg_power)
    else
        norm_factor = one(eltype(means))  # Automatically match type of means elements
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
    centered_constellation = center_means(mod.μ(2 .* Float32.(mod.all_unique_symbols) .- 1))
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
    else
        error("Unknown restrict_energy $(m.restrict_energy)")
    end
    means
end


"""
Return the constellation points of the modulator
"""
constellation(mod::NeuralMod) = mod(2 .* Float32.(mod.all_unique_symbols) .- 1)


"""
Modulate a bps x N symbol message to cartesian coordinates, with policy exploration
"""
modulate(m::NeuralMod, symbols::AbstractArray{UInt16}; explore::Bool=false) = modulate(m, 2 .* Float32.(symbols) .- 1, explore=explore)
function modulate(m::NeuralMod, symbols::AbstractArray{Float32}; explore::Bool=false)
    means = m(symbols)
    if explore
        log_std = clamp.(m.log_std, m.log_std_dict.min, m.log_std_dict.max)
        m.policy.means = means
        m.policy.stds = exp.(log_std)
        cartesian_points = ignore_derivatives() do
            rand(m.policy)
        end
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
    if mod.λdiversity > 0
        loss += mod.λdiversity * loss_diversity(modulate(mod, mod.all_unique_symbols, explore=false))
    end
    loss
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

function Base.show(io::IO, d::NeuralDemod)
    argstr = @sprintf("(bps=%d, hl=%s, 𝑓=%s, λ=%g)",
                      d.bits_per_symbol, plain_print_list(d.hidden_layers),
                      String(Symbol(d.activation_fn_hidden)), d.lr)
    print(io, color_type(d)..., argstr)
end

function NeuralDemod(;bits_per_symbol::Integer, hidden_layers::Vector{Int},
                     activation_fn_hidden::Union{Function,String}=elu,
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
# Graph Neural Network Modulator
##################################################################################
include("gnn_mod.jl")


##################################################################################
# Generic Mod/Demod constructors
##################################################################################
function Modulator(;kwargs...)
    if :layer_dims in keys(kwargs)
        mod = GNNMod(;kwargs...)
    elseif :hidden_layers in keys(kwargs)
        mod = NeuralMod(;kwargs...)
    else
        mod = ClassicMod(;kwargs...)
    end
    mod
end


function Demodulator(;kwargs...)
    if :hidden_layers in keys(kwargs)
        demod = NeuralDemod(;kwargs...)
    elseif :rotation_deg in keys(kwargs)
        demod = ClassicDemod(;kwargs...)
    else
        demod = ClusteringDemod(;kwargs...)
    end
    demod
end


##################################################################################
# Extras
##################################################################################
"""
    `isclassic([de]mod)`

Returns true if `mod`/`demod` is a Classic [de]modulator
"""
isclassic(::ClassicMod) = true
isclassic(::ClassicDemod) = true
isclassic(::Any) = false

"""
    `isclusterint(demod)`

Returns true if `demod` is a clustering demodulator
"""
isclustering(::ClusteringDemod) = true
isclustering(::Any) = false

"""
    `isneural([de]mode)`

Returns true if `mod`/`demod` is a neural (i.e. trainable via GD) [de]modulator, including both Neural and GNN modulators
"""
isneural(::NeuralMod) = true
isneural(::NeuralDemod) = true
isneural(::GNNMod) = true
isneural(::Any) = false

"""
    `ismod(m)`

Returns true if `m` is a Modulator
"""
ismod(::ClassicMod) = true
ismod(::NeuralMod) = true
ismod(::GNNMod) = true
ismod(::Any) = false

"""
    `iscuda(md)`

Returns true if `md` is a mod/demod with CUDA data
"""
iscuda(m::ClassicMod) = isa(m.symbol_map, CuArray)
iscuda(m::NeuralMod) = isa(m.log_std, CuArray)
iscuda(m::GNNMod) = isa(m.log_std, CuArray)
iscuda(d::ClassicDemod) = isa(d.symbol_map, CuArray)
iscuda(d::NeuralDemod) = isa(d.net[1].weight, CuArray)
iscuda(::ClusteringDemod) = false
iscuda(::Nothing) = false

end
