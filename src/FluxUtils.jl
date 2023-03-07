# Julia ML utils

module FluxUtils
using CUDA: CuArray
using Flux
using Zygote
using Optimisers
using ElasticArrays
using Statistics
using Distributions
import Random

using ..DataUtils
using ..ExtraOptimisers

export multi_agent_params, get_optimiser_type
export Memory, memorize!
export loss_vanilla_pg, loss_crossentropy
export symbols_to_onehot, logits_to_symbols_si, logits_to_symbols_sb, normlogpdf



"""
Deprecated

Construct Params object for all models in agents
Returns params, list of models
"""
function multi_agent_params(agents)
    models = collect(Iterators.flatten([(a.mod, a.demod) for a in agents]))
    Flux.params(models), models
end


"""
Return Flux optimiser type from name
"""
function get_optimiser_type(name)
    if lowercase(name) == "adam"
        return Optimisers.Adam
    elseif lowercase(name) == "sgd"
        return Optimisers.Descent
    elseif lowercase(name) == "adamw"
        return Optimisers.AdamW
    elseif lowercase(name) == "oadam"
        return Optimisers.OAdam
    elseif lowercase(name) == "adabelief"
        return Optimisers.AdaBelief
    elseif lowercase(name) == "yogi"
        return Yogi
    elseif lowercase(name) == "madgrad"
        return MADGrad
    else
        throw(ValueError("Unknown optimiser $(name)"))
    end
end


"""
Replay buffer struct for RL training
"""
struct Memory
    states::ElasticArray{Float32, 2}
    actions::ElasticArray{Float32, 2}
    rewards::AbstractVector{Float32}
    rewards_to_go::AbstractVector{Float32}
    done::AbstractVector{Bool}
end


function Memory(state_dim, action_dim)
    states = ElasticArray{Float32, 2}(undef, (state_dim, 0))
    actions = ElasticArray{Float32, 2}(undef, (action_dim, 0))
    rewards = Vector{Float32}(undef, 0)
    rewards_to_go = Vector{Float32}(undef, 0)
    done = Vector{Bool}(undef, 0)
    Memory(states, actions, rewards, rewards_to_go, done)
end


memorize_lock = ReentrantLock()
"""
Append rollout (named tuple) to memory, preserving max length of steps

Threadsafe
"""
function memorize!(memory::Memory, rollout; maxlen=10_000)
    global memorize_lock
    lock(memorize_lock)
    try
        # states, actions, rewards, rewards_to_go, done = rollout
        n_new = length(rollout.done)
        if length(memory) + n_new > maxlen
            # resize! keeps first n elements if shrinking
            resize!(memory.states, (size(memory.states)[1], maxlen - n_new))
            resize!(memory.actions, (size(memory.actions)[1], maxlen - n_new))
            resize!(memory.rewards, maxlen - n_new)
            resize!(memory.rewards_to_go, maxlen - n_new)
            resize!(memory.done, maxlen - n_new)
        end
        prepend!(memory.states, rollout.states)
        prepend!(memory.actions, rollout.actions)
        prepend!(memory.rewards, rollout.rewards)
        prepend!(memory.rewards_to_go, rollout.rewards)
        prepend!(memory.done, rollout.done)
    finally
        unlock(memorize_lock)
    end
    memory
end


"""
Slice a Memory buffer, returning a named tuple of contents at the index(es)
"""
@Base.propagate_inbounds function Base.getindex(memory::Memory, inds)
    return (states = reshape(memory.states[:, inds].data, (size(memory.states)[1], length(inds))),  # Assume we want full state vectors
            actions = reshape(memory.actions[:, inds].data, (size(memory.actions)[1], length(inds))),  # Assume we want full action vectors
            rewards = memory.rewards[inds],
            rewards_to_go = memory.rewards_to_go[inds],
            done = memory.done[inds]
           )
end


"""
Get last index of memory
"""
function Base.lastindex(memory::Memory)
    return length(memory)
end


"""
Return the length of a Memory buffer
"""
function Base.length(memory::Memory)
    slen = size(memory.states)[2]
    # alen = size(memory.actions)[2]
    # rlen = length(memory.rewards)
    # dlen = length(memory.done)
    # @assert slen == alen == rlen == dlen
    slen
end


"""
Sampler for memory buffer, single value
"""
function Random.rand(rng::Random.AbstractRNG, memory::Memory)
    memory[rand(rng, 1:length(memory))]
end


"""
Sampler for memory buffer, with rng
"""
function Random.rand(rng::Random.AbstractRNG, memory::Memory, n::Int)
    memory[(rand(rng, 1:length(memory), n))]
end


"""
Sampler for memory buffer, default rng
"""
function Random.rand(memory::Memory, n::Int)
    memory[sort(rand(Random.GLOBAL_RNG, 1:length(memory), n))]
end


"""
REINFORCE policy gradient loss for modulator
"""
function loss_vanilla_pg(;mod, reward, actions)
    log_probs = sum.(eachcol(log.(exp.(logpdf(mod.policy, actions)) .+ mod.lambda_prob)))
    baseline = mean(reward)
    loss = -mean(log_probs .* (reward .- baseline))
    loss
end


"""
Cross-Entropy loss for demodulator
"""
function loss_crossentropy(;logits, target)
    Flux.logitcrossentropy(logits, target, dims=1, agg=mean)
end


"""
Converts array of bit representation of symbols to one-hot array
Inputs:
data_sb: array of type integer containing 0-1 entries of shape [bits_per_symbol, m]
Output:
data_oh: matrix containing onehot representation of shape [2^bits_per_symbol, m]
"""
function symbols_to_onehot(data_sb)
    bps = size(data_sb, 1)
    data_si = symbols_to_integers(data_sb)
    data_oh = Flux.onehotbatch(data_si, 0:2^bps - 1)
    data_oh
end

# Need to specialize for CuArrays because onehotbatch moves to CPU (possibly through scalar indexing)
function symbols_to_onehot(data_sb::CuArray)
    bps = size(data_sb, 1)
    data_si = symbols_to_integers(data_sb)
    data_oh = Flux.onehotbatch(data_si |> cpu, 0:2^bps - 1) |> gpu
    data_oh
end




"""
Convert logits to 0-indexed integers for symbol values
"""
function logits_to_symbols_si(logits)
    symbs_si = getindex.(argmax(logits, dims=1), 1) .- 1
    dropdims(symbs_si, dims=1)
end


"""
Convert logits to array of symbol bits
"""
function logits_to_symbols_sb(logits, bits_per_symbol)
    integers_to_symbols(logits_to_symbols_si(logits), bits_per_symbol)
end


"""
GPU automatic differentiable version for the logpdf function of normal distributions.
Adding an epsilon value to guarantee numeric stability if sigma is exactly zero
(e.g. if relu is used in output layer).

https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl/blob/c14ec876311408096a677f1b998e307cafff4dc4/src/extensions/Distributions.jl
"""
function normlogpdf(μ, σ, x; ϵ = 1.0f-8)
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log(2.0f0π)) / 2.0f0 .- log.(σ .+ ϵ)
end

end
