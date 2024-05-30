# Julia ML utils

module FluxUtils
using ChainRules: ignore_derivatives
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
using ..Schedules

export get_optimiser_type, get_optimisers, get_schedule_type, get_schedules, get_true_lr
export Memory, memorize!
export loss_vanilla_pg, loss_crossentropy, loss_diversity, loss_ks_uniform
export symbols_to_onehot, logits_to_symbols_si, logits_to_symbols_sb, normlogpdf


"""
Return optimiser Type by name
"""
function get_optimiser_type(name)
    name = lowercase(name)
    opt_map = Dict(
        "adam" => Optimisers.Adam,
        "sgd" => Optimisers.Descent,
        "adamw" => Optimisers.AdamW,
        "oadam" => Optimisers.OAdam,
        "adabelief" => Optimisers.AdaBelief,
        "yogi" => Yogi,
        "madgrad" => MADGrad,
        "ldog" => LDoG,
        "lion" => Optimisers.Lion,
        "prodigy" => Prodigy,
    )
    opt_type = get(opt_map, name, nothing)
    if opt_type === nothing
        error("Unknown optimiser $(name)")
    end
    opt_type
end


"""
    optims = get_optimisers(agents, optimiser = Optimisers.Adam)

Initialize optimisers for neural models.

# Parameters
- Agents (`agents`): Iterable of Agents to optimize.
- Optimiser (`optimiser`): Optimiser type to use.

# Returns
- `optims`: Vector of optimiser states for every trainable model
"""
function get_optimisers(agents, optimiser=Optimisers.AdamW)
    optims = []
    for a in agents
        # Don't worry about setting lr here, it will be adjusted individually below
        state = Optimisers.setup(optimiser(), a)
        Optimisers.freeze!(state)
        if hasfield(typeof(a.mod), :μ)
            Optimisers.adjust!(state.mod.μ, a.mod.lr_dict.mu)
            Optimisers.thaw!(state.mod.μ)
            Optimisers.adjust!(state.mod.log_std, a.mod.lr_dict.std)
            Optimisers.thaw!(state.mod.log_std)
        end
        if hasfield(typeof(a.demod), :net)
            Optimisers.adjust!(state.demod.net, a.demod.lr)
            Optimisers.thaw!(state.demod.net)
        end
        # Partner model will always be a NeuralDemod if it exists
        if hasfield(typeof(a), :prtnr_model) && a.prtnr_model !== nothing
            Optimisers.adjust!(state.prtnr_model.net, a.prtnr_model.lr)
            Optimisers.thaw!(state.prtnr_model.net)
        end
        push!(optims, state)
    end
    optims
end


"""
Return schedule type by name.
"""
function get_schedule_type(name)
    name = lowercase(name)
    sched_map = Dict(
        "constant" => ConstantSchedule,
        "linear" => LinearSchedule,
        "cosine" => CosineSchedule,
        "restart" => RestartSchedule,
    )
    not_implemented = ["cyclic", "sequence"]
    if name ∈ not_implemented
        error("Nested schedules (like $name) are not yet supported")
    end
    sched_type = get(sched_map, name, nothing)
    if sched_type === nothing
        error("Unknown schedule type $name, try one of $(keys(sched_map))")
    end
    sched_type
end


"""
    schedules = get_schedules(optimisers, schedule_type = ConstantSchedule, schedule_kwargs = ())

Initialize schedules for each optimiser in `optimisers`.

# Parameters
- Optimisers (`optimisers`): Iterable of optimiser state trees to build schedules for.
- Schedule type (`schedule_type`): The type of schedules to construct.
- Schedule keyword arguments (`schedule_kwargs`): Keyword arguments for the schedules.
"""
function get_schedules(optimisers, schedule_type=ConstantSchedule, schedule_kwargs=())
    schedules = []
    for opt in optimisers
        push!(schedules, setup(schedule_type, opt; schedule_kwargs...))
    end
    schedules
end


"""
    lr_tree = get_true_lr(schedule, optim)

Returns a tree containing per-parameter LRs for `optim``, determined either by
`schedule` or, for LDoG optimisers, by the LDoG state.
"""
function get_true_lr(schedule, optim, step)
    if isa(optim.mod.log_std.rule, LDoG)
        ExtraOptimisers.getlr(optim)
    else
        getlr(schedule, step)
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
    loss_vanilla_pg(;mod, rewards, actions)

REINFORCE policy gradient loss for modulator
"""
function loss_vanilla_pg(;mod, reward, actions)
    log_probs = sum.(eachcol(log.(exp.(logpdf(mod.policy, actions)) .+ mod.lambda_prob)))
    baseline = mean(reward)
    loss = -mean(log_probs .* (reward .- baseline))
    loss
end


"""
    loss_ks_uniform(;mod)

Kolmogorov-Smirnov loss for modulator
"""
function loss_ks_uniform(;mod)
    empirical_cdf(x, data) = vec(sum(data' .<= x, dims=2) ./ length(data))

    # Rescale constellation points to [0, 1]
    means = mod.policy.means
    means_u = (means .- minimum(means, dims=2)) ./ (maximum(means, dims=2) - minimum(means, dims=2))
    # Sort points by each dimension
    idxs = vcat(sortperm.(eachrow(means_u))'...)
    means_sorted = vcat([means_u[i,idx] for (i, idx) in enumerate(eachrow(idxs))]'...)
    # Compute empirical CDFs
    x = collect(0:100) / 100
    cdf_means = [empirical_cdf(x, row) for row in eachrow(means_sorted)]
    # Compute KS statistic for each dimension
    ks_stats = [maximum(abs.(cdf_ .- x)) for cdf_ in cdf_means]
    sum(ks_stats) / length(ks_stats)
end


"""
    loss_crossentropy(;logits, target)

Cross-Entropy loss for demodulator
"""
function loss_crossentropy(;logits, target)
    Flux.logitcrossentropy(logits, target, dims=1, agg=mean)
end


function _myargsort(x::AbstractMatrix)::AbstractMatrix{CartesianIndex{2}}
    p = sortperm.(eachcol(x))
    reduce(hcat, [CartesianIndex.(c, j) for (j, c) in enumerate(p)])
end
"""
    loss_diversity(constellation)

Return the diversity loss, mean ratio of distance to nearest neighbor and distance to
second nearest neighbor for each constellation point.

# Parameters
- Constellation (`constellation`): 2xN matrix of IQ constellation points
"""
function loss_diversity(constellation)
    # Get distance of each constellation point from every other
    cT = transpose(constellation)
    diff = reshape(cT, (:, 2, 1)) .- reshape(constellation, (1, 2, :))
    dists = (@. diff[:, 1, :] ^ 2 + diff[:, 2, :] ^ 2) |> cpu
    # For small matrices, sorting on CPU is faster
    local sdists
    ignore_derivatives() do
        # We want to ignore self-distance for sorting
        dists[dists .== 0] .= Inf32
        sdists = _myargsort(dists)
    end
    # Ratio of 2nd nearest neighbor to nearest should approach 1 for evenly spaced points
    mean(dists[sdists[2, :]] ./ dists[sdists[1, :]])
end


"""
    symbols_to_onehot(data_sb)

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
    logits_to_symbols_si(logits)

Convert logits to 0-indexed integers for symbol values
"""
function logits_to_symbols_si(logits)
    symbs_si = getindex.(argmax(logits, dims=1), 1) .- 1
    dropdims(symbs_si, dims=1)
end


"""
    logits_to_symbols_sb(logits, bits_per_symbol)

Convert logits to array of symbol bits
"""
function logits_to_symbols_sb(logits, bits_per_symbol)
    integers_to_symbols(logits_to_symbols_si(logits), bits_per_symbol)
end


"""
    normlogpdf(μ, σ, x; ϵ = 1.0f-8)

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
