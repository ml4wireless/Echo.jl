module ExtraOptimisers
export Yogi, MADGrad, LDoG, Prodigy
import Optimisers

using Functors: fmap, isleaf
using LinearAlgebra: norm, dot


"""
    Yogi(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = 1f-3)

[Yogi](https://papers.nips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf) optimiser

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Epsilon (`ϵ`): controls amount of adaptivity in algorithm, recommended 1e-3
"""
struct Yogi{T} <: Optimisers.AbstractRule
    eta::T
    beta::Tuple{T, T}
    epsilon::T
end
Yogi(η = 1f-3, β = (9f-1, 9.99f-1), ϵ = 1f-3) = Yogi{typeof(η)}(η, β, ϵ)

function Optimisers.init(o::Yogi, x::AbstractArray)
    # State: m_t, v_t, β, t
    return (zero(x), zero(x), o.beta, 0)
end

function Optimisers.apply!(o::Yogi, state, x, dx)
    η, β, ϵ = o.eta, o.beta, o.epsilon
    mt, vt, βt, t = state

    Optimisers.@.. mt = β[1] * mt + (1 - β[1]) * dx
    dx2 = abs2.(dx)
    if t == 0
        # Initialize v_t with first minibatch instead of 0, per paper
        vt = dx2
    else
        Optimisers.@.. vt = vt - (1 - β[2]) * sign(vt - dx2) * dx2
    end
    dx′ = Optimisers.@lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η

    return (mt, vt, βt .* β, t + 1), dx′
end


"""
    MADGrad(η = 1f-3, c = 1f-1, ϵ = 1f-6)

[MADGRAD](https://arxiv.org/abs/2101.11075) optimiser

This optimiser maintains state equal to 3x the size of optimised params `x`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`momentum`): Momentum for gradient estimates.
- Decay (`decay`): Weight decay, i.e. L2 penalty.
- Machine epsilon (`ϵ`): Constant to prevent division by zero, may help for
                         problems with very small gradients to set to 0.
- Decouple decay (`decouple_decay`): Apply AdamW style decoupled weight decay

https://github.com/facebookresearch/madgrad/blob/main/madgrad/madgrad.py
"""
struct MADGrad{T} <: Optimisers.AbstractRule
    eta::T
    momentum::T
    decay::T
    epsilon::T
    decouple_decay::Bool
end
MADGrad(η = 1f-3, momentum = 9f-1, decay = 0f0, ϵ = 1f-6, decouple_decay = true) = MADGrad{typeof(η)}(η, momentum, decay, ϵ, decouple_decay)

Optimisers.init(::MADGrad, x::AbstractArray) = (deepcopy(x), zero(x), zero(x), 0)

function Optimisers.apply!(o::MADGrad, state, x, dx)
    η, momentum, decay, ϵ, decouple_decay = o.eta, o.momentum, o.decay, o.epsilon, o.decouple_decay
    x0, sk, vk, k  = state

    if η != 0.0
        η = η + ϵ  # For stability
    end

    ck = 1 - momentum
    λk = η * sqrt(k + 1)

    # Apply weight decay
    if decay != 0 && !decouple_decay
        Optimisers.@.. dx = dx + decay * x
    end

    Optimisers.@.. sk = sk + λk * dx
    Optimisers.@.. vk = vk + λk * abs2(dx)
    zk = @. x0 - sk / (cbrt(vk) + ϵ)
    # TODO: handle cuberoot zeros if ϵ==0
    x′ = Optimisers.@lazy (1 - ck) * x + ck * zk

    # Apply decoupled weight decay
    penalty = zero(eltype(x))
    if decay != 0 && decouple_decay
         penalty = Optimisers.@lazy η * decay * x
    end

    return (x0, sk, vk, k + 1), x .- x′ .- penalty
end


"""
    LDoG(rϵ = 1f-4, c = 1.0, weight_decay = 0.0, ϵ = 1f-8)

[LDoG](https://arxiv.org/pdf/2302.12022.pdf) optimiser

# Parameters
- rϵ (`rϵ`): Value used to compute the initial distance.
             Namely, the first step size is given by:
                (rϵ * (1+\\|x_0\\|)) / (\\|g_0\\|^2 + ϵ)^{1/2}  where x_0 are the initial
                weights of  the model (or the parameter group), and g_0 is the gradient of the
                first step.
                As discussed in the paper, this value should be small enough to ensure that the
                first update step will be small enough to not cause the model to diverge.
                Suggested value is 1f-6, unless the model uses batch-normalization,
                in which case the suggested value is 1f-4. (default: 1f-6).
- Learning rate (`c`): Implicit learning rate referred to as c in the paper. Changing this
                       from 1.0 is not recommended, and only the range 0.5-1.5 is reliable.
- Weight decay (`weight_decay`): L2 penalty, weight_decay * x_t is added directly to the gradient.
- Epsilon (`ϵ`): Machine epsilon used for numerical stability, added to the sum of gradients.
"""
struct LDoG{T} <: Optimisers.AbstractRule
    eta::T  # This is rϵ, but called eta to match other Rule implementations.
    c::T
    weight_decay::T
    ϵ::T
end
LDoG(rϵ = 1f-4, c = 1f0, weight_decay = 0f0, ϵ = 1f-8) = LDoG{typeof(rϵ)}(rϵ, c, weight_decay, ϵ)

Optimisers.init(::LDoG, x::AbstractArray) = (deepcopy(x), 0f0, 0f0, 0)

function Optimisers.apply!(o::LDoG, state, x, dx)
    rϵ, c, weight_decay, ϵ = o.eta, o.c, o.weight_decay, o.ϵ
    x0, rbar, Gsqr, t = state

    if t == 0
        # First step set η = rϵ / |g0|
        G = norm(dx)
        η = rϵ / (G + ϵ)
        Gsqr = G ^ 2
    else
        rbar = max(rbar, norm(x - x0))
        Gsqr += sum(abs2.(dx))
        η = c * rbar / (sqrt(Gsqr) + ϵ)
    end

    penalty = zero(eltype(x))
    if weight_decay > 0
        penalty = @. weight_decay * x
    end
    dx′ = Optimisers.@lazy η * (dx + penalty)

    return (x0, rbar, Gsqr, t + 1),  dx′
end

isOptimLeafOrLeaf(l) = isa(l, Optimisers.Leaf) || isleaf(l)

"""
Returns current η for LDoG state.
"""
getlr(l::Optimisers.Leaf) = l.state[end] > 0 ? l.rule.c * l.state[2] / (sqrt(l.state[3]) + l.rule.ϵ) : l.rule.eta
getlr(o::NamedTuple) = fmap(getlr, o, exclude=isOptimLeafOrLeaf)
getlr(::Tuple{}) = ()


"""
    Lion(eta = 1f-4, beta = (9f-1, 9.9f-1), weight_decay = 0f0)

[Lion](https://arxiv.org/abs/2302.06675) optimiser -- already implemented in Optimisers.jl

# Parameters
- Learning rate (`eta`): Amount by which gradients are discounted before updating
                         the weights.
- Beta (`beta`): Tuple of (beta1, beta2), representing gradient discounting for
                 adjusted gradient `sign(c_t)` and momentum `m_t` updates.
- Weight decay (`weight_decay`):  L2 penalty, eta * weight_decay * x_t is added
                                  directly to the gradient. Adjusting eta requires
                                  adjusting this value inversely.
"""
struct Lion{T} <: Optimisers.AbstractRule
    eta::T
    beta::Tuple{T, T}
    weight_decay::T
end
Lion(η = 1f-4, β = (9f-1, 9.9f-1), weight_decay = 0f0) = Lion{typeof(η)}(η, β, weight_decay)

Optimisers.init(::Lion, x::AbstractArray) = (zero(x),)

function Optimisers.apply!(o::Lion, state, x, dx)
    η, β, λ = o.eta, o.beta, o.weight_decay
    (mt,) = state

    # Weight decay penalty
    penalty = zero(eltype(x))
    if λ > 0
        penalty = @. λ * x
    end

    # Sign(c_t) grad update
    dx′ = Optimisers.@lazy η * (sign(β[1] * mt + (1 - β[1]) * dx) + penalty)

    # Momentum update
    Optimisers.@.. mt = β[2] * mt + (1 - β[2]) * dx

    return (mt,), dx′
end


"""
    Prodigy(eta = 1.0, beta = (9f-1, 9.99f-1), beta3 = nothing, d_coef=1f0, weight_decay = 0f0)

[Prodigy](https://arxiv.org/abs/2306.06101) optimiser

WARNING: This is a layer-wise implementation because Optimisers.jl does not have a mechanism
for globally calculated parameters. It may be possible to get around this by using a flattened
parameter vector (with `destructure`) during setup and update calls, but this is untested.

# Parameters
- Learning rate (`eta`): Amount by which gradients are discounted before updating
                         the weights. Should stay 1.0, learning rates can be tuned with
                         `d_coef` instead.
- Beta (`beta`): Tuple of (beta1, beta2), representing gradient discounting for running
                 averages of gradients and squared gradients.
- Beta3 (`beta3`): Optional third beta for running average of gradients. Defaults to
                   sqrt(beta2).
- Epsilon (`eps`): Small constant to prevent division by zero.
- Weight decay (`weight_decay`):  L2 penalty, weight_decay * x_t is added
                                  directly to the gradient.
- Decouple (`decouple`): Apply AdamW style decoupled weight decay, default=true.
- Use bias correction (`use_bias_correction`): Turn on Adam's bias correction. Off by default.
- Safeguard warmup (`safeguard_warmup`): Remove lr from the denominator of D estimate to
                                         avoid issues during warm-up stage. Off by default.
- d0 (`d0`): Initial value for D estimate. Defaults to 1f-6. Rarely needs to be changed.
- D coefficient (`d_coef`): Coefficient for D estimate. Defaults to 1.0. Can be used to
                            tune learning rates; d_coef < 1 decreases estimates, d_coef > 1
                            increases estimates. Changing this parameter is the preferred way
                            to tune learning rates.
- Growth rate (`growth_rate`): Prevent the D estimate from growing faster than this
                               multipicative rate. Default is Inf, unrestricted. Values like
                               1.02 give a kind of learning rate warmup effect.
"""
struct Prodigy{T} <: Optimisers.AbstractRule
    eta::T
    beta::Tuple{T, T}
    beta3::T
    ϵ::T
    weight_decay::T
    decouple::Bool
    use_bias_correction::Bool
    safeguard_warmup::Bool
    d0::T
    d_coef::T
    growth_rate::T
end
function Prodigy(η=1f0, β=(9f-1, 9.99f-1), β3=nothing, d_coef=0.75f0, weight_decay=0f0, growth_rate=Inf)
    if β3 === nothing
        β3 = sqrt(β[2])
    end
    Prodigy{typeof(η)}(
        η, β, β3, typeof(η)(1e-8), weight_decay, true,
        false, false, typeof(η)(1e-6), d_coef, typeof(η)(growth_rate))
end

# Prodigy state: x0, m, v, s, d, d_max, d_numerator, t
Optimisers.init(p::Prodigy, x::AbstractArray) = (deepcopy(x), zero(x), zero(x), zero(x), p.d0, p.d0, 0f0, 0)

function Optimisers.apply!(o::Prodigy, state, x, dx)
    x0, m, v, s, d, d_max, d_numerator, t = state

    d_numerator *= o.beta3

    # Bias correction for Adam
    if o.use_bias_correction
        bias_correction = (sqrt(one(eltype(x)) - o.beta[2] ^ (t + 1))) / (one(eltype(x)) - o.beta[1] ^ (t + 1))
    else
        bias_correction = one(eltype(x))
    end
    d_lr = d * o.eta * bias_correction

    # Weight decay penalty: coupled
    if o.weight_decay > 0 && !o.decouple
        Optimisers.@.. dx = dx + o.weight_decay * x
    end

    # Use d/d0 to avoid values that are too small
    d_numerator += (d / o.d0) * d_lr * dot(vec(dx), vec(x0 - x))

    # Adam EMA updates
    Optimisers.@.. m = o.beta[1] * m + d * (one(eltype(x)) - o.beta[1]) * dx
    Optimisers.@.. v = o.beta[2] * v + d * d * (one(eltype(x)) - o.beta[2]) * dx * dx

    if o.safeguard_warmup
        Optimisers.@.. s = o.beta3 * s + (d / o.d0) * d * dx
    else
        Optimisers.@.. s = o.beta3 * s + (d / o.d0) * d_lr * dx
    end

    d_denom = norm(s, 1)
    if d_denom == 0
        # If gradients exist, they are 0 so we can't update parameters
        return (x0, m, v, s, d, d_max, d_numerator, t + 1), zero(x)
    end

    d_hat = o.d_coef * d_numerator / d_denom

    # Clip d to growth rate
    if d == o.d0
        d = max(d, d_hat)
    end
    d_max = max(d_max, d_hat)
    d = min(d_max, d * o.growth_rate)

    # Calculate final update
    dx′ = zero(x)
    if o.weight_decay > 0 && o.decouple
        Optimisers.@.. dx′ = dx′ + x * o.weight_decay * d_lr
    end
    denom = @. sqrt(v) + d * o.ϵ
    Optimisers.@.. dx′ = dx′ + d_lr * m / denom

    return (x0, m, v, s, d, d_max, d_numerator, t + 1), dx′
end


end