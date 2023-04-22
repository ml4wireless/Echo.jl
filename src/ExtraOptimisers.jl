module ExtraOptimisers
export Yogi, MADGrad, LDoG
import Optimisers

using LinearAlgebra: norm

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

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`momentum`): Momentum for gradient estimates.
- Decay (`decay`): Weight decay, i.e. L2 penalty.
- Machine epsilon (`ϵ`): Constant to prevent division by zero, may help for
                         problems with very small gradients to set to 0.
- Decouple Decay (`decouple_decay`): Apply AdamW style decoupled weight decay

https://github.com/facebookresearch/madgrad/blob/main/madgrad/madgrad.py
"""
struct MADGrad{T} <: Optimisers.AbstractRule
    eta::T
    momentum::T
    decay::T
    epsilon::T
    decouple_decay::Bool
end
MADGrad(η = 1f-3, momentum = 9f-1, decay = 0f0, ϵ = 1f-6, decouple_decay = false) = MADGrad{typeof(η)}(η, momentum, decay, ϵ, decouple_decay)

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
    if decay != 0.0 && !decouple_decay
        Optimisers.@.. dx = dx + decay * x
    end

    Optimisers.@.. sk = sk + λk * dx
    Optimisers.@.. vk = vk + λk * abs2(dx)
    zk = @. x0 - sk / (cbrt(vk) + ϵ)
    # TODO: handle cuberoot zeros if ϵ==0
    x′ = Optimisers.@lazy (1 - ck) * x + ck * zk

    # Apply decoupled weight decay
    penalty = 0f0
    if decay != 0.0 && decouple_decay
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
    rϵ::T
    c::T
    weight_decay::T
    ϵ::T
end
LDoG(rϵ = 1f-4, c = 1.0, weight_decay = 0.0, ϵ = 1f-8) = LDoG{typeof(rϵ)}(rϵ, c, weight_decay, ϵ)

Optimisers.init(::LDoG, x::AbstractArray) = (deepcopy(x), 0f0, 0f0, 0)

function Optimisers.apply!(o::LDoG, state, x, dx)
    rϵ, c, weight_decay, ϵ = o.rϵ, o.c, o.weight_decay, o.ϵ
    x0, rbar, Gsqr, t = state

    if t == 0
        # First step set η = rϵ / |g0|
        G = norm(dx)
        η = rϵ / (G + ϵ)
        Gsqr = G ^ 2
    else
        rbar = max(rbar, norm(x - x0))
        Gsqr += sum(abs2.(dx))
        η = c * rbar / sqrt(Gsqr)
    end

    penalty = 0f0
    if weight_decay > 0f0
        penalty = @. weight_decay * x
    end
    dx′ = Optimisers.@lazy η * (dx + penalty)

    return (x0, rbar, Gsqr, t + 1),  dx′
end

end