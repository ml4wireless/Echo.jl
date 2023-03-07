module ExtraOptimisers

import Optimisers


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
    # State: m_t, v_t, β
    # Initialize v_t with first minibatch instead of 0, per paper
    println("Initializing Yogi with x = $x")
    return (zero(x), nothing, o.beta)
end

function Optimisers.apply!(o::Yogi, state, x, dx)
    η, β, ϵ = o.eta, o.beta, o.epsilon
    mt, vt, βt = state

    Optimisers.@.. mt = β[1] * mt + (1 - β[1]) * dx
    dx2 = abs2.(dx)
    if vt === nothing
        vt = dx2
    else
        Optimisers.@.. vt = vt - (1 - β[2]) * sign(vt - dx2) * dx2
    end
    dx′ = Optimisers.@lazy m1 / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η

    return (mt, vt, βt .* β), dx′
end


"""
    MADGrad(η = 1f-3, c = 1f-1, ϵ = 1f-6)

[MADGRAD](https://arxiv.org/abs/2101.11075) optimiser

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Momentum (`c`): Momentum for gradient estimates, applied as (1-c)x + cx'
                  instead of usual cx + (1-c)x'.
- Machine epsilon (`ϵ`): Constant to prevent division by zero, may help for
                         problems with very small gradients to set to 0.

TODO: add decoupled decay per pytorch implementation
"""
struct MADGrad{T} <: Optimisers.AbstractRule
    eta::T
    c::T
    decay::T
    epsilon::T
end
MADGrad(η = 1f-3, c = 1f-1, ϵ = 1f-6) = MADGrad{typeof(η)}(η, c, decay, ϵ)

Optimisers.init(o::MADGrad, x::AbstractArray) = (x, zero(x), zero(x), 0)

function Optimisers.apply!(o::MADGrad, state, x, dx)
    η, c, decay, ϵ = o.eta, o.c, o.decay, o.epsilon
    x0, sk, vk, k  = state

    λk = η * sqrt(k + 1)
    Optimisers.@.. sk = sk + λk * dx
    Optimisers.@.. vk = vk + λk * abs2(dx)
    Optimisers.@.. zk = x0 - sk / (cbrt(vk) + ϵ)
    # TODO: handle cuberoot zeros if ϵ==0
    x′ = Optimisers.@lazy (1 - c) * x + c * zk

    return (x0, sk, vk, k + 1), x .- x′
end



end