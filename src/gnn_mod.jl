# Graph Neural Network [De]Modulator implementation

using LinearAlgebra: norm
using Infiltrator
using ChainRules: ignore_derivatives


"""
    D = node_distances(g)

Returns a matrix of pairwise distances between node values in the graph `g`.
"""
function node_distances(g)
    n = size(g.x, 2)
    D = zeros(Float32, (n, n))
    for i in 1:n-1
        for j in i+1:n
            D[i, j] = norm(g.x[:,i] - g.x[:,j])
            D[j, i] = D[i, j]
        end
    end
    D
end


"""
    neighbors, distances = nearest_neighbors(g, k)

Computes the `k` nearest neighbors for each node in the graph `g`.
Returns a dictionary containing a vector of neighbor indices for each node,
and a dictionary containing a vector of distances to each neighbor.
"""
function nearest_neighbors(g, k)
    n = size(g.x, 2)
    D = node_distances(g)
    nn = zeros(Int, (k, n))
    nd = zeros(Float32, (k, n))
    # @infiltrate
    for i in 1:n
        nn[:, i] = sortperm(D[i, :])[2:k+1]
        nd[:, i] = D[i, nn[:, i]]
    end
    nn, nd
end


"""
    g = nearest_neighbors_subgraph(g, k)

Returns a subgraph of `g` containing edges for only the `k` nearest neighbors of each node.
"""
function nearest_neighbors_subgraph(g, k)
    nn, nd = nearest_neighbors(g, k)
    num_e = 2 * length(nn)
    n = size(g.x, 2)
    source = zeros(Int, num_e)
    target = zeros(Int, num_e)
    e_vec = zeros(Float32, (1, num_e))
    ind = 1
    for i in 1:n
        for j in 1:k
            source[ind:ind+1] = [i, nn[j, i]]
            target[ind:ind+1] = [nn[j, i], i]
            e_vec[1, ind:ind+1] .= nd[j, i]
            ind += 2
        end
    end
    gsub = GNNGraph(source, target)
    gsub.ndata.x = g.x
    gsub.edata.e = e_vec
    gsub
end


##################################################################################
# GNN Modulator
##################################################################################
function _make_graph_for_bps(bps::Int)
    labels = collect(1:2^bps)
    st_pairs = collect(filter(xy -> xy[1] != xy[2], collect(product(labels, labels))))
    s = [st[1] for st in st_pairs]
    t = [st[2] for st in st_pairs]
    g = GNNGraph(s,t)
    # graphplot(g4)
    labels = collect(0:2^bps-1)
    labels_sb = integers_to_symbols(labels, bps)
    labels_sb = @. Float32(labels_sb) * 2 - 1
    g.ndata.x = labels_sb
    g.edata.e = ones(Float32, (1, g.num_edges))
    g
end


struct GNNMod{A <: AbstractVector{Float32}} <: Modulator
    bits_per_symbol::Int
    layer_dims::Vector{<:Integer}
    restrict_energy::Int
    activation_fn_hidden::Function
    activation_fn_output::Function
    avg_power::Float32
    μ::Flux.Chain
    log_std::A
    policy::NeuralModPolicy
    log_std_dict::NamedTuple  # min, max, initial
    lr_dict::NamedTuple  # mu, std
    graph::GNNGraph
    lambda_prob::Float32  # For numerical stability while computing pg loss
    λdiversity::Float32  # Weighting for diversity loss
end


# Only the μ and log_std fields of a NeuralMod struct are trainable
Flux.@functor GNNMod (μ, log_std, policy, graph)
Flux.trainable(m::GNNMod) = (; μ=m.μ, log_std=m.log_std)

function Base.show(io::IO, m::GNNMod)
    argstr = @sprintf("(bps=%d, ld=%s, 𝑓=%s, re=%d, λμ=%g, λσ=%g, λdiv=%g)",
                      m.bits_per_symbol, plain_print_list(m.layer_dims),
                      string(Symbol(m.activation_fn_hidden)), m.restrict_energy,
                      m.lr_dict.mu, m.lr_dict.std, m.λdiversity)
    print(io, color_type(m)..., argstr)
end


"""
Build a GNN model with the given dimensions

Parameters:
- dims: Pair of input and output dimensions
- layer_dims: Vector of hidden layer dimensions
"""
function _build_gnn(dims::Pair{Int}, layer_dims, activation_fn_hidden=relu)
    # Get dimensions for edge and vertex networks
    din = dims[1]
    dout = dims[2]
    nin_e = vcat([2din + 1], [3ld for ld in layer_dims])
    nout_e = vcat([ld for ld in layer_dims], [layer_dims[end]])
    nin_v = vcat([din + layer_dims[1]], [2ld for ld in layer_dims])
    nout_v = vcat([ld for ld in layer_dims], [dout])
    # Setup edge and vertex networks, final chain of conv layers
    ϕe = Chain[]
    ϕv = Chain[]
    for (in, out) in zip(nin_e, nout_e)
        push!(ϕe, Chain(Dense(in, max(in, out), activation_fn_hidden),
                        Dense(max(in, out), out)))
    end
    for (in, out) in zip(nin_v, nout_v)
        push!(ϕv, Chain(Dense(in, max(in, out), activation_fn_hidden),
                        Dense(max(in, out), out)))
    end
    Chain([MEGNetConv(ϕe[i], ϕv[i], aggr=mean) for i in eachindex(ϕe)]...)
end


function GNNMod(;bits_per_symbol,
                layer_dims=nothing, hidden_layers=nothing,
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
    # Allow hidden_layers kwarg for compatibility with existing neuralmod configs
    @assert (layer_dims !== nothing || hidden_layers !== nothing) "Must provide either layer_dims or hidden_layers"
    if layer_dims !== nothing
        layer_dims = Vector{Int}(layer_dims)
    else
        layer_dims = Vector{Int}(hidden_layers)
    end
    # Two stages of the modulator: initial location suggestions followed by nearest-neighbors adjustment
    μ1 = _build_gnn(Pair(bits_per_symbol, 2), layer_dims, activation_fn_hidden)
    # μ2 = _build_gnn(Pair(2, 2), layer_dims, activation_fn_hidden)
    μ = μ1
    # Load weights if specified
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
    # Create base graph and final struct
    graph = _make_graph_for_bps(bits_per_symbol)
    GNNMod(bits_per_symbol, layer_dims,
           restrict_energy, activation_fn_hidden,
           activation_fn_output, avg_power,
           μ, log_std, NeuralModPolicy(),
           log_std_dict, lr_dict,
           graph, lambda_prob,
           Float32(lambda_diversity),
          )
end


get_kwargs(m::GNNMod; include_weights=false) = (;
    :bits_per_symbol => m.bits_per_symbol,
    :layer_dims => m.layer_dims,
    :restrict_energy => m.restrict_energy,
    :activation_fn_hidden => String(Symbol(m.activation_fn_hidden)),
    :avg_power => m.avg_power,
    :log_std_dict => m.log_std_dict,
    :lr_dict => m.lr_dict,
    :lambda_prob => m.lambda_prob,
    :lambda_diversity => m.λdiversity,
    :log_std => include_weights ? deepcopy(cpu(m.log_std)) : nothing,
    :weights => include_weights ? deepcopy(cpu(m.μ)) : nothing,
)


"""
Return the non-normalized constellation points of the modulator
"""
function _unnormed_constellation(mod::GNNMod)
    mod.μ(mod.graph).x
    # # Generate initial constellation suggestion
    # g_suggested = mod.μ[1](mod.graph)
    # # Select nearest neighbor edges
    # g_nn = nearest_neighbors_subgraph(g_suggested, 4)
    # # Refine constellation suggestion
    # g_final = mod.μ[2](g_nn)
    # g_final.x
end


"""
Normalize avg power of constellation to avg_power
Inputs:
mod: NeuralMod for settings
means: constellation points to normalize

Outputs:
normalized means
"""
function normalize_constellation(mod::GNNMod, means)
    avg_power = mean(sum(abs2.(_unnormed_constellation(mod)), dims=1))
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
function normalize_symbols(mod::GNNMod, means)
    avg_power = mean(sum(abs2.(means), dims=1))
    if mod.avg_power > 0.
        norm_factor = sqrt((relu(avg_power - mod.avg_power) + mod.avg_power) / mod.avg_power)
    else
        norm_factor = one(means[1])  # Automatically match type of means elements
    end
    means = means ./ norm_factor
    means
end


"""
Center means, then normalize avg power of constellation to avg_power
Inputs:
mod: NeuralMod for settings
means: constellation points to normalize

Outputs:
centered then normalized means
"""
function center_and_normalize_constellation(mod::GNNMod, means)
    centered_constellation = center_means(_unnormed_constellation(mod))
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
function (m::GNNMod)(symbols::AbstractArray{UInt16})
    symbol_inds = symbols_to_integers(symbols) .+ 1
    means = _unnormed_constellation(m)[:, symbol_inds]
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
constellation(mod::GNNMod) = mod(get_all_unique_symbols(mod.bits_per_symbol, cuda=iscuda(mod)))


"""
Modulate a bps x N symbol message to cartesian coordinates, with policy exploration
"""
modulate(m::GNNMod, symbols::AbstractArray{Float32}; explore::Bool=false) = modulate(m, UInt16.((symbols .+ 1) ./ 2), explore=explore)
function modulate(m::GNNMod, symbols::AbstractArray{UInt16}; explore::Bool=false)
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
function loss(mod::GNNMod; symbols, received_symbols, actions)
    reward = ignore_derivatives() do
        -sum.(eachcol(Float32.(xor.(symbols, received_symbols))))
    end
    loss = loss_vanilla_pg(mod=mod, reward=reward, actions=actions)
    if mod.λdiversity > 0
        loss += mod.λdiversity * loss_diversity(constellation(mod))
    end
    loss
end
