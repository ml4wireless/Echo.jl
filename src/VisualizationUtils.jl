module VisualizationUtils
export plot_demod_grid_unicode, plot_mod_constellation_unicode, plot_bers_train_unicode, plot_bers_snr_unicode, plot_bers_snr_unicode!
export plot_bers_train, plot_bers_snr
export modconstellation, demodgrid
export animate_mod_constellations, animate_demod_grids, animate_agents

using UnicodePlots
using ColorSchemes
using Plots
using Measures
using RecipesBase
using LinearAlgebra
using Flux: softmax

using ..ModulationModels
using ..DataUtils
using ..LookupTableUtils
using ..ResultsUtils


###############################
# COLORS
###############################
function get_colormap(nvals)
    if nvals <= 20
        if nvals <= 4
            colormap = :Dark2_4
        elseif nvals <= 6
            colormap = :Dark2_6
        elseif nvals <= 8
            colormap = :Dark2_8
        else
            colormap = :tab20b
        end
        return ColorSchemes.colorschemes[colormap]
    else
        return cgrad(:turbo, LinRange(0, 1, nvals + 1), categorical=true).colors
    end
end

function get_colormap_entries(nvals)
    # colormap = get_colormap(nvals)
    # colors = colorschemes[colormap].colors
    # [(c.r, c.g, c.b) for c in colors[1:nvals]]
    color_symbs = [:red, :green, :blue, :black, :cyan, :magenta, :purple, :brown]
    idxs = collect(0:nvals - 1) .% length(color_symbs) .+ 1
    color_symbs[idxs]
end


###############################
# DEMOD GRID
###############################
function get_demod_grid(demod, lims=[-1.5, 1.5], density=100)
    row = collect(range(lims[1], lims[2], length=density))
    grid = permutedims(hcat(repeat(row, inner=density), repeat(row, outer=density)))
    symbs_si = demodulate(demod, grid)
    grid, symbs_si
end

function get_demod_grid_probs(demod, lims=[-1.5, 1.5], density=100)
    row = collect(range(lims[1], lims[2], length=density))
    grid = permutedims(hcat(repeat(row, inner=density), repeat(row, outer=density)))
    logits = demodulate(demod, grid, soft=true)
    symbs_si = argmax.(eachcol(logits)) .- 1
    probs = softmax(logits, dims=1)[CartesianIndex.(symbs_si .+ 1, 1:length(symbs_si))]
    grid, symbs_si, probs
end


"""
    demodgrid(demod; labeled=true, density=100, lims=(-1.5, 1.5))
Plot the demodulation field of a demodulator.
If `labeled` is true, label the maximum probability point of each symbol.
Sample the grid with density points across each dimension, from lims[1] to lims[2].
"""
@userplot DemodGrid
@recipe function f(dg::DemodGrid; labeled=true, density=100, lims=(-1.5, 1.5))
    if !(isa(dg.args[1], NeuralDemod) || isa(dg.args[1], ClassicDemod) || isa(dg.args[1], ClusteringDemod))
        error("Expected a Neural, Clustering, or Classic Demod for demod grid")
    end
    demod = dg.args[1]

    # Get demod grid values
    _, symbols, probs = get_demod_grid_probs(demod, lims, density)
    cell_colors = get_colormap(2 ^ demod.bits_per_symbol)
    bit_probs = []
    colorgrads = []
    for b in 0:2^demod.bits_per_symbol-1
        push!(bit_probs, reshape([symbols[i] == b ? probs[i] : missing for i in eachindex(symbols)],
                                 (density, density)))
        push!(colorgrads, cgrad([RGBA(cell_colors[b+1].r, cell_colors[b+1].g, cell_colors[b+1].b, 0.1),
                                 RGBA(cell_colors[b+1].r, cell_colors[b+1].g, cell_colors[b+1].b, 1.)]))
    end
    grid_1axis = LinRange(lims[1], lims[2], density)

    # Setup default plot parameters
    # --> means set unless specified
    # := means force
    grid --> false
    legend --> false
    xlim --> lims
    ylim --> lims
    aspect_ratio --> :equal
    title --> "Demod Grid"
    seriestype := :heatmap
    colorbar --> false

    # Add each bit's own heatmap to same axes
    for (z, colors) in zip(bit_probs, colorgrads)
        @series begin
            color := colors
            grid_1axis, grid_1axis, z
        end
    end

    # Add peak prob labels
    if labeled
        valid_symbs = [!all(ismissing.(bp)) for bp in bit_probs]
        peaks = map(x -> argmax(skipmissing(x)), bit_probs[valid_symbs])
        labels = map(string, 0:2^demod.bits_per_symbol-1)
        @series begin
            seriestype := :scatter
            markeralpha := 0
            series_annotations := [(l, 11) for l in labels]
            [grid_1axis[p.I[2]] for p in peaks], [grid_1axis[p.I[1]] for p in peaks]
        end
    end

    # Add cluster centers
    if isclustering(demod)
        @series begin
            seriestype := :scatter
            markercolor := cell_colors.colors[1:2 ^ demod.bits_per_symbol]
            markershape := :x
            demod.centers[1, :], demod.centers[2, :]
        end
    end
end


###############################
# MOD CONSTELLATION
###############################
"""
    covellipse(μ, Σ; showaxes=false, n_std=1, n_ellipse_vertices=100)
Plot a confidence ellipse of the 2×2 covariance matrix `Σ`, centered at `μ`.
The ellipse is the contour line of a Gaussian density function with mean `μ`
and variance `Σ` at `n_std` standard deviations.
If `showaxes` is true, the two axes of the ellipse are also plotted.

From StatsPlots.jl, https://github.com/JuliaPlots/StatsPlots.jl/blob/master/src/covellipse.jl
"""
@userplot CovEllipse

@recipe function f(c::CovEllipse; showaxes = false, n_std = 1, n_ellipse_vertices = 100)
    μ, S = _covellipse_args(c.args; n_std = n_std)

    θ = range(0, 2π; length = n_ellipse_vertices)
    A = S * [cos.(θ)'; sin.(θ)']

    @series begin
        seriesalpha --> 0.3
        Shape(μ[1] .+ A[1, :], μ[2] .+ A[2, :])
    end
    showaxes && @series begin
        label := false
        linecolor --> "gray"
        ([μ[1] + S[1, 1], μ[1], μ[1] + S[1, 2]], [μ[2] + S[2, 1], μ[2], μ[2] + S[2, 2]])
    end
end

function _covellipse_args(
    (μ, Σ)::Tuple{AbstractVector{<:Real},AbstractMatrix{<:Real}};
    n_std::Real,
)
    size(μ) == (2,) && size(Σ) == (2, 2) ||
        error("covellipse requires mean of length 2 and covariance of size 2×2.")
    λ, U = eigen(Σ)
    μ, n_std * U * diagm(.√λ)
end
_covellipse_args(args; n_std) = error(
    "Wrong inputs for covellipse: $(typeof.(args)). " *
    "Expected real-valued vector μ, real-valued matrix Σ.",
)

"""
    modconstellation(mod; labeled=true, sampled=-1)
Plot the constellation points of a modulator.
If `labeled` is true, label each point.
If `sampled` > 0, instead of std_dev ellipses, sample a cloud of size `sampled` points and plot them all.
"""
@userplot ModConstellation
@recipe function f(mc::ModConstellation; labeled=true, sampled=-1)
    if !(isa(mc.args[1], NeuralMod) || isa(mc.args[1], ClassicMod))
        error("Expected a NeuralMod or ClassicMod for constellation plot")
    end
    mod = mc.args[1]

    # Setup default plot parameters
    # --> means set unless specified
    # := means force
    grid --> true
    legend --> false
    xlim --> (-1.5, 1.5)
    ylim --> (-1.5, 1.5)
    aspect_ratio --> :equal
    title --> "Mod Constellation"

    # Calculate scatter points
    symbs = get_all_unique_symbols(mod.bits_per_symbol)
    symbs_si = symbols_to_integers(symbs)
    points = modulate(mod, symbs, explore=false)
    nsymbs = 2 ^ mod.bits_per_symbol
    colors = get_colormap(nsymbs)

    # Plot constellation points
    my_annotations = collect(zip(
        string.(symbs_si),
        repeat([:left], length(symbs_si)),
        repeat([:bottom], length(symbs_si))))
    @series begin
        seriestype := :scatter
        if labeled
            series_annotations := my_annotations
        end
        markercolor := colors.colors[1:nsymbs]
        markertype --> :circle
        markersize --> 5
        points[1, :], points[2, :]
    end

    # Plotting for exploration
    if !isclassic(mod)
        if sampled > 0
            # Plot sampled exploration points
            sampled = Int(round(sampled))  # Force integer since recipes don't allow type annotation
            points_per_symb = sampled ÷ mod.bits_per_symbol
            for (i, symb) in enumerate(eachcol(mod.all_unique_symbols))
                iq = modulate(mod, reshape(repeat(symb, outer=points_per_symb), (2, :)), explore=true)
                @series begin
                    seriestype := :scatter
                    markercolor := colors.colors[i]
                    seriesalpha --> 0.5
                    markertype --> :circle
                    markersize --> 3
                    iq[1, :], iq[2, :]
                end
            end
        else
            # Plot sampling std_dev ellipses
            std_devs = exp.(mod.log_std)
            for i in 1:length(symbs_si)
                μ, S = _covellipse_args((points[:, i], [std_devs[1] 0; 0 std_devs[2]]), n_std = 1)
                θ = range(0, 2π; length = 100)
                A = S * [cos.(θ)'; sin.(θ)']

                @series begin
                    seriesalpha --> 0.3
                    color := colors[i]
                    Shape(μ[1] .+ A[1, :], μ[2] .+ A[2, :])
                end
            end
        end
    end
end


###############################
# UNICODE MOD, DEMOD, BER
###############################
function plot_demod_grid_unicode(demod; lims=[-1.5, 1.5], density=100, title="Demod Grid")
    _, symbols_si = get_demod_grid(demod, lims, density)
    symbs_mat = reshape(symbols_si, (density, density))
    offset = -(lims[2] - lims[1]) / 2
    scale = (lims[2] - lims[1]) / (density - 1)
    nsyms = length(unique(symbols_si))
    colormap = get_colormap((nsyms))
    UnicodePlots.heatmap(symbs_mat, colormap=colormap, colorbar=false, height=45, width=60,
            xoffset=offset, yoffset = offset, xfact=scale, yfact=scale, title=title)
end

function plot_mod_constellation_unicode(mod; lims=[-1.5, 1.5], title="Mod Constellation")
    points = modulate(mod, get_all_unique_symbols(mod.bits_per_symbol), explore=false)
    colors = get_colormap_entries(2 ^ mod.bits_per_symbol)
    UnicodePlots.scatterplot(points[1,:], points[2,:], marker=:circle, color=colors,
                xlim=lims, ylim=lims, title=title)
end

function plot_bers_train_unicode(results; roundtrip::Bool=true)
    bers = trainbers(results; roundtrip)
    epochs = sort(collect(keys(results)))
    UnicodePlots.lineplot(epochs, bers, xlabel="Epoch", ylabel="BER",
                          width=80)
end

function plot_bers_snr_unicode(bers::Vector; roundtrip::Bool=true, bits_per_symbol::Int=2, snrs=nothing, label=nothing)
    if snrs === nothing
        snrs = roundtrip ? get_test_SNR_dbs_roundtrip(bits_per_symbol) : get_test_SNR_dbs(bits_per_symbol)
    end
    data = hcat(snrs, bers)[bers .!= 0, :]
    UnicodePlots.lineplot(data[:, 1], data[:, 2], xlabel="SNR", ylabel="BER", width=80, yscale=:log10,
                          ylim=(8e-6, 0.6), xlim=(minimum(snrs), maximum(snrs)), name=label
                          )
end

function plot_bers_snr_unicode!(p, bers::Vector; roundtrip::Bool=true, bits_per_symbol::Int=2, snrs=nothing, label=nothing)
    if snrs === nothing
        snrs = roundtrip ? get_test_SNR_dbs_roundtrip(bits_per_symbol) : get_test_SNR_dbs(bits_per_symbol)
    end
    data = hcat(snrs, bers)[bers .!= 0, :]
    UnicodePlots.lineplot!(p, data[:, 1], data[:, 2], name=label)
end



###############################
# TRAINING ANIMATIONS
###############################
function animate_mod_constellations(results; agent_id::Int=1, only_epochs=nothing, fps=5, filename="mod_const_training.gif")
    if only_epochs !== nothing
        epochs = sort(only_epochs)
    else
        epochs = sort(collect(keys(results)))
    end
    anim = @animate for e in epochs
        cfg = results[e][:kwargs][Symbol("agent$agent_id")]
        mod = NeuralMod(;cfg.mod...)
        modconstellation(mod, top_margin=5Plots.mm)
        Plots.annotate!([(-1.45, 1.5, Plots.text("Epoch $e", :left, :bottom))])

    end
    gif(anim, filename, fps=fps)
end

function animate_demod_grids(results; agent_id::Int=1, only_epochs=nothing, fps=5, filename="demod_grid_training.gif")
    if only_epochs !== nothing
        epochs = sort(only_epochs)
    else
        epochs = sort(collect(keys(results)))
    end
    anim = @animate for e in epochs
        cfg = results[e][:kwargs][Symbol("agent$agent_id")]
        demod = NeuralDemod(;cfg.demod...)
        demodgrid(demod, top_margin=5Plots.mm)
        Plots.annotate!([(-1.45, 1.5, Plots.text("Epoch $e", :left, :bottom))])
    end
    gif(anim, filename, fps=fps)
end

function animate_agents(results; only_epochs=nothing, fps=5, filename="agent_training.gif", show=true, agent_ids=[1, 2])
    if only_epochs !== nothing
        epochs = sort(only_epochs)
    else
        epochs = sort(collect(keys(results)))
    end
    agent1 = Symbol("agent$(agent_ids[1])")
    agent2 = Symbol("agent$(agent_ids[2])")
    # TODO: animate all agents, with backgrounds to identify m/d pairs
    anim = @animate for e in epochs
        cfg = results[e][:kwargs]
        mod1 = Modulator(;cfg[agent1].mod...)
        demod2 = Demodulator(;cfg[agent2].demod...)
        roundtrip = false
        if cfg[agent2].mod != (;)
            roundtrip = true
            mod2 = Modulator(;cfg[agent2].mod...)
            demod1 = Demodulator(;cfg[agent1].demod...)
        end
        pm1 = modconstellation(mod1, top_margin=5Plots.mm, title="Mod 1")
        Plots.annotate!([(-1.45, 1.5, Plots.text("Epoch $e", :left, :bottom))])
        pd2 = demodgrid(demod2, title="Demod 2")
        if roundtrip
            pm2 = modconstellation(mod2, title="Mod 2")
            pd1 = demodgrid(demod1, title="Demod 1")
            plot(pm1, pd2, pd1, pm2, layout=grid(2, 2))
        else
            plot(pm1, pd2, layout=grid(1, 2))
        end
    end
    animation = gif(anim, filename, fps=fps);
    if show
        return animation
    else
        return nothing
    end
end


###############################
# BERS
###############################
function replacezeros(x::AbstractArray)
    y = similar(x, Union{Missing, eltype(x)})
    y[:] .= x[:]
    y[x .== 0] .= missing
    y
end

"""
    berplot(bers; snrs=nothing, bits_per_symbol=2, bps_ref=true, roundtrip=true)
Plot the BER vs SNR curve.
If `snrs` is `nothing`, assume standard test SNRs.
If `bps_ref` is true, include a classic reference curve.
`roundtrip` determines whether reference BERs are half- or round-trip.
"""
@userplot BERPlot
@recipe function f(bp::BERPlot; snrs=nothing, bits_per_symbol=2, bps_ref=true, roundtrip=true)
    bers = bp.args[1]
    @assert ndims(bers) == 1 "`bers` must be a vector"
    bers = replacezeros(bers)
    if snrs === nothing
        snrs = roundtrip ? get_test_SNR_dbs_roundtrip(bits_per_symbol) : get_test_SNR_dbs(bits_per_symbol)
    end

    # Setup default plot parameters
    # --> means set unless specified
    # := means force
    title --> "BER vs SNR(dB)"
    yscale --> :log10
    xlabel --> "SNR (dB)"
    ylabel --> "BER"
    ylim --> [1e-5, 0.6]
    yticks --> 10. .^ collect(-1:-1:-5)
    xticks --> snrs
    # Plot measured bers
    @series begin
        seriestype := :line
        label --> "Measured BER"
        linewidth --> 2
        snrs, bers
    end
    # Plot reference bers
    if bps_ref
        if roundtrip
            ref_snrs, ref_bers = get_optimal_BERs_roundtrip(minimum(snrs), maximum(snrs), bits_per_symbol)
        else
            ref_snrs, ref_bers = get_optimal_BERs(minimum(snrs), maximum(snrs), bits_per_symbol)
        end
        ref_bers = replacezeros(ref_bers)
        @series begin
            seriestype := :line
            label --> "Optimal, BPS=$bits_per_symbol"
            color := "grey"
            ref_snrs, ref_bers
        end
    end
end

function plot_bers_train(results; filename="training_bers.png", show=true)
    epochs = sort(collect(keys(results)))
    bers = reduce(hcat, results[e][:ber][:, 5] for e in epochs)
    bers[bers .== 0.] .= NaN
    p = plot(epochs, bers', labels=["ht1" "ht2" "rt"],
         yscale=:log10,
         yticks=(10. .^ [0, -1, -2, -3, -4, -5]),
         xlabel="Epoch",
         ylabel="BER",
         linewidth=[1 1 2],
         )
    plot!([0, maximum(epochs)], [0.5, 0.5], linestyle=:dot, color=:grey, label="BER = 1/2")
    Plots.savefig(p, filename)
    @info "Saved train BER plot" filename
    if show
        return p
    else
        return nothing
    end
end

function plot_bers_snr(bers; bits_per_symbol=2, filename="final_bers.png", roundtrip=true, show=true)
    p = berplot(bers, bits_per_symbol=bits_per_symbol, roundtrip=roundtrip)
    Plots.savefig(p, filename)
    @info "Saved final BER plot" filename
    if show
        return p
    else
        return nothing
    end
end


end