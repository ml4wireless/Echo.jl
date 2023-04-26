module ResultsUtils
export loadresults, finalbers, trainbers, finalagents, reconstruct_epoch, find_3db_off, find_dboff

using BSON
# Bring Flux and Zygote into module namespace for BSON loading
using Flux
using Zygote

using ..Agents
using ..Protocols
using ..LookupTableUtils
using ..Echo



"""
Load saved results from an experiment run
Returns a NamedTuple of (config, results)
"""
function loadresults(filename)
    res = BSON.load(filename, @__MODULE__)
    (; config=res[:config], results=res[:results])
end


"""
Return final measured BER array from an experiment
"""
function finalbers(results::NamedTuple)
    finalbers(results.results)
end

function finalbers(results::Dict{Int64, Any})
    efinal = maximum(keys(results))
    results[efinal][:ber]
end


"""
    trainbers(results; roundtrip = true, column = 5)

Return {ht,rt} BERs at a given test SNR through training epochs
If not `roundtrip``, return the first halftrip BER measurement.
`column` corresponds to the test SNR, with 5 usually representing target BER 0.01.
"""
function trainbers(results::Dict{Int64, Any}; roundtrip::Bool=true, column::Int=5)
    epochs = sort(collect(keys(results)))
    berdims = (length(epochs), size(results[0][:ber], 3))
    bers = Array{Float32}(undef, berdims)
    irow = roundtrip ? 3 : 1
    for (i, e) in enumerate(epochs)
        bers[i, :] = results[e][:ber][irow, column, :]
    end
    bers
end

trainbers(results::NamedTuple; roundtrip::Bool=true, column::Int=5) = trainbers(results.results; roundtrip, column)


"""
Return agents with state from end of training
"""
function finalagents(results::Dict{Int64, Any})
    efinal = maximum(keys(results))
    agents = Agent[]
    for kwargs in results[efinal][:kwargs]
        push!(agents, Agent(; kwargs...))
    end
    agents
end


"""
Reconstruct training state at a particular epoch
"""
function reconstruct_epoch(results::Dict{Int64, Any}, epoch::Int)
    if epoch ∉ keys(results)
        epochs = collect(keys(results))
        diffs = abs.(epochs .- epoch)
        imin = argmin(diffs)
        throw(KeyError("Epoch $epoch not in results, try $efinal or $(epochs[imin])"))
    end
    bers = results[epoch][:ber]
    agents = Agent[]
    for kwargs in results[epoch][:kwargs]
        push!(agents, Agent(; kwargs...))
    end
    (;agents=agents, bers=bers)
end


"""
Find first `dboff` dB-off BER result and return number of training iterations to reach it
"""
function find_dboff(config, results, dboff; worst_ber=true)
    roundtrip = config.train_kwargs.protocol ∈ [ESP, EPP]
    if roundtrip
        trainSNR = get_optimal_SNR_for_BER_roundtrip(config.train_kwargs.target_ber, config.bps)
        dboffBER = get_optimal_BER_roundtrip(trainSNR - dboff, config.bps)
    else
        trainSNR = get_optimal_SNR_for_BER(config.train_kwargs.target_ber, config.bps)
        dboffBER = get_optimal_BER(trainSNR - dboff, config.bps)
    end
    iters = sort(collect(keys(results)))
    dboff_iters = -1
    ber = 0
    for it in iters
        # Choose either worst (maximum) BER or mean BER across agent pairs
        if worst_ber
            bers = maximum(results[it][:ber], dims=3)[:, 5]
        else
            bers = mean(results[it][:ber], dims=3)[:, 5]
        end
        # For HT protocols, only one ber will be >0
        # For RT protocols, the last ber is what we want
        ber = bers[findlast(x -> x >= 0., bers)]
        if ber <= dboffBER
            dboff_iters = it
            break
        end
    end
    dboff_iters
end

"""
Find first 3dB-off BER result and return number of training iterations to reach it
"""
find_3db_off(config, results) = find_dboff(config, results, 3.)


end