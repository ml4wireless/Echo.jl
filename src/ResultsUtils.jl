module ResultsUtils
export loadresults, finalbers, finalagents, reconstruct_epoch, find_3db_off, find_dboff

using BSON
# Bring Flux and Zygote into module namespace for BSON loading
using Flux
using Zygote

using ..Agents
using ..Protocols
using ..LookupTableUtils



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
    run_results = results.results
    efinal = maximum(keys(run_results))
    run_results[efinal][:ber]
end

function finalbers(results::Dict{Int64, Any})
    efinal = maximum(keys(results))
    results[efinal][:ber]
end


"""
Return agents with state from end of training
"""
function finalagents(results::Dict{Int64, Any})
    efinal = maximum(keys(results))
    a1 = Agent(; results[efinal][:kwargs].tx...)
    a2 = Agent(; results[efinal][:kwargs].rx...)
    (a1, a2)
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
    a1 = MixedAgent(; results[epoch][:kwargs].tx...)
    a2 = MixedAgent(; results[epoch][:kwargs].rx...)
    (;agents=(a1, a2), bers=bers)
end


"""
Find first `dboff` dB-off BER result and return number of training iterations to reach it
"""
function find_dboff(config, results, dboff)
    roundtrip = config.train_kwargs.protocol ∈ [ESP, EPP]
    if roundtrip
        trainSNR = get_optimal_SNR_for_BER_roundtrip(config.train_kwargs.target_ber, config.bps)
        dboffBER = get_optimal_BER_roundtrip(trainSNR - 3, config.bps)
    else
        trainSNR = get_optimal_SNR_for_BER(config.train_kwargs.target_ber, config.bps)
        dboffBER = get_optimal_BER(trainSNR - 3, config.bps)
    end
    iters = sort(collect(keys(results)))
    dboff_iters = -1
    ber = 0
    for it in iters
        bers = results[it][:ber][:, 5]
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