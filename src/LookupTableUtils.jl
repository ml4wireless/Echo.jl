module LookupTableUtils
export get_optimal_SNR_for_BER, get_optimal_SNR_for_SER, get_optimal_SNR_for_BER_roundtrip, get_optimal_SNR_for_SER_roundtrip
export get_optimal_BER, get_optimal_SER, get_optimal_BER_roundtrip, get_optimal_SER_roundtrip, get_optimal_BERs, get_optimal_BERs_roundtrip
export get_test_SNR_dbs, get_test_SNR_dbs_roundtrip

using PyCall


function read_pickle_data(file_name)
    py"""
    import pickle
    def read_pickle(file_name):
        with open(file_name, "rb") as f:
            return pickle.load(f)
    """
    py"read_pickle"(file_name)
end
const ber_lookup_table = read_pickle_data("$(Base.@__DIR__)/ber_lookup_table.pkl")


function bisect_left(a, x, lo = 1, hi = nothing)
    if lo < 1
        throw(BoundsError(a, lo))
    end
    if hi === nothing
        hi = length(a) + 1  # It's not `length(a)`!
    end
    while lo < hi
        mid = (lo + hi) ÷ 2
        a[mid] < x ? lo = mid + 1 : hi = mid
    end
    return lo
end


"""
Find smallest item greater-than or equal to key.
Raise ValueError if no such item exists.
If multiple keys are equal, return the leftmost.
"""
function find_ge(a, key)
    i = bisect_left(a, key)
    if i == length(a)
        return nothing
    else
        return i
    end
end


"""
Return the smallest SNR in db so that ber in classics using mod type corresponding to bits_per_symbol is less than target_ber + err_tolerance
"""
function get_optimal_SNR_for_BER(target_ber, bits_per_symbol; err_tolerance=1e-9)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(log10(target_ber + err_tolerance), digits=2)
    ck = find_ge(-lut[bits_per_symbol][:, 2], -target_ber-err_tolerance)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 1]
        return Float32(val)
    else
        @error "A BER value below target $target_ber was not achieved for any SNR in classics"
    end
end


"""
Return the smallest SNR in db so that ber_roundtrip in classics using mod type corresponding to bits_per_symbol is less than target_ber + err_tolerance
"""
function get_optimal_SNR_for_BER_roundtrip(target_ber_roundtrip, bits_per_symbol; err_tolerance=1e-9)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(log10(target_ber_roundtrip + err_tolerance), digits=2)
    ck = find_ge(-lut[bits_per_symbol][:, 4], -target_ber_roundtrip - err_tolerance)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 1]
        return Float32(val)
    else
        @error "A roundtrip BER value below target $target_ber_roundtrip was not achieved for any SNR in classics"
    end
end


"""
Return the smallest SNR in db so that ser in classics using mod type corresponding to bits_per_symbol is less than target_ser + err_tolerance
"""
function get_optimal_SNR_for_SER(target_ser, bits_per_symbol; err_tolerance=1e-9)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(log10(target_ser + err_tolerance), digits=2)
    ck = find_ge(-lut[bits_per_symbol][:, 3], -target_ser - err_tolerance)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 1]
        return Float32(val)
    else
        @error "A SER value below target $target_ser was not achieved for any SNR in classics"
    end
end


"""
Return the smallest SNR in db so that ser_roundtrip in classics using mod type corresponding to bits_per_symbol is less than target_ser + err_tolerance
"""
function get_optimal_SNR_for_SER_roundtrip(target_ser_roundtrip, bits_per_symbol; err_tolerance=1e-9)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(log10(target_ser_roundtrip + err_tolerance), digits=2)
    ck = find_ge(-lut[bits_per_symbol][:, 5], -target_ser_roundtrip - err_tolerance)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 1]
        return Float32(val)
    else
        @error "A roundtrip SER value below target $target_ser_roundtrip was not achieved for any SNR in classics"
    end
end


"""
Return ber of classics at this SNR value (in dB)
"""
function get_optimal_BER(target_SNR, bits_per_symbol)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(target_SNR, digits=1)
    ck = find_ge(lut[bits_per_symbol][:, 1], target_SNR)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 2]
        return Float32(val)
    else
        return 0
    end
end


"""
Return ser of classics at this SNR value (in dB)
"""
function get_optimal_SER(target_SNR, bits_per_symbol)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(target_SNR, digits=1)
    ck = find_ge(lut[bits_per_symbol][:, 1], target_SNR)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 3]
        return Float32(val)
    else
        return 0
    end
end


"""
Return ber_roundtrip of classics at this SNR value (in dB)
"""
function get_optimal_BER_roundtrip(target_SNR, bits_per_symbol)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(target_SNR, digits=1)
    ck = find_ge(lut[bits_per_symbol][:, 1], target_SNR)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 4]
        return Float32(val)
    else
        return 0
    end
end


"""
Return ser_roundtrip of classics at this SNR value (in dB)
"""
function get_optimal_SER_roundtrip(target_SNR, bits_per_symbol)::Float32
    global ber_lookup_table
    lut = ber_lookup_table
    ckey = round(target_SNR, digits=1)
    ck = find_ge(lut[bits_per_symbol][:, 1], target_SNR)
    if ck !== nothing
        val = lut[bits_per_symbol][ck, 5]
        return Float32(val)
    else
        return 0
    end
end


"""
Return vector of test SNRs (in dB) for various half-trip BERs
"""
function get_test_SNR_dbs(bits_per_symbol)::Vector{Float32}
    bers = [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    test_SNRs = [get_optimal_SNR_for_BER(ber, bits_per_symbol) for ber in bers]
    test_SNRs
end


"""
Return vector of test SNRs (in dB) for various round-trip BERs
"""
function get_test_SNR_dbs_roundtrip(bits_per_symbol)::Vector{Float32}
    bers = [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    test_SNRs = [get_optimal_SNR_for_BER_roundtrip(ber, bits_per_symbol) for ber in bers]
    test_SNRs
end


"""
Return tuple of half-trip (snrs, bers) for SNRs between low and high
"""
function get_optimal_BERs(SNR_low, SNR_high, bits_per_symbol; nsamples=100)::Tuple{Vector{Float32}, Vector{Float32}}
    snrs = LinRange(SNR_low, SNR_high, nsamples)
    bers = [get_optimal_BER(snr, bits_per_symbol) for snr in snrs]
    snrs = [get_optimal_SNR_for_BER(ber, bits_per_symbol) for ber in bers]
    (snrs, bers)
end


"""
Return tuple of (snrs, bers) for SNRs between low and high
"""
function get_optimal_BERs_roundtrip(SNR_low, SNR_high, bits_per_symbol; nsamples=100)::Tuple{Vector{Float32}, Vector{Float32}}
    snrs = LinRange(SNR_low, SNR_high, nsamples)
    bers = [get_optimal_BER_roundtrip(snr, bits_per_symbol) for snr in snrs]
    bers = unique(bers)
    snrs = [get_optimal_SNR_for_BER_roundtrip(ber, bits_per_symbol) for ber in bers]
    (snrs, bers)
end


end