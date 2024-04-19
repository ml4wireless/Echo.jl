module ModulationUtils
export get_gray_code, gray_coded_IQ, get_mqam_symbol, get_symbol_map, generate_symbol_map
export calc_EbN0, calc_N0, predict_classic_ber, predict_classic_ser, predict_classic_ber_roundtrip, predict_classic_ser_roundtrip
export demodulate_soft, demodulate_hard

using ..DataUtils
using Distributions: Normal, ccdf
using SpecialFunctions: erfc


"""
Left zero-pads input a to desired length n
"""
function zero_pad(a::Vector, n::Integer)
    npad = n - length(a)
    vcat(zeros(eltype(a), npad), a)
end


"""
Returns inverse gray code map for numbers upto 2**n-1

Inputs:
n: Number of bits. Find inverse gray code maps for [0,2**n-1]
Outputs:
g_inverse: Int vector of size [2**n]. For i in [0,2**n-1],
           g_inverse[i] contains position where i goes to when gray coded
"""
function get_gray_code(n)
    if n < 1
        g = []
    else
        g = ["0"; "1"]
        n -= 1
        while n > 0
            k = length(g)
            for i in k:-1:1
                char = "1" * g[i]
                push!(g, char)
            end
            for i in k:-1:1
                g[i] = "0" * g[i]
            end
            n -= 1
        end
    end
    g_inverse = zeros(Int16, length(g))
    for (i, gi) in enumerate(g)
        g_inverse[parse(Int16, gi, base=2) + 1] = i - 1
    end
    g_inverse
end


"""
Splits bit representation of k into two halves and returns
gray coded I and Q to be used to perform the QAM modulation
Inputs:
k: Integer
d: Integer, Length of bit representation
gray_code: Inverse gray code map for n = d/2
Outputs:
kI_gray: Integer in 0,2**(d/2) - 1. Gray coded integer for I
kQ_gray: Integer in 0,2**(d/2) - 1. Gray coded integer for Q
"""
function gray_coded_IQ(k, d, gray_code)
    IQ = integers_to_bits([k], d)
    I = IQ[1:div(d, 2)]
    Q = IQ[div(d, 2) + 1:end]
    kI = bits_to_integers(zero_pad(I, d), d)[1]
    kQ = bits_to_integers(zero_pad(Q, d), d)[1]
    I_gray = gray_code[kI+1]
    Q_gray = gray_code[kQ+1]
    I_gray, Q_gray
end


"""
Returns complex number corresponding to k (between 0 and M-1) for MQAM
Inputs:
k: Integer in [0,M-1]
M: Integer (Must be perfect square) (16 for QAM16, 64 for QAM64)
gray_code: Inverse gray code map for n=log2(sqrt(M))

Output:
mpsk_symbol: Complex number representing I + jQ for the symbol k in MPSK scheme
"""
function get_mqam_symbol(k, M, gray_code)
    @assert (k < M) "k=$k must be less than M=$M"
    K = sqrt(M)
    d = Int(log2(M))
    kI_gray, kQ_gray = gray_coded_IQ(k, d, gray_code)
    # scaling factor so overall constellation has unit average energy
    scaling_factor = 1 / (sqrt((2 / 3) * (M - 1)))
    mqam_symbol_I = scaling_factor * ((-K + 1) + (2 * kI_gray))
    mqam_symbol_Q = scaling_factor * ((-K + 1) + (2 * kQ_gray))
    mqam_symbol = mqam_symbol_I + 1im * mqam_symbol_Q
    ComplexF32.(mqam_symbol)
end


# Table of symbol maps for quick recall
# Stored as cartesian coordinates
symbol_maps = Dict{Int, Matrix{Float32}}()
symbol_maps[1] = permutedims(
    [[1.0 0.0]
     [-1.0 0.0]],
    (2, 1))

symbol_maps[2] = permutedims(
    [[0.7071067811865476 0.7071067811865475]
     [-0.7071067811865475 0.7071067811865476]
     [0.7071067811865474 -0.7071067811865477]
     [-0.7071067811865477 -0.7071067811865475]],
    (2, 1))

symbol_maps[3] = permutedims(
    [[0.9238795325112867 0.3826834323650898]
     [0.38268343236508984 0.9238795325112867]
     [-0.9238795325112867 0.3826834323650899]
     [-0.3826834323650897 0.9238795325112867]
     [0.9238795325112865 -0.3826834323650904]
     [0.38268343236509 -0.9238795325112866]
     [-0.9238795325112868 -0.38268343236508967]
     [-0.38268343236509034 -0.9238795325112865]],
    (2, 1))

symbol_maps[4] = permutedims(
    [[-0.9486832980505138 -0.9486832980505138];  [-0.9486832980505138 -0.31622776601683794]
     [-0.9486832980505138 0.9486832980505138];   [-0.9486832980505138 0.31622776601683794]
     [-0.31622776601683794 -0.9486832980505138]; [-0.31622776601683794 -0.31622776601683794]
     [-0.31622776601683794 0.9486832980505138];  [-0.31622776601683794 0.31622776601683794]
     [0.9486832980505138 -0.9486832980505138];   [0.9486832980505138 -0.31622776601683794]
     [0.9486832980505138 0.9486832980505138];    [0.9486832980505138 0.31622776601683794]
     [0.31622776601683794 -0.9486832980505138];  [0.31622776601683794 -0.31622776601683794]
     [0.31622776601683794 0.9486832980505138];   [0.31622776601683794 0.31622776601683794]],
    (2, 1))

symbol_maps[6] = permutedims(
    [[-1.0801234497346435 -1.0801234497346435];  [-1.0801234497346435 -0.7715167498104596]
     [-1.0801234497346435 -0.1543033499620919];  [-1.0801234497346435 -0.4629100498862757]
     [-1.0801234497346435 1.0801234497346435];   [-1.0801234497346435 0.7715167498104596]
     [-1.0801234497346435 0.1543033499620919];   [-1.0801234497346435 0.4629100498862757]
     [-0.7715167498104596 -1.0801234497346435];  [-0.7715167498104596 -0.7715167498104596]
     [-0.7715167498104596 -0.1543033499620919];  [-0.7715167498104596 -0.4629100498862757]
     [-0.7715167498104596 1.0801234497346435];   [-0.7715167498104596 0.7715167498104596]
     [-0.7715167498104596 0.1543033499620919];   [-0.7715167498104596 0.4629100498862757]
     [-0.1543033499620919 -1.0801234497346435];  [-0.1543033499620919 -0.7715167498104596]
     [-0.1543033499620919 -0.1543033499620919];  [-0.1543033499620919 -0.4629100498862757]
     [-0.1543033499620919 1.0801234497346435];   [-0.1543033499620919 0.7715167498104596]
     [-0.1543033499620919 0.1543033499620919];   [-0.1543033499620919 0.4629100498862757]
     [-0.4629100498862757 -1.0801234497346435];  [-0.4629100498862757 -0.7715167498104596]
     [-0.4629100498862757 -0.1543033499620919];  [-0.4629100498862757 -0.4629100498862757]
     [-0.4629100498862757 1.0801234497346435];   [-0.4629100498862757 0.7715167498104596]
     [-0.4629100498862757 0.1543033499620919];   [-0.4629100498862757 0.4629100498862757]
     [1.0801234497346435 -1.0801234497346435];   [1.0801234497346435 -0.7715167498104596]
     [1.0801234497346435 -0.1543033499620919];   [1.0801234497346435 -0.4629100498862757]
     [1.0801234497346435 1.0801234497346435];    [1.0801234497346435 0.7715167498104596]
     [1.0801234497346435 0.1543033499620919];    [1.0801234497346435 0.4629100498862757]
     [0.7715167498104596 -1.0801234497346435];   [0.7715167498104596 -0.7715167498104596]
     [0.7715167498104596 -0.1543033499620919];   [0.7715167498104596 -0.4629100498862757]
     [0.7715167498104596 1.0801234497346435];    [0.7715167498104596 0.7715167498104596]
     [0.7715167498104596 0.1543033499620919];    [0.7715167498104596 0.4629100498862757]
     [0.1543033499620919 -1.0801234497346435];   [0.1543033499620919 -0.7715167498104596]
     [0.1543033499620919 -0.1543033499620919];   [0.1543033499620919 -0.4629100498862757]
     [0.1543033499620919 1.0801234497346435];    [0.1543033499620919 0.7715167498104596]
     [0.1543033499620919 0.1543033499620919];    [0.1543033499620919 0.4629100498862757]
     [0.4629100498862757 -1.0801234497346435];   [0.4629100498862757 -0.7715167498104596]
     [0.4629100498862757 -0.1543033499620919];   [0.4629100498862757 -0.4629100498862757]
     [0.4629100498862757 1.0801234497346435];    [0.4629100498862757 0.7715167498104596]
     [0.4629100498862757 0.1543033499620919];    [0.4629100498862757 0.4629100498862757]],
    (2, 1))

function generate_symbol_map(bits_per_symbol)::Matrix{Float32}
    gcode = get_gray_code(bits_per_symbol)
    symbs_complex = [get_mqam_symbol(i, 2 ^ bits_per_symbol, gcode) for i in 0:(2 ^ bits_per_symbol - 1)]
    symbs_cartesian = permutedims(hcat(real(symbs_complex), imag(symbs_complex)), (2, 1))
    symbs_cartesian
end

function get_symbol_map(bits_per_symbol)::Matrix{Float32}
    if bits_per_symbol ∉ keys(symbol_maps)
        symbol_maps[bits_per_symbol] = generate_symbol_map(bits_per_symbol)
    end
    symbol_maps[bits_per_symbol]
end


"""
Calculates EbN0 for given modulator and N0 values.
Inputs:
modulator: Modulator object whose constellation is used.
N0: Float np.array or constant.

Outputs:
EbN0: EbN0 in decibels.
"""
function calc_EbN0(modulator, N0)::Float32
    symbols_i = 0:((2 ^ modulator.bits_per_symbol) - 1)
    constellation = modulator(symbols_i)
    Es = sum(abs2(constellation)) / length(constellation)
    EbN0_lin = Es / (modulator.bits_per_symbol * N0)
    EbN0 = 10. * log10(EbN0_lin)
    EbN0
end


"""
Calculates N0 for given modulator and EbN0 values
Inputs:
modulator: Modulator object whose constellation is used.
EbN0: Float np.array or constant.

Outputs:
N0: N0 values
"""
function calc_N0(modulator, EbN0)::Float32
    symbols_i = 0:((2 ^ modulator.bits_per_symbol) - 1)
    constellation = modulator(symbols_i)
    Es = sum(abs2(constellation)) / length(constellation)
    EbN0_lin = 10. ^ (EbN0 / 10)
    N0 = Es / (EbN0_lin * modulator.bits_per_symbol)
    N0
end


"""
Soft demodulation with log-probs for each possible symbol.
May be useful for decoding.
"""
function demodulate_soft(demod, iq)
    logits = demod(iq)
    logprobs = Flux.logsoftmax(logits, dims=1)
    logprobs
end


"""
Hard demodulation with ML choices.
Returns [bps x n] symbols array
"""
function demodulate_hard(demod, iq)
    logits = demod(iq)
    labels_si_g = getindex.(argmax(logits, dims=1), 1)
    labels_sb_g = integers_to_symbols(labels_si_g, demod.bits_per_symbol)
    labels_sb_g
end


"""
    `ser = predict_classic_ser(bits_per_symbol, SNR_db)`

Predicts the theoretical symbol error rate for a classic square constellation at `SNR_db`
From https://dsplog.com/2012/01/01/symbol-error-rate-16qam-64qam-256qam
"""
function predict_classic_ser(bits_per_symbol, SNR_db)::Float32
    @assert (iseven(bits_per_symbol) || bits_per_symbol == 1) "Can only calculate SER for square constellations (bps=1,2,4,...)"
    N0 = get_N0(SNR_db, 1)
    if bits_per_symbol == 1
        p_ser = ccdf(Normal(0, sqrt(N0 / 2)), 1)
        return p_ser
    end

    M = 2 ^ bits_per_symbol
    k = 1 / sqrt(2 * (M - 1) / 3)
    p_ser = 2 * (1 - 1 / sqrt(M)) * erfc(k / sqrt(N0)) - (1 - 2 / sqrt(M) + 1 / M) * erfc(k / sqrt(N0)) ^ 2
    p_ser
end


"""
    `ber = predict_classic_ber(bits_per_symbol, SNR_db)`

Predicts the theoretical bit error rate for a gray coded classic constellation at `SNR_db`
From Yoon, Cho, Lee, "Bit Error Probability of M-ary Quadrature Amplitude Modulation". IEEE VTS Fall VTC2000
"""
function predict_classic_ber(bits_per_symbol, SNR_db)::Float32
    @assert (iseven(bits_per_symbol) || bits_per_symbol == 1) "Can only calculate BER for square constellations (bps=1,2,4,...)"
    N0 = get_N0(SNR_db, 1)
    std = sqrt(N0 / 2)
    if bits_per_symbol == 1
        pb = ccdf(Normal(0, std), 1)
        return pb
    end

    M = 2 ^ bits_per_symbol
    sqrtM = sqrt(M)
    Eb = 1 / bits_per_symbol
    r = Eb / N0

    function _pb_k(k)
        if k == 1
            pb1 = sum(erfc((2*j + 1) * sqrt((3 * log2(M) * r) / (2 * (M - 1)))) for j in 0:(sqrtM / 2 - 1))
            return pb1 / sqrtM
        end
        pbk = sum((-1) ^ floor(j * 2 ^ (k-1) / sqrtM) * (2 ^ (k-1) - floor(j * 2 ^ (k-1) / sqrtM + 0.5)) *
                  erfc((2 * j + 1) * sqrt((3 * log2(M) * r) / (2 * (M - 1))))
                  for j in 0:((1 - 2 ^ -k) * sqrtM - 1))
        pbk / sqrtM
    end

    pb = 1 / log2(sqrtM) * sum(_pb_k(k) for k in 1:log2(sqrtM))
    pb
end


"""
    `ber = predict_classic_ber_roundtrip(bits_per_symbol, SNR_db)`

Predicts the theoretical bit error rate for a gray coded classic constellation at `SNR_db` for a roundtrip
Assuming only single-bit errors are likely (~5% BER or less):
P(error) = P(error, 1st halftrip) + P(error, 2nd halftrip) - P(error, 1st halftrip & 2nd halftrip)
"""
function predict_classic_ber_roundtrip(bits_per_symbol, SNR_db)::Float32
    p_ht = predict_classic_ber(bits_per_symbol, SNR_db)
    return 2 * p_ht - p_ht ^ 2
end


"""
    `ber = predict_classic_ser_roundtrip(bits_per_symbol, SNR_db)`

Predicts the theoretical symbol error rate for a classic square constellation at `SNR_db` for a roundtrip
P(error) = P(error, 1st halftrip) + P(error, 2nd halftrip) - 1/4 P(error, 1st halftrip & 2nd halftrip)
If an error occurs during ht1, w.p. ~1/4 it will be reversed during ht2, assuming SNR is high enough that
double-bit errors are unlikely (~5% SER or less).
"""
function predict_classic_ser_roundtrip(bits_per_symbol, SNR_db)::Float32
    p_ht = predict_classic_ser(bits_per_symbol, SNR_db)
    return 2 * p_ht - 1/4 * p_ht ^ 2
end



end