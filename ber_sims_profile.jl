# %%
using Echo
using BSON
using Accessors
using Flux
using LinearAlgebra
using Plots
using ProgressMeter
using Profile
plotlyjs()


# %%

# %%
function sim_ber_ser_ht(bps, snr, N=1_000_000)
    m = ClassicMod(bits_per_symbol=bps, rotation_deg=0,)
    d = ClassicDemod(bits_per_symbol=bps, rotation_deg=0,)
    a1 = Agent(m, nothing)
    a2 = Agent(nothing, d)

    n = min(100_000, N)
    reps = ceil(N / n)
    ber = 0.
    ser = 0.
    for _ in 1:reps
        res = simulate_half_trip(a1, a2, bps, n, snr, false)
        ber += sum(res.preamble .!= res.d2_symbs) / n / bps
        ser += sum(any.(eachcol(res.preamble .!= res.d2_symbs))) / n
    end
    ber / reps, ser / reps
end

function sim_ber_ser_rt(bps, snr, N=1_000_000)
    m1 = ClassicMod(bits_per_symbol=bps, rotation_deg=0,)
    m2 = ClassicMod(bits_per_symbol=bps, rotation_deg=0,)
    d1 = ClassicDemod(bits_per_symbol=bps, rotation_deg=0,)
    d2 = ClassicDemod(bits_per_symbol=bps, rotation_deg=0,)
    a1 = Agent(m1, d1)
    a2 = Agent(m2, d2)

    n = min(100_000, N)
    if bps > 6
        n = min(10_000, n)
    end
    reps = ceil(N / n)
    ber_ht = ber_rt = 0.
    ser_ht = ser_rt = 0.
    for _ in 1:reps
        res = simulate_round_trip(a1, a2, bps, n, snr, false, final_halftrip=false, shared_preamble=true)
        ber_ht += sum(res.preamble1 .!= res.d2_ht_symbs) / n / bps
        ser_ht += sum(any.(eachcol(res.preamble1 .!= res.d2_ht_symbs))) / n
        ber_rt += sum(res.preamble1 .!= res.d1_rt_symbs) / n / bps
        ser_rt += sum(any.(eachcol(res.preamble1 .!= res.d1_rt_symbs))) / n
    end
    return ber_ht/reps, ber_rt/reps, ser_ht/reps, ser_rt/reps
end

# sim_ber_ser_ht(2, 8.4f0, 1_000_000)
sim_ber_ser_rt(2, 8.4f0, 1_000_000)


SNRs = Float32.([10])
for bps in [10]
    println("BPS $bps")
    for i in eachindex(SNRs)
        @profile sim_ber_ser_rt(bps, SNRs[i], 10_000_000)

    end
end
