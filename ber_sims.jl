# %%
using Echo
using BSON
using Accessors
using Flux
using LinearAlgebra
using Plots
using ProgressMeter
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

# %%
new_tbl = Dict{Int, Matrix{Float32}}()

# %%
# pb = plot(title="BER HT Curves", xlabel="SNR dB", ylabel="BER", yscale=:log)
# pbr = plot(title="BER RT Curves", xlabel="SNR dB", ylabel="BER", yscale=:log)
# ps = plot(title="SER HT Curves", xlabel="SNR dB", ylabel="SER", yscale=:log)
# psr = plot(title="SER RT Curves", xlabel="SNR dB", ylabel="SER", yscale=:log)
SNRs = Float32.(collect(LinRange(-10, 34.8, 225)))
colors = Dict(1 => :purple, 2 => :blue, 4 => :green, 6 => :orange)
for bps in [1,2,3,4,6,8,10]
    println("BPS $bps")
    # pb_th = [predict_classic_ber(bps, Float32(snr)) for snr in SNRs]
    # pb_th_rt = [predict_classic_ber_roundtrip(bps, Float32(snr)) for snr in SNRs]
    # ps_th = [predict_classic_ser(bps, Float32(snr)) for snr in SNRs]
    # ps_th_rt = [predict_classic_ser_roundtrip(bps, Float32(snr)) for snr in SNRs]
    # plot!(pb, SNRs, pb_th, label="Thy, BPS=$bps, BER", color=colors[bps], linewidth=2)
    # plot!(pbr, SNRs, pb_th_rt, label="Thy, BPS=$bps, BER", color=colors[bps], linewidth=2)
    # plot!(ps, SNRs, ps_th, label="Thy, BPS=$bps, SER", color=colors[bps], linewidth=2)
    # plot!(psr, SNRs, ps_th_rt, label="Thy, BPS=$bps, SER", color=colors[bps], linewidth=2)
    # if bps <= 6
    #     pb_tbl = [get_optimal_BER(snr, bps) for snr in SNRs]
    #     pb_tbl_rt = [get_optimal_BER_roundtrip(snr, bps) for snr in SNRs]
    #     ps_tbl = [get_optimal_SER(snr, bps) for snr in SNRs]
    #     ps_tbl_rt = [get_optimal_SER_roundtrip(snr, bps) for snr in SNRs]
        # plot!(pb, SNRs, pb_tbl, linestyle=:dash, label="Tbl, BPS=$bps, BER", color=colors[bps], linewidth=2)
        # plot!(pbr, SNRs, pb_tbl_rt, linestyle=:dash, label="Tbl, BPS=$bps, BER", color=colors[bps], linewidth=2)
        # plot!(ps, SNRs, ps_tbl, linestyle=:dash, label="Tbl, BPS=$bps, SER", color=colors[bps], linewidth=2)
        # plot!(psr, SNRs, ps_tbl_rt, linestyle=:dash, label="Tbl, BPS=$bps, SER", color=colors[bps], linewidth=2)
    # end
    pb_ex_ht = zeros(Float32, length(SNRs))
    pb_ex_rt = zeros(Float32, length(SNRs))
    ps_ex_ht = zeros(Float32, length(SNRs))
    ps_ex_rt = zeros(Float32, length(SNRs))
    @showprogress Threads.@threads for i in eachindex(SNRs)
        bh, br, sh, sr = sim_ber_ser_rt(bps, SNRs[i], 10_000_000)
        pb_ex_ht[i] = bh
        pb_ex_rt[i] = br
        ps_ex_ht[i] = sh
        ps_ex_rt[i] = sr
    end
    new_tbl[bps] = hcat(SNRs, pb_ex_ht, ps_ex_ht, pb_ex_rt, ps_ex_rt)
    # plot!(pb, SNRs, pb_ex_ht, linestyle=:dot, label="Exp, BPS=$bps, BER", color=colors[bps], linewidth=2)
    # plot!(pbr, SNRs, pb_ex_rt, linestyle=:dot, label="Exp, BPS=$bps, BER", color=colors[bps], linewidth=2)
    # plot!(ps, SNRs, ps_ex_ht, linestyle=:dot, label="Exp, BPS=$bps, SER", color=colors[bps], linewidth=2)
    # plot!(psr, SNRs, ps_ex_rt, linestyle=:dot, label="Exp, BPS=$bps, SER", color=colors[bps], linewidth=2)
    bson("new_ber_lookup_table.bson", new_tbl)
end

bson("new_ber_lookup_table.bson", new_tbl)

# plot!(pb, ylim=[1e-7, 1.1])
# plot!(pbr, ylim=[1e-7, 1.1])
# plot!(ps, ylim=[1e-7, 1.1])
# plot!(psr, ylim=[1e-7, 1.1])

# display(plot(pb, ps, layout=(1, 2), size=(1200, 500), legend=:outertopright))
# display(plot(pbr, psr, layout=(1, 2), size=(1200, 500), legend=:outertopright))

# %%
