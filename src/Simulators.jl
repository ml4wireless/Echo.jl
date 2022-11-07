module Simulators
export GradientPassingSimulator, LossPassingSimulator, SharedPreambleSimulator, PrivatePreambleSimulator, Simulator, simulate
export simulate_round_trip, RTResults
export simulate_half_trip, HTResults
export compute_ber_halftrip, compute_ber_roundtrip, compute_ber_htrt

using Flux
using ChainRules: @ignore_derivatives
using Statistics

using ..DataUtils
using ..FluxUtils
using ..ModulationModels
using ..Agents


abstract type Simulator end


struct GradientPassingSimulator{A1 <: Agent, A2 <: Agent} <: Simulator
    agent1::A1
    agent2::A2
    bits_per_symbol::Int
    len_preamble::Int
end

struct LossPassingSimulator{A1 <: Agent, A2 <: Agent} <: Simulator
    agent1::A1
    agent2::A2
    bits_per_symbol::Int
    len_preamble::Int
end

struct SharedPreambleSimulator{A1 <: Agent, A2 <: Agent} <: Simulator
    agent1::A1
    agent2::A2
    bits_per_symbol::Int
    len_preamble::Int
end

struct PrivatePreambleSimulator{A1 <: Agent, A2 <: Agent} <: Simulator
    agent1::A1
    agent2::A2
    bits_per_symbol::Int
    len_preamble::Int
end


struct RTResults{F <: AbstractArray{Float32}, U <: AbstractArray{UInt16}}
    preamble1::U
    preamble2::U
    # First half-trip data for loss calculations
    m1_actions::F
    d2_logits::F
    d2_ht_symbs::U
    # Second half-trip data for loss calculations
    m2_actions::F
    d1_logits::F
    d1_ht_symbs::U
    d1_rt_logits::F
    d1_rt_symbs::U
    # Final half-trip for m2 (EPP: d1, d2) update
    d2_rt_logits::F
    d2_rt_symbs::U
end


struct HTResults{F <: AbstractArray{Float32}, U <: AbstractArray{UInt16}}
    preamble::U
    # First half-trip data for loss calculations
    m1_actions::F
    d2_logits::F
    d2_symbs::U
end


function simulate_half_trip(a1, a2, bits_per_symbol, len_preamble, SNR_db, explore::Bool=true; cuda::Bool=false)::HTResults
    preamble = @ignore_derivatives get_random_data_sb(len_preamble, bits_per_symbol, cuda=cuda)
    # Forward
    m1_actions = modulate(a1.mod, preamble, explore=explore)
    m1_actions_noisy = @ignore_derivatives add_cartesian_awgn(m1_actions, SNR_db, signal_power=1f0)
    d2_logits = demodulate(a2.demod, m1_actions_noisy, soft=true)
    d2_sb = logits_to_symbols_sb(d2_logits, bits_per_symbol)
    HTResults(preamble, m1_actions, d2_logits, d2_sb)
end


function simulate_round_trip(a1, a2, bits_per_symbol, len_preamble, SNR_db, explore::Bool=true;
                             final_halftrip::Bool=true, shared_preamble::Bool=true, cuda::Bool=false)::RTResults
    preamble1 = @ignore_derivatives get_random_data_sb(len_preamble, bits_per_symbol, cuda=cuda)
    preamble2 = @ignore_derivatives shared_preamble ? preamble1 : get_random_data_sb(len_preamble, bits_per_symbol, cuda=cuda)
    # Forward
    m1_actions = modulate(a1.mod, preamble1, explore=explore)
    m1_actions_noisy = @ignore_derivatives add_cartesian_awgn(m1_actions, SNR_db, signal_power=1f0)
    d2_logits = demodulate(a2.demod, m1_actions_noisy, soft=true)
    d2_sb = logits_to_symbols_sb(d2_logits, bits_per_symbol)
    # Backward, include new preamble and round-trip message
    # New preamble is same as previous for ESP
    m2_actions_p = modulate(a2.mod, preamble2, explore=explore)
    m2_actions_g = modulate(a2.mod, d2_sb, explore=false)
    m2_actions_p_noisy = @ignore_derivatives add_cartesian_awgn(m2_actions_p, SNR_db, signal_power=1f0)
    m2_actions_g_noisy = @ignore_derivatives add_cartesian_awgn(m2_actions_g, SNR_db, signal_power=1f0)
    d1_p_logits = demodulate(a1.demod, m2_actions_p_noisy, soft=true)
    d1_ht_sb = logits_to_symbols_sb(d1_p_logits, bits_per_symbol)
    d1_rt_logits = demodulate(a1.demod, m2_actions_g_noisy, soft=true)
    d1_rt_sb = logits_to_symbols_sb(d1_rt_logits, bits_per_symbol)
    # Final half-trip for updating a2 mod, (EPP) a2 demod
    if !isclassic(a2.mod) && final_halftrip
        m1_actions_rt = modulate(a1.mod, d1_ht_sb, explore=false)
        m1_actions_rt_noisy = @ignore_derivatives add_cartesian_awgn(m1_actions_rt, SNR_db, signal_power=1f0)
        d2_rt_logits = demodulate(a2.demod, m1_actions_rt_noisy, soft=true)
        d2_rt_sb = logits_to_symbols_sb(d2_rt_logits, bits_per_symbol)
    else
        d2_rt_logits = similar(d1_rt_logits, (0, 0))
        d2_rt_sb = similar(d1_rt_sb, (0, 0))
    end

    RTResults(preamble1, preamble2,
              m1_actions, d2_logits, d2_sb,
              m2_actions_p, d1_p_logits, d1_ht_sb, d1_rt_logits, d1_rt_sb,
              d2_rt_logits, d2_rt_sb)
end


function simulate(sim::GradientPassingSimulator, SNR_db::Float32; explore::Bool=false, swapagents::Bool=false, cuda::Bool=false) end
function simulate(sim::LossPassingSimulator, SNR_db::Float32; explore::Bool=false, swapagents::Bool=false, cuda::Bool=false)
    A = swapagents ? sim.agent2 : sim.agent1
    B = swapagents ? sim.agent1 : sim.agent2
    simulate_half_trip(A, B, sim.bits_per_symbol, sim.len_preamble, SNR_db, explore; cuda=cuda)
end

function simulate(sim::SharedPreambleSimulator, SNR_db::Float32; explore::Bool=false)
    simulate_round_trip(sim.agent1, sim.agent2, sim.bits_per_symbol, sim.len_preamble, SNR_db, explore;
                        final_halftrip=true, shared_preamble=true, cuda=cuda)
end

function simulate(sim::PrivatePreambleSimulator, SNR_db::Float32; explore::Bool=false)
    simulate_round_trip(sim.agent1, sim.agent2, sim.bits_per_symbol, sim.len_preamble, SNR_db, explore;
                        final_halftrip=true, shared_preamble=false, cuda=cuda)
end


function compute_ber_roundtrip(sim::Simulator, SNR_db::Float32)
    if isa(sim, GradientPassingSimulator) || isa(sim, LossPassingSimulator)
        return -1f0
    end
    res = simulate_round_trip(sim.agent1, sim.agent2, sim.bits_per_symbol, sim.len_preamble, SNR_db, false)
    ber = get_bit_error_rate_sb(res.preamble1, res.d1_rt_symbs)
    ber
end


function compute_ber_halftrip(sim::Simulator, SNR_db::Float32)::Tuple{Float32, Float32}
    res = simulate_half_trip(sim.agent1, sim.agent2, sim.bits_per_symbol, sim.len_preamble, SNR_db, false)
    ber1 = get_bit_error_rate_sb(res.preamble, res.d2_symbs)
    if isa(sim, GradientPassingSimulator) || isa(sim, LossPassingSimulator)
        ber2 = -1f0
    else
        res = simulate_half_trip(sim.agent2, sim.agent1, sim.bits_per_symbol, sim.len_preamble, SNR_db, false)
        ber2 = get_bit_error_rate_sb(res.preamble, res.d2_symbs)
    end
    (ber1, ber2)
end


function compute_ber_htrt(sim::Simulator, SNR_db::Float32)::Tuple{Float32, Float32, Float32}
    res = simulate(sim, SNR_db, explore=false)
    if isa(sim, GradientPassingSimulator) || isa(sim, LossPassingSimulator)
        ber_12 = get_bit_error_rate_sb(res.preamble, res.d2_symbs)
        return ber_12, -1f0, -1f0
    else
        ber_rt = get_bit_error_rate_sb(res.preamble1, res.d1_rt_symbs)
        ber_12 = get_bit_error_rate_sb(res.preamble1, res.d2_ht_symbs)
        ber_21 = get_bit_error_rate_sb(res.preamble2, res.d1_ht_symbs)
        return ber_12, ber_21, ber_rt
    end
end


end