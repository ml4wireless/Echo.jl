using Test
using Accessors
using Flux: params, Chain

push!(LOAD_PATH, "../src")
using Echo

include("schedule_tests.jl")


function run_converges(configfile; enable_self_play=false, enable_partner_modeling=false)
    testname = uppercase(basename(configfile)[1:end - 4])
    if enable_self_play
        testname *= "-SP"
    end
    if enable_partner_modeling
        testname *= "-PM"
    end
    @info testname
    cfg = loadconfig(configfile)
    if cfg.agent_types[1].mod == "neural"
        cfg = @set cfg.agent_types[1].self_play = enable_self_play
        cfg = @set cfg.agent_types[1].use_prtnr_model = enable_partner_modeling
    end
    if cfg.agent_types[2].mod == "neural"
        cfg = @set cfg.agent_types[2].self_play = enable_self_play
        cfg = @set cfg.agent_types[2].use_prtnr_model = enable_partner_modeling
    end
    ec = ExperimentConfig(cfg, "/tmp")
    results, _ = run_experiment(ec, save_results=false)
    bers = finalbers(results)
    # Worst-case BER
    train_snr_bers = maximum(bers, dims=3)[:, 5]
    @info "Final BERs for $configfile" train_snr_bers
    println()
    # Add extra cases for EPP with swapped bit interperpretations
    all(train_snr_bers .< 0.02) || (0 < train_snr_bers[3] < 0.02 && (train_snr_bers[1] > 0.98 || 0.48 < train_snr_bers[1] < 0.52))
end


function run_meets_timing(configfile, maxtime)
    testname = uppercase(basename(configfile)[1:end - 4])
    @info testname
    cfg = loadconfig(configfile)
    # Run once to compile
    cfg = @set cfg.train_kwargs.num_iterations_train = 3
    ec = ExperimentConfig(cfg, "/tmp")
    _ = run_experiment(ec, save_results=false)
    # Run again to time
    cfg = @set cfg.train_kwargs.num_iterations_train = 100
    ec = ExperimentConfig(cfg, "/tmp")
    t0 = time()
    _ = run_experiment(ec, save_results=false)
    t1 = time()
    elapsed = t1 - t0
    @info "Runtime for $configfile" elapsed maxtime
    println()
    elapsed <= maxtime
end


_isclose(A::AbstractArray, B::AbstractArray, tol=1e-8) = all(abs.((A .- B) ./ A) .< tol .&& abs.((A .- B) ./ B) .< tol)
_isclose(A::Chain, B::Chain) = all([_isclose(p1, p2) for (p1, p2) in zip(params(A), params(B))])

function run_gradient_check(configfile)
    testname = uppercase(basename(configfile)[1:end - 4])
    @info testname
    cfg = loadconfig(configfile)
    cfg = @set cfg.train_kwargs.num_iterations_train = 10
    ec = ExperimentConfig(cfg, "/tmp")
    results, agents = run_experiment(ec, save_results=false)
    oldagents, _ = reconstruct_epoch(results, 0)
    for (i, (agent, oldagent)) in enumerate(zip(agents, oldagents))
        if isneural(agent.mod)
            if _isclose(oldagent.mod.μ, agent.mod.μ)
                @warn "Mod $i μ unchanged"
                return false
            end
            if cfg.train_kwargs.protocol != GP && _isclose(oldagent.mod.log_std, agent.mod.log_std)
                @warn "Mod $i log_std unchanged"
                return false
            end
        end
        if isneural(agent.demod)
            oldagents, _ = reconstruct_epoch(results, 0)
            if _isclose(oldagent.demod.net, agent.demod.net)
                @warn "Demod $i net unchanged"
                return false
            end
        end
    end
    # All parameters updated, success
    true
end


function testconvergence(nnconf, ncconf, ncluconf)
    @testset "BER convergence" begin
        @testset "NN convergence" begin
            for c in nnconf
                @test run_converges(c)
            end
        end

        @testset "NC convergence" begin
            for c in ncconf
                @test run_converges(c)
            end
        end

        @testset "NClu convergence" begin
            for c in ncluconf
                @test run_converges(c)
            end
        end
    end
end


function testtiming(nnconf, ncconf, ncluconf)
    @testset "Run timing" begin
        @testset "NN runtime" begin
            # timings = repeat([2], length(nnconf))
            timings = [2, 1.75, 2.5, 1.25, 1.25, 1]
            for (t, c) in zip(timings, nnconf)
                @test run_meets_timing(c, t)
            end
        end

        @testset "NC runtime" begin
            # timings = repeat([2], length(ncconf))
            timings = [1, 1, 1.25, .75, 1, .75]
            for (t, c) in zip(timings, ncconf)
                @test run_meets_timing(c, t)
            end
        end

        @testset "NClu runtime" begin
            # timings = repeat([2], length(ncluconf))
            timings = [3, 1.5, 1.75, 3.25, 2, 1.5]
            for (t, c) in zip(timings, ncluconf)
                @test run_meets_timing(c, t)
            end
        end
    end
end


function testgradients(nnconf, ncconf, ncluconf)
    @testset "Gradient check" begin
        @testset "NN gradients" begin
            for c in nnconf
                @test run_gradient_check(c)
            end
        end

        @testset "NC gradients" begin
            for c in ncconf
                @test run_gradient_check(c)
            end
        end

        @testset "NClu gradients" begin
            for c in ncluconf
                @test run_gradient_check(c)
            end
        end
    end
end


function testmultiagent(maconf)
    @testset "Multiagent convergence" begin
        for c in maconf
            @test run_converges(c)
        end
    end
end


function testoptimizers(optconf)
    @testset "Optimizers" begin
        for c in optconf
            @test run_converges(c)
        end
    end
end


function testselfplay(nnconf, ncconf, ncluconf)
    @testset "Self-play" begin
        @testset "NN self-play" begin
            for c in nnconf
                @test run_converges(c, enable_self_play=true)
            end
        end

        @testset "NC self-play" begin
            for c in ncconf
                @test run_converges(c, enable_self_play=true)
            end
        end

        @testset "NClu self-play" begin
            for c in ncluconf
                @test run_converges(c, enable_self_play=true)
            end
        end
    end
end


function testpartnermodeling(nnconf, ncconf, ncluconf)
    @testset "Partner-modeling" begin
        @testset "NN partner-modeling" begin
            for c in nnconf
                @test run_converges(c, enable_partner_modeling=true)
            end
        end

        @testset "NC partner-modeling" begin
            for c in ncconf
                @test run_converges(c, enable_partner_modeling=true)
            end
        end

        @testset "NClu partner-modeling" begin
            for c in ncluconf
                @test run_converges(c, enable_partner_modeling=true)
            end
        end
    end
end


function main(args)
    helpinfo = """
julia $PROGRAM_FILE [-h] [-b] [-c] [-m] [-o] [-p] [-s] [-S] [-t]
    -h to print help
    -b to run backprop tests only
    -c to run convergence tests only
    -m to run multiagent tests only
    -o to run optimizer tests only
    -p to run partner-modeling tests only
    -s to run scheduler tests only
    -S to run self-play tests only
    -t to run timing tests only
"""

    # Uncomment to disable progressbar printouts
    # ENV["CI"] = "true"

    configs = readdir("configs/convergence/", join=true, sort=true)
    nnconf = filter(contains("nn_"), configs)
    ncconf = filter(contains("nc_"), configs)
    ncluconf = filter(contains("nclu_"), configs)
    maconf = readdir("configs/multiagent/", join=true, sort=true)
    optconf = readdir("configs/optim/", join=true, sort=true)


    if "-h" ∈ args || length(args) > 1
        println(helpinfo)
        return
    end

    testsets = Dict("-c" => "convergence", "-b" => "backprop", "-m" => "multiagent",
                    "-o" => "optimisers", "-t" => "timing", "-s" => "schedules",
                    "-S" => "self-play", "-p" => "partner-modeling")
    @testset "All tests" begin
        if length(args) > 0
            print("Running ")
            print(join([testsets[s] for s in args], ", "))
            println(" tests")
        end

        if "-t" ∈ args
            testtiming(nnconf, ncconf, ncluconf)
        elseif "-c" ∈ args
            testconvergence(nnconf, ncconf, ncluconf)
        elseif "-b" ∈ args
            testgradients(nnconf, ncconf, ncluconf)
        elseif "-m" ∈ args
            testmultiagent(maconf)
        elseif "-o" ∈ args
            testoptimizers(optconf)
        elseif "-s" ∈ args
            testschedules(optconf[1])
        elseif "-S" ∈ args
            testselfplay(nnconf, ncconf, ncluconf)
        elseif "-p" ∈ args
            testpartnermodeling(nnconf, ncconf, ncluconf)
        else
            testtiming(nnconf, ncconf, ncluconf)
            testconvergence(nnconf, ncconf, ncluconf)
            testgradients(nnconf, ncconf, ncluconf)
            testmultiagent(maconf)
            testoptimizers(optconf)
            testschedules(optconf[1])
            testselfplay(nnconf, ncconf, ncluconf)
            testpartnermodeling(nnconf, ncconf, ncluconf)
        end
    # End all tests
    end
end

if !isinteractive()
    main(ARGS)
end


