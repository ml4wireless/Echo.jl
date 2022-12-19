using Test
using Accessors
using Flux: params, Chain

push!(LOAD_PATH, "../src")
using Echo


function run_converges(configfile)
    testname = uppercase(basename(configfile)[1:end - 4])
    @info testname
    cfg = loadconfig(configfile)
    ec = ExperimentConfig(cfg, "/tmp")
    results, _ = run_experiment(ec, save_results=false)
    bers = finalbers(results)
    train_snr_bers = bers[:, 5]
    @info "Final BERs for $configfile" train_snr_bers
    println()
    # Add extra cases for EPP with swapped bit interperpretations
    all(train_snr_bers .< 0.02) || (train_snr_bers[3] < 0.02 && (train_snr_bers[1] > 0.98 || 0.48 < train_snr_bers[1] < 0.52))
end


function run_meets_timing(configfile, maxtime)
    testname = uppercase(basename(configfile)[1:end - 4])
    @info testname
    cfg = loadconfig(configfile)
    cfg = @set cfg.train_kwargs.num_iterations_train = 100
    ec = ExperimentConfig(cfg, "/tmp")
    # Run once to compile
    _ = run_experiment(ec, save_results=false)
    # Run again to time
    t0 = time()
    _ = run_experiment(ec, save_results=false)
    t1 = time()
    elapsed = t1 - t0
    @info "Runtime for $configfile" elapsed
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


function main(args)
    helpinfo = """
julia $PROGRAM_FILE [-t] [-c] [-b] [-h]
    -c to run convergence tests only
    -t to run timing tests only
    -b to run backprop tests only
    -h to print help
"""
    # Uncomment to disable progressbar printouts
    # ENV["CI"] = "true"
    configs = sort(readdir("configs/convergence/", join=true, sort=true))
    nnconf = filter(contains("nn_"), configs)
    ncconf = filter(contains("nc_"), configs)
    ncluconf = filter(contains("nclu_"), configs)

    if "-h" ∈ args
        println(helpinfo)
        return
    end

    @testset "All tests" begin
        if "-t" ∉ args && "-b" ∉ args
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
            # End BER convergence
        end

        if "-c" ∉ args && "-b" ∉ args
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
                    timings = [3, 1.5, 3, 1.75, 2, 1.5]
                    for (t, c) in zip(timings, ncluconf)
                        @test run_meets_timing(c, t)
                    end
                end

            end
            # End run timing
        end

        if "-t" ∉ args && "-c" ∉ args
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
            # End gradient check
        end
    # End all tests
    end
end

main(ARGS)
