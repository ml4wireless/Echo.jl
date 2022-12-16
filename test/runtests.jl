using Test
using Accessors

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


function main()
    helpinfo = "Run with -c to run convergence tests only; -t to run timing tests only; -h to print help"
    # Uncomment to disable progressbar printouts
    # ENV["CI"] = "true"
    configs = sort(readdir("configs/convergence/", join=true, sort=true))
    nnconf = filter(contains("nn_"), configs)
    ncconf = filter(contains("nc_"), configs)
    ncluconf = filter(contains("nclu_"), configs)

    if "-h" ∈ ARGS
        println(helpinfo)
        return
    end

    if "-t" ∉ ARGS
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

    if "-c" ∉ ARGS
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
end

main()
