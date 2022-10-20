using Test
using Accessors

push!(LOAD_PATH, "../src")
using Echo 


function run_converges(configfile)
    cfg = loadconfig(configfile)
    ec = ExperimentConfig(cfg, "/tmp")
    results, _ = run_experiment(ec, save_results=false)
    bers = finalbers(results)
    train_snr_bers = bers[:, 5]
    @info "Final BERs for $configfile" train_snr_bers
    # Add extra case for EPP with swapped bit interperpretations
    all(train_snr_bers .< 0.02) || (train_snr_bers[3] < 0.02 && train_snr_bers[1] > 0.98)
end


function run_meets_timing(configfile, maxtime)
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
    elapsed <= maxtime
end


function main()
    helpinfo = "Run with -c to run convergence tests only; -t to run timing tests only; -h to print help"
    configs = readdir("configs/convergence/", join=true, sort=true)

    if "-h" ∈ ARGS
        println(helpinfo)
        return
    end

    if "-t" ∉ ARGS
        @testset "BER convergence" begin
            for c in configs
                @test run_converges(c)
            end
        end
    end

    if "-c" ∉ ARGS
        @testset "Run timing" begin
            timings = [2 2 2 2 2 2]
            for (t, c) in zip(timings, configs)
                @test run_meets_timing(c, t)
            end
        end
    end
end

main()
