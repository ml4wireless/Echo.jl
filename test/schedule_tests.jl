# Tests for Schedules.jl behavior

function test_matching_lr(config)
    config = loadconfig(config)
    sched_args = (type="constant", T_max=1000)
    cfg = @set config.train_kwargs.schedule = sched_args
    excfg = ExperimentConfig(cfg)
    a1 = get_agent(excfg, 1)
    a2 = get_agent(excfg, 2)
    optims = get_optimisers([a1, a2], get_optimiser_type(cfg.train_kwargs.optimiser))
    opt = optims[1]
    @assert opt.mod.μ.layers[1].weight.rule.eta != opt.demod.net.layers[1].weight.rule.eta "μ and net must have different LRs to test schedule behavior"
    scheds = get_schedules(optims, get_schedule_type(cfg.train_kwargs.schedule.type), cfg.train_kwargs.schedule)
    sch = scheds[1]
    molr = opt.mod.μ.layers[1].weight.rule.eta
    mslr = sch.mod.μ.layers[1].weight.η
    dolr = opt.demod.net.layers[1].weight.rule.eta
    dslr = sch.demod.net.layers[1].weight.η
    if !((molr == mslr) && (dolr == dslr))
         @warn "Mismatch in LRs molr($molr) == mslr($mslr) or dolr($dolr) == dslr($dslr)"
         return false
    end
    true
end


function test_changing_lr(config)
    config = loadconfig(config)
    @assert config.train_kwargs.optimiser != "ldog" "LDoG has no learning rate parameter"
    @assert config.neural_mod_kwargs.lr_dict.mu != 1f0 "Need mu LR != 1.0 to test schedule changes"
    sched_args = (type="linear", T_max=100, etamax=1f0, etamin=0.5f0)
    cfg = @set config.train_kwargs.schedule = sched_args
    excfg = ExperimentConfig(cfg)
    a1 = get_agent(excfg, 1)
    a2 = get_agent(excfg, 2)
    optims = get_optimisers([a1, a2], get_optimiser_type(cfg.train_kwargs.optimiser))
    scheds = get_schedules(optims, get_schedule_type(cfg.train_kwargs.schedule.type), cfg.train_kwargs.schedule)
    opt = optims[1]
    sch = scheds[1]
    # Check initial value
    opt = step!(opt, sch, a1, 1)
    if opt.mod.μ.layers[1].weight.rule.eta != Float32(1 - .5 / 100)
        @warn "Optimiser eta ($(opt.mod.μ.layers[1].weight.rule.eta)) not set to 0.995 @ step 1"
        return false
    end
    # Check mid-way value
    opt = step!(opt, sch, a1, 50)
    if opt.mod.μ.layers[1].weight.rule.eta != 0.75f0
        @warn "Optimiser eta ($(opt.mod.μ.layers[1].weight.rule.eta)) not set to 0.75 @ step 50"
        return false
    end
    # Check final value
    opt = step!(opt, sch, a1, 100)
    if opt.mod.μ.layers[1].weight.rule.eta != 0.5f0
        @warn "Optimiser eta ($(opt.mod.μ.layers[1].weight.rule.eta)) not set to 0.5 @ step 100"
        return false
    end
    true
end


function testschedules(conf)
    @testset "Schedules" begin
        @test test_matching_lr(conf)
        @test test_changing_lr(conf)
    end
end