using Flux
using Random

using Echo


config_file = "./configs/convergence/nn_esp.yml"
config = loadconfig(config_file)
excfg = ExperimentConfig(config, "./results")

m1 = get_mod(excfg)
m2 = get_opp_mod(excfg)
d1 = get_demod(excfg)
d2 = get_opp_demod(excfg)

preamble = get_random_data_sb(10, 2)
reward = randn(Float32, 10)

params = Flux.params(m1)
local actions

loss, grads = Flux.withgradient(params) do
    actions = modulate(m1, preamble, explore=true)
    loss = loss_vanilla_pg(mod=m1, reward=reward, actions=actions)
end

println("loss $(loss)")
for p in params
    println("grad $(grads[p])")
end
@assert all([any(g .!= 0) for g in grads])
