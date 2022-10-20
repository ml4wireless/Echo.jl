# Echo.jl
Julia library for Echo training and utilities

## Configuration Files
See examples under `test/configs`

## Running experiments
- Load a config file with `loadconfig`
- Create an `ExperimentConfig` with the results directory
- Run the test with `run_experiment(excfg)`
  - Results are saved to disk 
  - Results are also returned as `(results, final_agents)`
- Plot training animation with `animate_agents(results)`
