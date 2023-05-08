module ConfigUtils
export loadconfig, writeconfig, match_opponent, set_activations, set_keys, setbps, configsummary

using Accessors
using Crayons
using YAML
using SymDict
using YAML

using ..Protocols


"""
Convert Dict to NamedTuple for speed, .<> notation access
"""
dicts_to_nt(x) = x
dicts_to_nt(d::Dict) = (; (Symbol(k) => dicts_to_nt(v) for (k, v) in d)...)
dicts_to_nt(a::AbstractArray) = [dicts_to_nt(d) for d in a]

"""
Convert NamedTuple to Dict for writing to file
"""
nt_to_dict(x) = x
nt_to_dict(nt::NamedTuple) = Dict((k => nt_to_dict(v) for (k, v) in pairs(nt))...)

"""
Make kwargs for opponent neural mod match primary [meta-learned] neural mod
"""
function match_opponent(config)
    nmod = config.neural_mod_kwargs
    for k in keys(nmod)
        config = @set config.neural_mod_2_kwargs[k] = nmod[k]
    end
    config
end


"""
    set_keys(config::NamedTuple, setkey::Symbol, val)

Set all `setkey` leaf parameters to `val`.

Returns a copy of `config` with adjusted values.
"""
set_keys(config, _::Symbol, _) = config
function set_keys(config::NamedTuple, setkey::Symbol, val)
    ckeys = collect(keys(config))
    for k in ckeys
        if k == setkey
            config = @set config[k] = val
        else
            config = @set config[k] = set_keys(config[k], setkey, val)
        end
    end
    config
end
set_keys(config::NamedTuple, setkey::AbstractString, val) = set_keys(config, Symbol(setkey), val)


"""
    set_activations(config, fn)

Set all hidden layer activation functions to `fn`.
"""
set_activations(config, fn) = set_keys(config, :activation_fn_hidden, fn)


"""
Convert Dict{String, Any} to Dict{Symbol, Any}
"""
function to_symdict(d::Dict)
    d = symboldict(d)
    for k in keys(d)
        if isa(d[k], Dict)
            d[k] = to_symdict(d[k])
        end
    end
    d
end


"""
Force all floats to be Float32
"""
floats_to_32!(x) = x
floats_to_32!(f::F) where {F<:AbstractFloat} = Float32(f)
function floats_to_32!(d::Dict)
    for (k, v) in d
        d[k] = floats_to_32!(v)
    end
    d
end


"""
    get_nested_value(d::Dict, keys)

Get value specified by a list of `keys` from nested dictionary `d`.
"""
function get_nested_value(d::Dict, keys)
    cur_d = d
    for k in keys
        try
            cur_d = cur_d[k]
        catch e
            if isa(e, KeyError)
                @error "Failed to descend at key $k of $keys"
            else
                rethrow(e)
            end
        end
    end
    cur_d
end


"""
    replace_vars!(root; verbose = false)

Walk config dictionary, replace \$var with matching entry
"""
function replace_vars!(root, cur_node; path) end
function replace_vars!(root, cur_node::Dict; path="", verbose::Bool=false)
    for (k, v) in cur_node
        if isa(v, String) && v[1] == '$'
            topics = split(v[2:end], ".")
            cur_node[k] = get_nested_value(root, topics)
            if verbose
                println("replace_vars!: replacing $(path)$(k).$(v) with $(cur_node[k])")
            end
        end
        replace_vars!(root, cur_node[k], path=path * "$(k).")
    end
end
replace_vars!(root::Dict; verbose::Bool=false) = replace_vars!(root, root, verbose=verbose)


"""
    loadconfig(filename; verbose = false)

Load configuration file; expand \$var variables from top level, convert to NamedTuple.
"""
function loadconfig(filename; verbose::Bool=false)
    if splitext(filename)[2] ∈ [".yml", ".yaml"]
        config = YAML.load_file(filename)
    elseif splitext(filename)[2] == ".json"
        config = JSON3.load_file(filename)
    else
        println(stderr, "Error loading config with unknown extension $(splitext(filename)[2])")
    end
    # Walk config tree, replace $var with matching top-level entry
    replace_vars!(config, verbose=verbose)
    floats_to_32!(config)
    # Ensure valid protocol flag
    protocol = lowercase(config["train_kwargs"]["protocol"])
    if protocol ∈ ["gp", "gradient_passing"]
        config["train_kwargs"]["protocol"] = GP
    elseif protocol ∈ ["lp", "loss_passing"]
        config["train_kwargs"]["protocol"] = LP
    elseif protocol ∈ ["esp", "shared_preamble"]
        config["train_kwargs"]["protocol"] = ESP
    elseif protocol ∈ ["epp", "private_preamble"]
        config["train_kwargs"]["protocol"] = EPP
    else
        throw(ValueError("Unrecognized protocol $(protocol)"))
    end
    config = dicts_to_nt(config)
    config
end


"""
    writeconfig(config, filename)

Write configuration `config` to file `filename`.
"""
function writeconfig(config::Dict, filename)
    config["train_kwargs"]["protocol"] = string(config["train_kwargs"]["protocol"])
    YAML.write_file(filename, config)
end
writeconfig(config::NamedTuple, filename) = writeconfig(nt_to_dict(config), filename)


"""
    setbps(cfg, bps)

Set all `bits_per_symbol` parameters in `cfg` to `bps`.

Returns an adjusted copy of `cfg`.
"""
function setbps(cfg::NamedTuple, bps::Integer)
    cfg = set_keys(cfg, :bps, bps)
    cfg = set_keys(cfg, :bits_per_symbol, bps)
    cfg
end


function _config_info(cfg)
    title = "$(cfg.experiment_type) → $(cfg.experiment_id), $(cfg.train_kwargs.protocol)"
    basic_train = (
        "BPS=$(cfg.bps), OPT=$(cfg.train_kwargs.optimiser), " *
        "SCHED=$(cfg.train_kwargs.schedule.type)[$(cfg.train_kwargs.schedule.T_max)], " *
        "ITERS=$(cfg.train_kwargs.num_iterations_train)"
    )
    agents = join(["($(ag.mod) + $(ag.demod))" for ag in cfg.agent_types], ", ")
    agent_detail = (
        "HL_mod=$(cfg.neural_mod_kwargs.hidden_layers), HL_demod=$(cfg.neural_demod_kwargs.hidden_layers), " *
        "η_mod=$(cfg.neural_mod_kwargs.lr_dict.mu), η_demod=$(cfg.neural_demod_kwargs.lr), " *
        "logσ=$(cfg.neural_mod_kwargs.log_std_dict.initial)"
    )
    title, basic_train, agents, agent_detail
end

"""
    configsummary(cfg::NamedTuple)

Print formatted summary of config parameters.
"""
function configsummary(io::IO, cfg::NamedTuple)
    title, basic_train, agents, agent_detail = _config_info(cfg)
    print(io,
        Crayon(bold=true), title * "\n", Crayon(bold=false, foreground=:blue),
        basic_train * "\n", Crayon(foreground=:green), agents * "\n", Crayon(foreground=:light_gray),
        agent_detail * "\n", Crayon(reset=true)
    )
end

function configsummary(cfg::NamedTuple)
    configsummary(stdout, cfg)
end


end