module ConfigUtils
export loadconfig, writeconfig, match_opponent, set_activations, set_keys

using SymDict
using Accessors
using YAML
using JSON3

using ..Protocols


"""
Convert Dict to NamedTuple for speed, .<> notation access
"""
dicts_to_nt(x) = x
dicts_to_nt(d::Dict) = (; (Symbol(k) => dicts_to_nt(v) for (k, v) in d)...)

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
Set all setkey parameters to val
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


"""Set hidden layer activation functions"""
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
Walk config dictionary, replace \$var with matching root-level entry
"""
function replace_vars!(root, cur_node; path) end
function replace_vars!(root, cur_node::Dict; path="", verbose::Bool=false)
    for (k, v) in cur_node
        if isa(v, String) && v[1] == '$' && v[2:end] in keys(root)
            cur_node[k] = root[v[2:end]]
            if verbose
                println("replace_vars!: replacing $(path)$(k).$(v) with $(root[v[2:end]])")
            end
        end
        replace_vars!(root, cur_node[k], path=path * "$(k).")
    end
end
replace_vars!(root::Dict; verbose::Bool=false) = replace_vars!(root, root, verbose=verbose)


"""
Load configuration file; expand \$var variables from top level, convert to NamedTuple
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
    # Ensure valid protocol flag
    config
end


"""
Write configuration to file
"""
function writeconfig(config::Dict, filename)
    config["train_kwargs"]["protocol"] = string(config["train_kwargs"]["protocol"])
    YAML.write_file(filename, config)
end
writeconfig(config::NamedTuple, filename) = writeconfig(nt_to_dict(config), filename)


end