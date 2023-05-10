module Schedules
export ConstantSchedule, LinearSchedule, CosineSchedule, RestartSchedule, CyclicSchedule, SequenceSchedule
export step!, getlr, setup

using Accessors
using Optimisers
using Functors: fmap, isleaf


# TODO: Build infrastructure for nested (cyclic/sequence) schedules after state tree has been built

# Returns a tuple tree matching the layout of `tree` with `val` as every leaf
treeify(tree, val; exclude=isleaf) = fmap(_ -> val, tree, exclude=exclude)

# Need to avoid recursing into Optimisers Leafs
isOptimLeafOrLeaf(l) = isa(l, Optimisers.Leaf) || isleaf(l)
isOptimLeafOnly(l) = isa(l, Optimisers.Leaf)


###################################################################
# Schedule Definitions
###################################################################

abstract type Schedule end


struct ConstantSchedule <: Schedule
    η::Float32
    T_max::Int
end

ConstantSchedule(;eta, T_max=100, kwargs...) = ConstantSchedule(eta, T_max)


struct LinearSchedule <: Schedule
    ηmin::Float32
    ηmax::Float32
    T_max::Int
end

function LinearSchedule(;etamin, etamax, T_max, kwargs...)
    @assert etamin < etamax "etamin must be less than etamax"
    LinearSchedule(etamin, etamax, T_max)
end

struct CosineSchedule <: Schedule
    ηmin::Float32
    ηmax::Float32
    T_max::Int
    restart::Bool
end

function CosineSchedule(;etamin, etamax, T_max, restart=true, kwargs...)
    @assert etamin < etamax "etamin must be less than etamax"
    CosineSchedule(etamin, etamax, T_max, restart)
end


struct RestartSchedule <: Schedule
    η::Float32
    T_max::Int
    opt_type::Type
end

RestartSchedule(;eta, T_max, opt_type, kwargs...) = RestartSchedule(eta, T_max, opt_type)


struct CyclicSchedule <: Schedule
    schedule::Schedule
end

function CyclicSchedule(;schedule, kwargs...)
    @assert !isa(schedule, CyclicSchedule) "Cyclic schedules cannot be nested"
    if isa(schedule, Type)
        # Construct the inner schedule when specified by type using kwargs
        return CyclicSchedule(schedule(;kwargs...))
    end
    CyclicSchedule(schedule)
end


struct SequenceSchedule <: Schedule
    schedules::Vector
    T_max::Int
end

function SequenceSchedule(schedules, kwargs...)
    _allowed(s::Schedule) = !isa(s, CyclicSchedule) && !isa(s, SequenceSchedule)
    @assert all(_allowed, schedules) "Cyclic and Sequence schedules are not allowed inside a SequenceSchedule"
    T_max = sum([s.T_max for s in schedules])
    SequenceSchedule(schedules, T_max)
end

"""
    schedule, new_step = current_schedule(s::SequenceSchedule, step)

Returns the current schedule and adjusted step for that schedule.
"""
function current_schedule(s::SequenceSchedule, step::Int)
    T_max = 0
    for sch in s.schedules[1:end-1]
        if step < T_max + sch.T_max
            return sch, step - T_max
        end
        T_max += sch.T_max
    end
    s.schedules[end], step - T_max
end


###################################################################
# LR calculations
###################################################################

"""
    η = getlr(s, step)

Return the learning rate `η` for schedule `s` at step `step`.

If `s` is a state tree, returns a matching tree with individual
`η`s.
"""
getlr(s::Schedule, _)::Float32 = s.η

function getlr(s::LinearSchedule, step::Int)::Float32
    if step > s.T_max
        η = s.ηmin
    else
        η = s.ηmin + (s.ηmax - s.ηmin) * (s.T_max - step) / s.T_max
    end
    η
end

function getlr(s::CosineSchedule, step::Int)::Float32
    step = s.restart ? mod(step - 1, s.T_max) : (step - 1)
    η = s.ηmin + 0.5 * (s.ηmax - s.ηmin) * (1 + cos(pi * step / s.T_max))
    η
end

function getlr(s::CyclicSchedule, step::Int)::Float32
    step = mod(step - 1, s.T_max)
    getlr(s, step)
end

function getlr(s::SequenceSchedule, step::Int)::Float32
    sch, step = current_schedule(s, step)
    getlr(sch, step)
end

# Recursive version of getlr, with intialization and empty tuple base case
function getlr(s::NamedTuple, step::Int)
    steptree = treeify(s, step)
    getlr(s, steptree)
end
getlr(s::NamedTuple, step::NamedTuple) = fmap(getlr, s, step)
getlr(::Tuple{}, _::Int) = ()

###################################################################
# Optimiser updates
###################################################################

"""
    opt_state′ = step!(opt_state, s::Schedule, model, step)

Update the optimiser state based on schedule `s` at `step`.

# Parameters
- Optimiser state (`opt_state`): state tree for optimiser
- Schedule (`s`): learning rate schedule
- Model (`model`): optimised model, potentially needed for restarts
- Step (`step`): current training step
"""
function step!(opt_state, s::Schedule, _, step::Int)
    Optimisers.adjust!(opt_state, eta=getlr(s, step))
    opt_state
end

function step!(opt_state, s::SequenceSchedule, model, step::Int)
    if !isa(current_schedule(s, step)[1], RestartSchedule)
        Optimisers.adjust!(opt_state, eta=getlr(s, step))
    else
        sched, step = current_schedule(s, step)
        opt_state = step!(opt_state, sched, model, step)
    end
    opt_state
end

function step!(opt_state, s::RestartSchedule, model, step::Int)
    if mod(step, s.T_max) == 0
        # Preserve non-default struct values besides η, which may have been modified previously
        rule = opt_state.rule
        rule = @set rule.eta = s.η
        opt_state = Optimisers.setup(rule, model)
    end
    opt_state
end


# Include empty tuple base case for `fmap`
step!(::Tuple{}, ::Tuple{}, _, ::Int64) = ()

# If given a state tree for schedule, use `fmap` to recurse into tree
# and match schedules to optimisers
step!(opt_state, s, model, step_tree) = fmap(step!, opt_state, s, model, step_tree, exclude=isOptimLeafOrLeaf)

# Need to turn `step` scalar into a tree for `fmap`
step!(opt_state, s, model, step::Int) = fmap(step!, opt_state, s, model, treeify(s, step), exclude=isOptimLeafOrLeaf)


###################################################################
# Schedule state construction
###################################################################

function _schedule_maker(schedule_type, args)
    function make_schedule(leaf)
        η = 0f0
        # Account for OptimiserChain leaves
        if leaf.rule isa Optimisers.OptimiserChain
            for opt in leaf.rule.opts
                if hasfield(typeof(opt), :eta)
                    η = opt.eta
                    break
                end
            end
        else
            η = leaf.rule.eta
        end
        # If ηmax not specified, set it to rule's η
        ηmax = get(args, :etamax, η)
        # If ηmin not specified, set it to 0
        ηmin = get(args, :etamin, zero(η))
        # If opt_type not specified, set it to rule type
        opt_type = get(args, :opt_type, typeof(leaf.rule))
        defaults = (eta=η, etamax=ηmax, etamin=ηmin, opt_type=opt_type)
        # `merge` overwrites values with the second tuple's entries
        # Make sure not to overwrite `args` because it will propagate to
        # other calls to this function!
        full_args = merge(defaults, args)
        schedule_type(;full_args...)
    end
    make_schedule
end

"""
    schedule_state = setup(schedule_type, opt_state; args...)

Create a tree container schedule state for each optimiser in opt_state
based on `args`.

Leave `η` and `ηmax` unspecified to use the Rule's current `eta` value
as `η` or `ηmax`.

# Parameters
- schedule type (`schedule_type`): The Type of schedule desired
- Optimiser state (`opt_state`): State tree for optimisers which schedules must match
- Schedule arguments (`args`): Keyword arguments passed to schedule constructor
"""
function setup(schedule_type, opt_state; args...)
    maker = _schedule_maker(schedule_type, args)
    fmap(maker, opt_state, exclude = isOptimLeafOnly)
end

end