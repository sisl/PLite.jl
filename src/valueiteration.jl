export
  SerialValueIteration,
  ParallelValueIteration,
  discretize_statevariable!

const MaxIter = 1000
const Tol = 1e-4

abstract type ValueIteration <: Solver

type SerialValueIteration <: ValueIteration

  maxiter::Int64
  tol::Float64
  verbose::Bool
  discretizemap::Dict{String, LazyDiscrete}

  SerialValueIteration(
      maxiter::Int64=MaxIter,
      tol::Float64=Tol,
      verbose::Bool=false) =
    new(maxiter, tol, verbose, Dict{String, LazyDiscrete}())

end

type ParallelValueIteration <: ValueIteration

  maxiter::Int64
  tol::Float64
  verbose::Bool
  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}

  ParallelValueIteration(
      maxiter::Int64=MaxIter,
      tol::Float64=Tol,
      verbose::Bool=false) =
    new(
      maxiter,
      tol,
      verbose,
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}())

end

type LazyDiscrete

  varname::String
  step::Float64

  LazyDiscrete(varname::String, step::Float64) = new(varname, step)

end

type ValueIterationSolution <: Solution

  qval::Matrix{Float64}
  cputime::Float64
  niter::Int64
  finaltol::Float64

  ValueIterationSolution(
      qval::Matrix{Float64},
      cputime::Float64,
      niter::Int64,
      finaltol::Float64) =
    new(qval, cputime, niter, finaltol)

end

function discretize_statevariable!(vi::ValueIteration, varname::String, step::Float64)
  if haskey(vi.statemap, varname)
    warn(string(
      "state variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.statemap[varname] = LazyDiscrete(varname, step)
end

function discretize_actionvariable!(vi::ValueIteration, varname::String, step::Float64)
  if haskey(vi.actionmap, varname)
    warn(string(
      "action variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.actionmap[varname] = LazyDiscrete(varname, step)
end

function lazyCheck(mdp::MDP, vi::ValueIteration)
  checkDiscretize(mdp, vi)
  checkTransition(mdp, vi)
end

# Check that all |mdp| RangeVar variables have valid discretization schemes in |vi|
function checkDiscretize(mdp::MDP, vi::ValueIteration)

  for lazyvar in values(mdp.statemap)
    if isa(lazyvar, RangeVar)
      if !haskey(vi.statemap, lazyvar.varname)
        error(string(
          "state variable ", lazyvar.varname,
          " does not have a discretization scheme"))
      elseif lazyvar.maxval - lazyvar.minval < vi.statemap[lazyvar.varname].step
        error(string(
          "state variable ", lazyvar.varname,
          " has a discretization step size larger than than its range"))
      end
    end
  end

  for lazyvar in values(mdp.actionmap)
    if isa(lazyvar, RangeVar)
      if !haskey(vi.actionmap, lazyvar.varname)
        error(string(
          "action variable ", lazyvar.varname,
          " does not have a discretization scheme"))
      elseif lazyvar.maxval - lazyvar.minval < vi.actionmap[lazyvar.varname].step
        error(string(
          "action variable ", lazyvar.varname,
          " has a discretization step size larger than than its range"))
      end
    end
  end

end

# Check transition function with a few random examples for correct input types, return types and corresponding probability or state variable value boundedness
function checkTransition(mdp::MDP, vi::ValueIteration)
  args = randargs(mdp)
  transitionval = mdp.transition.fn(args...)
  if isa(transitionval, Float64)  # T(s,a,s')
    if transitionval < 0.0 || transitionval > 1.0
      warn(string(  # warn not error because we might have sampled a non-existent state
        "transition function provided is of type T(s,a,s'), ",
        "but the value returned from a random state is not ",
        "a valid probability value bounded to [0,1]\n",
        "argument names: ", mdp.transition.argnames, "\n",
        "random state: ", args, "\n",
        "return value: ", transitionval))
    end
  else isa(transitionval, Vector)  # T(s,a)
    nargs = length(mdp.transition.argnames)
    for returnval in transitionval
      if length(returnval) != nargs || !isvalid(mdp, returnval)
        error(string(
          "transition function provided is of type T(s,a), ",
          "but one of the values returned from a random state is not ",
          "a valid state either bounded by its range or set of values\n",
          "argument names: ", mdp.transition.argnames, "\n",
          "random state: ", args, "\n",
          "return value: ", returnval))
      end
    end
  end
end

function randargs(mdp::MDP)
  nargs = length(mdp.transition.argnames)
  args = Array(Any, nargs)
  for iarg in 1:nargs
    argname = mdp.transition.argnames[iarg]
    if haskey(mdp.statemap, argname)
      args[iarg] = lazySample(mdp.statemap[argname])
    elseif haskey(mdp.actionmap, argname)
      args[iarg] = lazySample(mdp.actionmap[argname])
    else
      error(string("variable ", lazyvar.varname, " has not been defined"))
    end
  end
  args
end

function lazySample(lazyvar::LazyVar)
  if isa(lazyvar, RangeVar)
    return lazyvar.minval + rand() * (lazyvar.maxval - lazyvar.minval)
  elseif isa(lazyvar, ValuesVar)
    return rand(lazyvar.values)
  else
    error(string("variable", lazyvar.varname, " is not a valid subtype of LazyVar"))
  end
end

function isvalid(mdp::MDP, state::Tuple)
  for iarg in length(mdp.transition.argnames)
    argname = mdp.transition.argnames[iarg]
    if isa(mdp.statemap[argname], RangeVar) &&
        (state[iarg] < mdp.statemap[argname].minval ||
        state[iarg] > mdp.statemap[argname].maxval)
      return false
    elseif isa(mdp.statemap[argname], ValuesVar) &&
        !(state[iarg] in mdp.statemap[argname].values)
      return false
    end
  end
  return true
end

function lazySolve!(mdp::MDP, vi::ValueIteration)
  # todo: figure out whether T(s,a,s') or T(s,a) type transition
  # todo: if T(s,a), use gridinterpolations
  # todo: dispatch to serial or parallel value iteration solvers
end

function solve(mdp::MDP, svi::SerialValueIteration)
  # todo
end

function solve(mdp::MDP, pvi::ParallelValueIteration)
  # todo
end
