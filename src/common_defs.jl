export
  MDP,
  statevariable!,
  actionvariable!,
  transition!,
  reward!,
  solve,
  getpolicy

abstract LazyVar

type RangeVar <: LazyVar

  varname::AbstractString
  minval::Float64
  maxval::Float64

  function RangeVar(varname::AbstractString, minval::Float64, maxval::Float64)
    if minval > maxval
      throw(ArgumentError("minimum value must be smaller than maximum value"))
    end
    new(varname, minval, maxval)
  end

end

type ValuesVar <: LazyVar

  varname::AbstractString
  values::Vector

  ValuesVar(varname::AbstractString, values::Vector) = new(varname, values)

end

type LazyFunc

  empty::Bool
  argnames::Vector{ASCIIString}
  fn::Function

  LazyFunc() = new(true, Array(AbstractString, 0), function emptyfunc() end)
  LazyFunc(argnames::Vector{ASCIIString}, fn::Function) = new(false, argnames, fn)

end

type MDP

  statemap::Dict{AbstractString, LazyVar}
  actionmap::Dict{AbstractString, LazyVar}
  transition::LazyFunc
  reward::LazyFunc

  MDP() = new(
    Dict{AbstractString, LazyVar}(),
    Dict{AbstractString, LazyVar}(),
    LazyFunc(),
    LazyFunc())

end

abstract Solver
abstract Solution

function statevariable!(mdp::MDP, varname::AbstractString, minval::Real, maxval::Real)
  if haskey(mdp.statemap, varname)
    warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = RangeVar(varname, Float64(minval), Float64(maxval))
end

function statevariable!(mdp::MDP, varname::AbstractString, values::Vector)
  if haskey(mdp.statemap, varname)
    warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = ValuesVar(varname, values)
end

function actionvariable!(mdp::MDP, varname::AbstractString, minval::Real, maxval::Real)
  if haskey(mdp.actionmap, varname)
    warn(string(
      "action variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.actionmap[varname] = RangeVar(varname, Float64(minval), Float64(maxval))
end

function actionvariable!(mdp::MDP, varname::AbstractString, values::Vector)
  if haskey(mdp.actionmap, varname)
    warn(string(
      "action variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.actionmap[varname] = ValuesVar(varname, values)
end

# |argnames| is an ordered list of argument names for |transition|
function transition!(mdp::MDP, argnames::Vector{ASCIIString}, transition::Function)
  if !mdp.transition.empty
    warn(string(
      "transition function already exists in MDP object, ",
      "replacing existing function definition"))
  end
  mdp.transition = LazyFunc(argnames, transition)
end

# |argnames| is an ordered list of argument names for |reward|
function reward!(mdp::MDP, argnames::Vector{ASCIIString}, reward::Function)
  if !mdp.reward.empty
    warn(string(
      "reward function already exists in MDP object, ",
      "replacing existing function definition"))
  end
  mdp.reward = LazyFunc(argnames, reward)
end

function solve(mdp::MDP, solver::Solver)
  lazyCheck(mdp, solver)
  return lazySolve(mdp, solver)
end
