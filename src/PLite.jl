module PLite

export
  MDP,
  statevariable!,
  actionvariable!,
  transition!,
  reward!,
  solver!,
  solve!

abstract LazyVar

type RangeVar <: LazyVar

  varname::String
  minval::Float64
  maxval::Float64

  function RangeVar(varname::String, minval::Float64, maxval::Float64)
    if minval > maxval
      throw(ArgumentError("minimum value must be smaller than maximum value"))
    end
    new(varname, minval, maxval)
  end

end

type ValuesVar <: LazyVar

  varname::String
  values::Vector

  ValuesVar(varname::String, values::Vector) = new(varname, values)

end

type LazyFunc

  empty::Bool
  argnames::Vector{String}
  fn::Function

  LazyFunc() = new(true, Array(String, 0), function emptyfunc() end)
  LazyFunc(argnames::Vector{ASCIIString}, fn::Function) = new(false, argnames, fn)

end

abstract Solution

type EmptySolution <: Solution
end

type LazySolution

  empty::Bool
  solution::Solution

  LazySolution() = new(true, EmptySolution())
  LazySolution(solution::Solution) = new(false, solution)

end

type MDP

  statemap::Dict{String, LazyVar}
  actionmap::Dict{String, LazyVar}
  transition::LazyFunc
  reward::LazyFunc
  solution::LazySolution

  MDP() = new(
    Dict{String, LazyVar}(),
    Dict{String, LazyVar}(),
    LazyFunc(),
    LazyFunc(),
    LazySolution())

end

include("solvers.jl")

function statevariable!(mdp::MDP, varname::String, minval::Real, maxval::Real)
  if haskey(mdp.statemap, varname)
    warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = RangeVar(varname, float64(minval), float64(maxval))
end

function statevariable!(mdp::MDP, varname::String, values::Vector)
  if haskey(mdp.statemap, varname)
    warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = ValuesVar(varname, values)
end

function actionvariable!(mdp::MDP, varname::String, minval::Real, maxval::Real)
  if haskey(mdp.actionmap, varname)
    warn(string(
      "action variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.actionmap[varname] = RangeVar(varname, float64(minval), float64(maxval))
end

function actionvariable!(mdp::MDP, varname::String, values::Vector)
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

function solve!(mdp::MDP, solver::Solver)
  lazyCheck(mdp, solver)
  lazySolve!(mdp, solver)  # replaces existing |mdp.solution|
end

end
