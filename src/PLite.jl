module PLite

include("solvers.jl")

export
  MDP,
  statevariable!,
  actionvariable!,
  transition!,
  reward!,
  solver!,
  solve!

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

abstract type LazyVar

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
  fn::Function
  argnames::Vector{String}

  LazyFunc() = new(true, function emptyfunc() end, Array(String, 0))
  LazyFunc(fn::Function, argnames::Vector{String}) = new(false, fn, argnames)

end

abstract type Solution

type EmptySolution <: Solution
end

type LazySolution

  empty::Bool
  solution::Solution

  LazySolution() = new(true, EmptySolution())
  LazySolution(solution::Solution) = new(false, solution)

end

function statevariable!(mdp::MDP, varname::String, minval::Float64, maxval::Float64)
  if haskey(mdp.statemap, varname)
    warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = RangeVar(varname, minval, maxval)
end

function statevariable!(mdp::MDP, varname::String, values::Vector)
  if haskey(mdp.statemap, varname)
    warn(string(
      "state variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.statemap[varname] = ValuesVar(varname, values)
end

function actionvariable!(mdp::MDP, varname::String, minval::Float64, maxval::Float64)
  if haskey(mdp.actionmap, varname)
    warn(string(
      "action variable ", varname, " already exists in MDP object, ",
      "replacing existing variable definition"))
  end
  mdp.actionmap[varname] = RangeVar(varname, minval, maxval)
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
function transition!(mdp::MDP, transition::Function, argnames::Vector{String})
  if !mdp.transition.empty
    warn(string(
      "transition function already exists in MDP object, ",
      "replacing existing function definition"))
  end
  mdp.transition = LazyFunc(transition, argnames)
end

# |argnames| is an ordered list of argument names for |reward|
function reward!(mdp::MDP, reward::Function, argnames::Vector{String})
  if !mdp.reward.empty
    warn(string(
      "reward function already exists in MDP object, ",
      "replacing existing function definition"))
  end
  reward.transition = LazyFunc(reward, argnames)
end

function solve!(mdp::MDP, solver::Solver)
  lazyCheck(mdp, solver)
  lazySolve!(mdp, solver)  # replaces existing |mdp.solution|
end
