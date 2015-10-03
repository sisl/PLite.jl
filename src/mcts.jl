#= Implements Monte-Carlo tree search algorithm.

A key assumption is that both the action space and the state space are finite.
Otherwise, we will keep selecting unexplored actions, and no node of depth
higher than one would be added. Thus, after the state and action spaces are
discretized, the algorithm only works with these discrete states and actions.
For transition, we will map the given current state and action pair to the
closest discrete state and action. Additionally, the next state transitioned
into (via sampling) will also be mapped to the closest discrete state.
=#

using
  StatsBase

export
  SerialMCTS

const NIter = 50
const MaxDepth = 20
const ExploreExploit = 3.0
const Discount = 0.99

type StateNode

  n::Vector{Int64}  # number of visits at the node for each action
  qval::Vector{Float64}  # estimated value for each action

  StateNode(nactions::Real) =
    new(zeros(Int64, int64(nactions)), zeros(Float64, int64(nactions)))

end

type LazyDiscrete

  varname::String
  step::Float64

  LazyDiscrete(varname::String, step::Float64) = new(varname, step)

end

abstract MCTS <: Solver

type SerialMCTS <: MCTS

  niter::Int64  # number of iterations during each action selection
  maxdepth::Int64  # max depth of search tree
  exex::Float64  #  exploration-exploitation constant
  discount::Float64  # simulation/rollout discount factor

  tree::Dict{State, StateNode}
  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  SerialMCTS(;
      niter::Real=NIter,
      maxdepth::Real=MaxDepth,
      exex::Real=ExploreExploit,
      discount::Real=Discount) =
    new(
      int64(niter),
      int64(maxdepth),
      float64(exex),
      float64(discount),
      Dict{State, StateNode}(),
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

end

type MCTSSolution <: Solution

  tree::Dict{State, StateNode}
  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  MCTSSolution(
      tree::Dict{State, StateNode},
      statemap::Dict{String, LazyDiscrete},
      actionmap::Dict{String, LazyDiscrete},
      stategrid::RectangleGrid,
      actiongrid::RectangleGrid) =
    new (tree, statemap, actionmap, stategrid, actiongrid)

end

function discretize_statevariable!(mcts::MCTS, varname::String, step::Real)
  if haskey(mcts.statemap, varname)
    warn(string(
      "state variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  mcts.statemap[varname] = LazyDiscrete(varname, float64(step))
end

function discretize_actionvariable!(mcts::MCTS, varname::String, step::Real)
  if haskey(mcts.actionmap, varname)
    warn(string(
      "action variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  mcts.actionmap[varname] = LazyDiscrete(varname, float64(step))
end

# returned policy is "live" and grows the search tree
function getpolicy(mdp::MDP, solution::MCTSSolution)
  # todo
end

include("mcts_checks.jl")
include("mcts_solver.jl")
