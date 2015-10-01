export
  SerialMCTS

type State
  # todo
end

type StateNode

  n::Vector{Int64}  # number of visits at the node for each action
  qval::Vector{Float64}  # estimated value for each action

  StateNode(nactions::Real) =
    new(
      zeros(Int64, int64(nactions)),
      zeros(Float64, int64(nactions)))

end

abstract MCTS <: Solver

type SerialMCTS <: MCTS

  niter::Int64
  maxdepth::Int64
  explr::Float64
  tree::Dict{State, StateNode}

end

type MCTSSolution <: Solution
  # todo
end

function getpolicy(mdp::MDP, solution::MCTSSolution)
  # todo
end

include("mcts_checks.jl")
include("mcts_solver.jl")
