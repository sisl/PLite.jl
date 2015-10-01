export
  SerialMCTS

type State
  # todo
end

type StateNode
  # todo
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
