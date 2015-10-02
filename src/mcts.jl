export
  SerialMCTS

const NIter = 50
const MaxDepth = 20
const ExploreExploit = 3.0

typealias State Vector{Any}

type StateNode

  n::Vector{Int64}  # number of visits at the node for each action
  qval::Vector{Float64}  # estimated value for each action

  StateNode(nactions::Real) =
    new(zeros(Int64, int64(nactions)), zeros(Float64, int64(nactions)))

end

abstract MCTS <: Solver

type SerialMCTS <: MCTS

  niter::Int64  # number of iterations during each action selection
  maxdepth::Int64  # max depth of search tree
  exex::Float64  #  exploration-exploitation constant

  tree::Dict{State, StateNode}

  SerialMCTS(;
      niter::Int64=NIter,
      maxdepth::Int64=MaxDepth,
      exex::Float64=ExploreExploit) =
    new(niter, maxdepth, explore, Dict{State, StateNode}())

end

type MCTSSolution <: Solution

  tree::Dict{State, StateNode}
  actiongrid::RectangleGrid

end

function getpolicy(mdp::MDP, solution::MCTSSolution)
  # todo
end

include("mcts_checks.jl")
include("mcts_solver.jl")
