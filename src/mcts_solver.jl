function lazySolve(mdp::MDP, mcts::MCTS)
  statespace, actionspace = getspaces(mdp, mcts)
  mcts.stategrid = RectangleGrid(statespace...)
  mcts.actiongrid = RectangleGrid(actionspace...)
  return internalsolve(mdp, mcts)
end

function getspaces(mdp::MDP, mcts::MCTS)
  statedim = length(mdp.statemap)
  actiondim = length(mdp.actionmap)

  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  statespace = getspace(statedim, stateargs, mdp.statemap, mcts.statemap)
  actionspace = getspace(actiondim, actionargs, mdp.actionmap, mcts.actionmap)

  return statespace, actionspace
end

function getspace(
    dim::Int64,
    args::Vector{ASCIIString},
    lazymap::Dict{AbstractString, LazyVar},
    discmap::Dict{AbstractString, LazyDiscrete})

  space = Array(Vector{Float64}, dim)
  for ivar in 1:length(args)
    var = args[ivar]
    lazy = lazymap[var]

    if isa(lazy, RangeVar)
      space[ivar] = collect(lazy.minval : discmap[var].step : lazy.maxval)
    elseif isa(lazy, ValuesVar)
      space[ivar] = map(Float64, collect(1:length(lazy.values)))  # map to indices
    else
      error(string(
        "unknown state/action variable definition type for ", var))
    end
  end

  return space

end

include("mcts_serial.jl")
