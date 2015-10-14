function internalsolve(mdp::MDP, smcts::SerialMCTS)
  return smcts
end

function smctsaction!(mdp::MDP, smcts::SerialMCTS, actualstate::Vector)
  for iter in 1:smcts.niter
    simulate(mdp, smcts, smcts.maxdepth, actualstate)
  end
  statedim = length(mdp.statemap)
  actionargs = mdp.reward.argnames[1 + statedim:end]
  iaction = indmax(smcts.tree[actualstate].qval)
  return getvar(smcts.actiongrid, mdp.actionmap, actionargs, iaction)
end

function simulate(mdp::MDP, smcts::SerialMCTS, depth::Int64, actualstate::Vector)

  if depth == 0
    return 0.0
  end

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]

  state = getgridstate(smcts.stategrid, mdp.statemap, stateargs, actualstate)

  # run rollout and add statistics to tree if state unexplored
  if !haskey(smcts.tree, actualstate)

    smcts.tree[deepcopy(actualstate)] = StateNode(length(smcts.actiongrid))
    return rollout(mdp, smcts, depth, state)

  else

    actionargs = mdp.reward.argnames[1 + statedim:end]
    snode = smcts.tree[actualstate]

    # pick action using UCT
    iaction = indmax(snode.qval + smcts.exex * real(sqrt(complex(log(sum(snode.n)) ./ snode.n))))
    action = ind2x(smcts.actiongrid, iaction)

    # transition to next state
    nextstate, reward = sampletransition(mdp, smcts, state, action)

    # update n and qval in state node
    actualstate = getvar(mdp.statemap, stateargs, state)
    actualaction = getvar(mdp.actionmap, actionargs, action)
    actualnextstate = getvar(mdp.statemap, stateargs, nextstate)

    q = reward + smcts.discount * simulate(mdp, smcts, depth - 1, actualnextstate)

    snode.n[iaction] += 1
    snode.qval[iaction] += (q - snode.qval[iaction]) / snode.n[iaction]

    return q

  end

end

function rollout(mdp::MDP, smcts::SerialMCTS, depth::Int64, state::Vector{Float64})
  if depth == 0
    return 0.0
  else
    iaction = rand(1:length(smcts.actiongrid))  # uniformly random rollout
    action = ind2x(smcts.actiongrid, iaction)
    nextstate, reward = sampletransition(mdp, smcts, state, action)
    return reward + smcts.discount * rollout(mdp, smcts, depth - 1, nextstate)
  end
end

function sampletransition(
    mdp::MDP,
    smcts::SerialMCTS,
    state::Vector{Float64},
    action::Vector{Float64})

  if length(mdp.transition.argnames) == length(mdp.statemap) + length(mdp.actionmap)
    return sampleset(mdp, smcts, state, action)
  elseif length(mdp.transition.argnames) > length(mdp.statemap) + length(mdp.actionmap)
    return sampleprob(mdp, smcts, state, action)
  else
    error(string(
      "unkown transition function of neither T(s,a,s') nor T(s,a) type\n",
      "number of transition arguments: ",
      length(mdp.transition.argnames), "\n",
      "number of state and action variables",
      length(mdp.statemap) + length(mdp.actionmap)))
  end
end

function sampleset(
    mdp::MDP,
    smcts::SerialMCTS,
    state::Vector{Float64},
    action::Vector{Float64})

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  actualstate = getvar(mdp.statemap, stateargs, state)
  actualaction = getvar(mdp.actionmap, actionargs, action)

  nextstateprobs = mdp.transition.fn(actualstate..., actualaction...)
  weightvec = WeightVec([nextstateprob[2]::Float64
    for nextstateprob in nextstateprobs])

  nextstateprob = sample(nextstateprobs, weightvec)
  nextstate = getgridstate(smcts.stategrid, mdp.statemap, stateargs, nextstateprob[1])
  reward = mdp.reward.fn(actualstate..., actualaction...)

  return nextstate, reward

end

function sampleprob(
    mdp::MDP,
    smcts::SerialMCTS,
    state::Vector{Float64},
    action::Vector{Float64})

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  nstates = length(smcts.stategrid)

  actualstate = getvar(mdp.statemap, stateargs, state)
  actualaction = getvar(mdp.actionmap, actionargs, action)

  probs = zeros(nstates)
  for inextstate in 1:nstates
    actualnextstate = getvar(smcts.stategrid, mdp.statemap, stateargs, inextstate)
    probs[inextstate] = mdp.transition.fn(actualstate..., actualaction..., actualnextstate...)
  end
  weightvec = WeightVec(probs)

  nextstate = ind2x(smcts.stategrid, sample(1:nstates, weightvec))
  reward = mdp.reward.fn(actualstate..., actualaction...)

  return nextstate, reward

end

# Returns the actual variable from GridInterpolations indices
function getvar(
    grid::RectangleGrid,
    map::Dict{AbstractString, LazyVar},
    argnames::Vector{ASCIIString},
    index::Int64)

  raw = ind2x(grid, index)
  var = Array(Any, length(raw))

  for ivar in 1:length(raw)
    lazy = map[argnames[ivar]]
    if isa(lazy, RangeVar)
      var[ivar] = raw[ivar]
    elseif isa(lazy, ValuesVar)
      var[ivar] = lazy.values[Int64(raw[ivar])]
    else
      error(string(
        "unknown state/action variable definition type for ", argnames[ivar]))
    end
  end

  return var

end

# Returns the actual variable from GridInterpolations grid state
function getvar(
    map::Dict{AbstractString, LazyVar},
    argnames::Vector{ASCIIString},
    raw::Vector{Float64})

  var = Array(Any, length(raw))

  for ivar in 1:length(raw)
    lazy = map[argnames[ivar]]
    if isa(lazy, RangeVar)
      var[ivar] = raw[ivar]
    elseif isa(lazy, ValuesVar)
      var[ivar] = lazy.values[Int64(raw[ivar])]
    else
      error(string(
        "unknown state/action variable definition type for ", argnames[ivar]))
    end
  end

  return var

end

# Inverse of getvar; returns the GridInterpolations state for an actual state
function getgridstate(
    grid::RectangleGrid,
    map::Dict{AbstractString, LazyVar},
    argnames::Vector{ASCIIString},
    state::Vector)

  gridstate = Array(Float64, length(state))

  for i in 1:length(gridstate)
    lazy = map[argnames[i]]
    if isa(lazy, RangeVar)
      cutpoints = grid.cutPoints[i]
      gridstate[i] = cutpoints[indmin(abs(cutpoints - state[i]))]
    elseif isa(lazy, ValuesVar)
      gridstate[i] = findfirst(lazy.values, state[i])
    else
      error(string(
        "unknown state/action variable definition type for ", argnames[ivar]))
    end
  end

  return gridstate

end