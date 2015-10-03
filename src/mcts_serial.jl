function internalsolve(mdp::MDP, smcts::SerialMCTS)
  # todo
end

function rollout(mdp::MDP, smcts::SerialMCTS, depth::Int64, state::Vector{Float64})
  if depth == 0
    return 0.0
  else
    iaction = rand(1:length(scmts.actionmap))  # uniformly random rollout
    action = ind2x(scmts.actiongrid, iaction)
    nextstate, reward = sampletransition(mdp, smcts, state, action)
    return reward + scmts.discount * rollout(mdp, scmts, depth - 1, nextstate)
  end
  # todo: tail call elimination
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

  actualstate = # todo
  actualaction = # todo

  nextstateprobs = mdp.transition.fn(actualstate..., actualaction...)
  weightvec = WeightVec([nextstateprob[2] for nextstateprob in nextstateprobs)])

  nextstateprob = sample(nextstateprobs, weightvec)
  nextstate = # todo: get gridstate of nextstateprob[1]
  reward = mdp.reward.fn(actualstate..., actualaction...)

  return nextstate, reward

end

function sampleprob(
    mdp::MDP,
    smcts::SerialMCTS,
    state::Vector{Float64},
    action::Vector{Float64}))

  # todo differentiate between gridstate and actualstate (and actions)

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  nstates = length(smcts.stategrid)

  probs = zeros(nstates)
  for inextstate in 1:nstates
    nextstate = getvar(smcts.stategrid, mdp.statemap, stateargs, inextstate)
    probs[inextstate] = mdp.transition.fn(state..., action..., nextstate...)
  end
  weightvec = WeightVec(probs)

  nextstate = ind2x(smcts.stategrid, sample(1:nstates, weightvec))
  reward = mdp.reward.fn(state..., action...)

  return nextstate, reward

end

# Returns the actual variable from GridInterpolations indices
function getvar(
    grid::RectangleGrid,
    map::Dict{String, LazyVar},
    argnames::Vector{String},
    index::Int64)

  raw = ind2x(grid, index)
  var = Array(Any, length(raw))

  for ivar in 1:length(raw)
    lazy = map[argnames[ivar]]
    if isa(lazy, RangeVar)
      var[ivar] = raw[ivar]
    elseif isa(lazy, ValuesVar)
      var[ivar] = lazy.values[raw[ivar]]
    else
      error(string(
        "unknown state/action variable definition type for ", argnames[ivar]))
    end
  end

  return var

end

# Returns the actual variable from GridInterpolations grid state
function getvar(
    grid::RectangleGrid,
    map::Dict{String, LazyVar},
    argnames::Vector{String},
    raw::Vector{Float64})

  var = Array(Any, length(raw))

  for ivar in 1:length(raw)
    lazy = map[argnames[ivar]]
    if isa(lazy, RangeVar)
      var[ivar] = raw[ivar]
    elseif isa(lazy, ValuesVar)
      var[ivar] = lazy.values[raw[ivar]]
    else
      error(string(
        "unknown state/action variable definition type for ", argnames[ivar]))
    end
  end

  return var

end
