function lazySolve!(mdp::MDP, vi::ValueIteration)
  statespace, actionspace = getspaces(mdp, vi)
  vi.stategrid = RectangleGrid(statespace...)
  vi.actiongrid = RectangleGrid(actionspace...)
  internalsolve!(mdp, vi)
end

function getspaces(mdp::MDP, vi::ValueIteration)
  statedim = length(mdp.statemap)
  actiondim = length(mdp.actionmap)

  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  statespace = getspace(statedim, stateargs, mdp.statemap, vi.statemap)
  actionspace = getspace(actiondim, actionargs, mdp.actionmap, vi.actionmap)

  return statespace, actionspace
end

function getspace(
    dim::Int64,
    args::Vector{String},
    lazymap::Dict{String, LazyVar},
    discmap::Dict{String, LazyDiscrete})

  space = Array(Vector{Float64}, dim)
  for ivar in 1:length(args)
    var = args[ivar]
    lazy = lazymap[var]

    if isa(lazy, RangeVar)
      space[ivar] = [lazy.minval : discmap[var].step : lazy.maxval]
    elseif isa(lazy, ValuesVar)
      space[ivar] = float64([1:length(lazy.values)])  # map to indices
    else
      error(string(
        "unknown state/action variable definition type for ", var))
    end
  end

  return space

end

# Inverse of getvar; returns the GridInterpolations index for a state
function getidx(
    grid::RectangleGrid,
    map::Dict{String, LazyVar},
    argnames::Vector{String},
    state::Vector)

  idx = Array(Float64, length(state))

  for i in 1:length(idx)
    lazy = map[argnames[i]]
    if isa(lazy, RangeVar)
      idx[i] = state[i]
    elseif isa(lazy, ValuesVar)
      idx[i] = findfirst(lazy.values, state[i])
    else
      error(string(
        "unknown state/action variable definition type for ", argnames[ivar]))
    end
  end

  return idx

end

function transition(
    mdp::MDP,
    vi::ValueIteration,
    state::Vector,
    action::Vector,
    stateargnames::Vector{String})

  results = mdp.transition.fn(state..., action...)
  nresults = length(results)

  interps = [
    interpolants(
      vi.stategrid,
      getidx(vi.stategrid, mdp.statemap, stateargnames, results[i][1]))
    for i in 1:nresults]

  nstateps = 0
  for interp in interps
    nstateps += length(interp[1])
  end

  states = Array(Int64, nstateps)
  probs = Array(Float64, nstateps)

  istate = 0
  for iresult in 1:nresults
    for iinterp in 1:length(interps[iresult][1])
      istate += 1
      states[istate] = interps[iresult][1][iinterp]
      probs[istate] = interps[iresult][2][iinterp] * results[iresult][2]
    end
  end

  return states, probs
end

function reward(mdp::MDP, state::Vector, action::Vector)
  mdp.reward.fn(state..., action...)
end

function segment(n::Int64, u::UnitRange)
    chunks = {}

    cov = u.stop - u.start
    stride = int(floor((u.stop - u.start) / (n-1)))
    for i = 1:(n-1)
        si = u.start + (i-1)*stride
        ei = si + stride - 1
        i == (n-1) ? (ei = u.stop) : nothing
        push!(chunks, si:ei)
    end
    return chunks
end

function shmat2mat(shmat::SharedArray{Float64, 2})
    result = zeros(shmat.dims)
    for i in 1:shmat.dims[1]
        for j in 1:shmat.dims[2]
            result[i, j] = shmat[i, j]
        end # for j
    end # for i
    return result
end # function shmat2mat

include("valueiteration_serial.jl")
include("valueiteration_parallel.jl")
