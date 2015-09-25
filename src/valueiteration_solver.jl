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

function internalsolve!(mdp::MDP, svi::SerialValueIteration)
  if length(mdp.transition.argnames) == length(mdp.statemap) + length(mdp.actionmap)
    solveset!(mdp, svi)
  elseif length(mdp.transition.argnames) > length(mdp.statemap) + length(mdp.actionmap)
    solveprob!(mdp, svi)
  else
    error(string(
      "unkown transition function of neither T(s,a,s') nor T(s,a) type\n",
      "number of transition arguments: ",
      length(mdp.transition.argnames), "\n",
      "number of state and action variables",
      length(mdp.statemap) + length(mdp.actionmap)))
  end
end

function solveset!(mdp::MDP, svi::SerialValueIteration)
  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  nstates = length(svi.stategrid)
  nactions = length(svi.actiongrid)

  vold = zeros(nstates)
  vnew = zeros(nstates)
  qval = zeros(nactions, nstates)
  resid = 0.0

  iter = 0
  itertime = 0.0
  cputime = 0.0

  for iter in 1:svi.maxiter

    tic()
    resid = 0.0

    for istate = 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      qhi = 0.0

      for iaction = 1:nactions

        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)

        statepIdxs, probs = transition(mdp, svi, state, action, stateargs)
        qnow = 0.0

        for istatep in 1:length(statepIdxs)
          qnow += probs[istatep] * vold[statepIdxs[istatep]]
        end

        qnow *= svi.discount
        qnow += reward(mdp, state, action)

        qval[iaction, istate] = qnow

        if iaction == 1 || qnow > qhi
          qhi = qnow
          vnew[istate] = qhi
        end

      end

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? (resid = newresid) : (nothing)

      vold[istate] = vnew[istate]

    end

    itertime = toq()
    cputime += itertime

    svi.verbose ?
      (println(
        "iter $iter, resid: $resid, itertime: $itertime, cputime: $cputime")) :
      nothing

    resid < svi.tol ? break : nothing

  end

  if iter == svi.maxiter
    warn(string("maximum number of iterations reached; check accuracy of solutions"))
  end

  mdp.solution = LazySolution(ValueIterationSolution(
    qval,
    cputime,
    iter,
    resid))
end

function solveprob!(mdp::MDP, svi::SerialValueIteration)
  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  nstates = length(svi.stategrid)
  nactions = length(svi.actiongrid)

  vold = zeros(nstates)
  vnew = zeros(nstates)
  qval = zeros(nactions, nstates)
  resid = 0.0

  iter = 0
  itertime = 0.0
  totaltime = 0.0

  for iter in 1:svi.maxiter

    tic()
    for istate = 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      qhi = 0.0

      for iaction = 1:nactions

        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)
        qnow = 0.0

        for istatep in 1:nstates
          statep = getvar(svi.stategrid, mdp.statemap, stateargs, istatep)
          prob = mdp.transition.fn(state..., action..., statep...)
          qnow += prob * vold[istatep]
        end

        qnow *= svi.discount
        qnow += reward(mdp, state, action)

        qval[iaction, istate] = qnow

        if iaction == 1 || qnow > qhi
          qhi = qnow
          vnew[istate] = qhi
        end

      end

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? (resid = newresid) : (nothing)

      vold[istate] = vnew[istate]

    end

    itertime = toq()
    cputime += itertime
    svi.verbose ?
      (println("iter $iter, resid: $resid, itertime: $itertime, cputime: $cputime")) : nothing

    resid < svi.tol ? break : nothing

  end

  mdp.solution = LazySolution(ValueIterationSolution(
    qval,
    cputime,
    iter,
    resid))
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

function solve!(mdp::MDP, pvi::ParallelValueIteration)
  # todo
end