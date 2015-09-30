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

function internalsolve!(mdp::MDP, pvi::ParallelValueIteration)

  solvechunk! = function emptyfunc() end
  if length(mdp.transition.argnames) == length(mdp.statemap) + length(mdp.actionmap)
    solvechunk! = solveset_chunk!
  elseif length(mdp.transition.argnames) > length(mdp.statemap) + length(mdp.actionmap)
    solvechunk! = solveprob_chunk!
  else
    error(string(
      "unkown transition function of neither T(s,a,s') nor T(s,a) type\n",
      "number of transition arguments: ",
      length(mdp.transition.argnames), "\n",
      "number of state and action variables",
      length(mdp.statemap) + length(mdp.actionmap)))
  end

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  nstates = length(pvi.stategrid)
  nactions = length(pvi.actiongrid)

  vold = SharedArray(
    Float64,
    (nstates),
    init = S -> S[localindexes(S)] = 0.0,
    pids = [1:pvi.nthreads])

  vnew = SharedArray(
    Float64,
    (nstates),
    init = S -> S[localindexes(S)] = 0.0,
    pids = [1:pvi.nthreads])

  qval = SharedArray(
    Float64,
    (nactions, nstates),
    init = S -> S[localindexes(S)] = 0.0,
    pids = [1:pvi.nthreads])

  lst = segment(pvi.nthreads, 1:nstates)

  iter = 0
  itertime = 0.0
  cputime = 0.0

  resid = -Inf

  for iter in 1:pvi.maxiter

    tic()
    resid = -Inf

    newresids = pmap(
      x -> (idxs = x; solvechunk!(
        mdp,
        pvi,
        nstates,
        nactions,
        stateargs,
        actionargs,
        vold,
        vnew,
        qval,
        idxs)),
      lst)

    newresid = maximum(newresids)
    newresid > resid ? resid = newresid : nothing
    results = pmap(x -> (idxs = x; updatechunk!(vold, vnew, idxs)), lst)

    vtmp = vold
    vold = vnew
    vnew = vtmp

    itertime = toq()
    cputime += itertime

    if pvi.verbose
      println(string(
        "iter $iter, resid: $resid, itertime: $itertime, cputime: $cputime"))
    end

    resid < pvi.tol ? break : nothing

  end

  if iter == pvi.maxiter
    warn(string(
      "maximum number of iterations reached; check accuracy of solutions"))
  end

  mdp.solution = LazySolution(ValueIterationSolution(
    shmat2mat(qval),
    pvi.stategrid,
    pvi.actiongrid,
    cputime,
    iter,
    resid))

  info(string(
    "value iteration solution generated\n",
    "cputime [s] = ", cputime, "\n",
    "number of iterations = ", iter, "\n",
    "residual = ", resid))

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
    resid = -Inf

    for istate in 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      qhi = -Inf

      for iaction in 1:nactions

        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)

        statepIdxs, probs = transition(mdp, svi, state, action, stateargs)
        qnow = 0.0

        for istatep in 1:length(statepIdxs)
          qnow += probs[istatep] * vold[statepIdxs[istatep]]
        end

        qnow *= svi.discount
        qnow += reward(mdp, state, action)

        qval[iaction, istate] = qnow

        if qnow > qhi
          qhi = qnow
          vnew[istate] = qhi
        end

      end

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? resid = newresid : nothing

    end

    itertime = toq()
    cputime += itertime

    if svi.verbose
      println(string(
        "iter $iter, resid: $resid, itertime: $itertime, cputime: $cputime"))
    end

    resid < svi.tol ? break : nothing

    vtmp = vold
    vold = vnew
    vnew = vtmp

  end

  if iter == svi.maxiter
    warn(string(
      "maximum number of iterations reached; check accuracy of solutions"))
  end

  mdp.solution = LazySolution(ValueIterationSolution(
    qval,
    svi.stategrid,
    svi.actiongrid,
    cputime,
    iter,
    resid))

  info(string(
    "value iteration solution generated\n",
    "cputime [s] = ", cputime, "\n",
    "number of iterations = ", iter, "\n",
    "residual = ", resid))

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
  cputime = 0.0

  for iter in 1:svi.maxiter

    tic()
    resid = -Inf

    for istate in 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      qhi = -Inf

      for iaction in 1:nactions

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

        if qnow > qhi
          qhi = qnow
          vnew[istate] = qhi
        end

      end

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? resid = newresid : nothing

    end

    itertime = toq()
    cputime += itertime

    if svi.verbose
      println(string(
        "iter $iter, resid: $resid, itertime: $itertime, cputime: $cputime"))
    end

    resid < svi.tol ? break : nothing

    vtmp = vold
    vold = vnew
    vnew = vtmp

  end

  if iter == svi.maxiter
    warn(string(
      "maximum number of iterations reached; check accuracy of solutions"))
  end

  mdp.solution = LazySolution(ValueIterationSolution(
    qval,
    svi.stategrid,
    svi.actiongrid,
    cputime,
    iter,
    resid))

  info(string(
    "value iteration solution generated\n",
    "cputime [s] = ", cputime, "\n",
    "number of iterations = ", iter, "\n",
    "residual = ", resid))

end

function solveset_chunk!(
    mdp::MDP,
    pvi::ParallelValueIteration,
    nstates::Int64,
    nactions::Int64,
    stateargs::Vector{String},
    actionargs::Vector{String},
    vold::SharedArray{Float64, 1},
    vnew::SharedArray{Float64, 1},
    qval::SharedArray{Float64, 2},
    istates::UnitRange)

  resid = -Inf

  for istate in istates

    state = getvar(pvi.stategrid, mdp.statemap, stateargs, istate)
    qhi = -Inf

    for iaction in 1:nactions

      action = getvar(pvi.actiongrid, mdp.actionmap, actionargs, iaction)

      statepIdxs, probs = transition(mdp, pvi, state, action, stateargs)
      qnow = 0.0

      for istatep in 1:length(statepIdxs)
        qnow += probs[istatep] * vold[statepIdxs[istatep]]
      end

      qnow *= pvi.discount
      qnow += reward(mdp, state, action)

      qval[iaction, istate] = qnow

      if qnow > qhi
        qhi = qnow
        vnew[istate] = qhi
      end

    end

    # use infinity-norm
    newresid = (vold[istate] - vnew[istate])^2
    newresid > resid ? resid = newresid : nothing

  end

  return resid

end

function solveprob_chunk!(
    mdp::MDP,
    pvi::ParallelValueIteration,
    nstates::Int64,
    nactions::Int64,
    stateargs::Vector{String},
    actionargs::Vector{String},
    vold::SharedArray{Float64, 1},
    vnew::SharedArray{Float64, 1},
    qval::SharedArray{Float64, 2},
    istates::UnitRange)

  resid = -Inf

  for istate in istates

    state = getvar(pvi.stategrid, mdp.statemap, stateargs, istate)
    qhi = -Inf

    for iaction in 1:nactions

      action = getvar(pvi.actiongrid, mdp.actionmap, actionargs, iaction)
      qnow = 0.0

      for istatep in 1:nstates
        statep = getvar(pvi.stategrid, mdp.statemap, stateargs, istatep)
        prob = mdp.transition.fn(state..., action..., statep...)
        qnow += prob * vold[istatep]
      end

      qnow *= pvi.discount
      qnow += reward(mdp, state, action)

      qval[iaction, istate] = qnow

      if qnow > qhi
        qhi = qnow
        vnew[istate] = qhi
      end

    end

    # use infinity-norm
    newresid = (vold[istate] - vnew[istate])^2
    newresid > resid ? (resid = newresid) : (nothing)

  end

  return resid

end

function updatechunk!(
    vold::SharedArray{Float64, 1},
    vnew::SharedArray{Float64, 1},
    idxs::UnitRange)

  for i in idxs
    vold[i] = vnew[i]
  end
  return idxs
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
