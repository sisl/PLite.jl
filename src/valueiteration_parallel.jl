function internalsolve(mdp::MDP, pvi::ParallelValueIteration)

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

  info(string(
    "value iteration solution generated\n",
    "cputime [s] = ", cputime, "\n",
    "number of iterations = ", iter, "\n",
    "residual = ", resid))

  return ValueIterationSolution(
    shmat2mat(qval),
    pvi.stategrid,
    pvi.actiongrid,
    cputime,
    iter,
    resid)

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
