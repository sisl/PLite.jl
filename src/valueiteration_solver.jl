function lazySolve!(mdp::MDP, vi::ValueIteration)
  statespace, actionspace = getspaces(mdp, vi)
  vi.stategrid = RectangleGrid(statespace...)
  vi.actiongrid = RectangleGrid(actionspace...)
  solve(mdp, vi)
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
      space[ivar] = [lazy.minval : discmap[statevar].step : lazy.maxval]
    elseif isa(lazy, ValuesVar)
      space[ivar] = float64([1:length(lazy.values)])  # map to indices
    else
      error(string(
        "unknown state/action variable definition type for ", var))
    end
  end
  return space
end

function solve(mdp::MDP, svi::SerialValueIteration)
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

      state = getstate(mdp, vi, istate)
      qhi = 0.0

      for iaction = 1:nactions

        action = getaction(mdp, vi, iaction)

        statepIdxs, probs = transition(svi, state, action)
        qnow = reward(svi, state, action)

        for istatep in 1:length(statepIdxs)
          qnow += probs[istatep] * vold[statepIdxs[istatep]]
        end

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

    if resid < svi.tol; lastidx = uidx; break; end

  end

  mdp.solution = LazySolution(ValueIterationSolution(
    qval,
    cputime,
    iter,
    resid))
end

function solve(mdp::MDP, pvi::ParallelValueIteration)
  # todo
end

function getstate(mdp::MDP, vi::ValueIteration, istate::Int64)
  rawstate = ind2x(vi.stategrid, istate)
  # todo
end

function getaction(mdp::MDP, vi::ValueIteration, istate::Int64)
  rawaction = ind2x(vi.stategrid, iaction)
  # todo
end

function transition(mdp::MDP, state::Vector, action::Vector)
  # todo: deal with both T(s,a) and T(s,a,s') type transitions
  # todo: remember to convert state and action index elements to their actual ones for ValuesVar
  # todo: return states-probs vectors pair
end

function reward(mdp::MDP, state::Vector, action::Vector)
  # todo
end
