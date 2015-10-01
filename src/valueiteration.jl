using
  GridInterpolations

export
  SerialValueIteration,
  ParallelValueIteration,
  discretize_statevariable!

const MaxIter = 1000
const Tol = 1e-4
const Discount = 0.99
const NThreads = CPU_CORES / 2

type LazyDiscrete

  varname::String
  step::Float64

  LazyDiscrete(varname::String, step::Float64) = new(varname, step)

end

abstract ValueIteration <: Solver

type SerialValueIteration <: ValueIteration

  verbose::Bool

  maxiter::Int64
  tol::Float64
  discount::Float64

  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  SerialValueIteration(;
      verbose::Bool=true,
      maxiter::Real=MaxIter,
      tol::Float64=Tol,
      discount::Real=Discount) =
    new(
      verbose,
      int64(maxiter),
      tol,
      float64(discount),
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

end

type ParallelValueIteration <: ValueIteration

  nthreads::Int64
  verbose::Bool

  maxiter::Int64
  tol::Float64
  discount::Float64

  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  ParallelValueIteration(;
      nthreads::Real=NThreads,
      verbose::Bool=true,
      maxiter::Real=MaxIter,
      tol::Float64=Tol,
      discount::Real=Discount) =
    new(
      int64(nthreads),
      verbose,
      int64(maxiter),
      tol,
      float64(discount),
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

end

type ValueIterationSolution <: Solution

  qval::Matrix{Float64}  # nactions x nstates Q-value matrix
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  cputime::Float64
  niter::Int64
  finaltol::Float64

  ValueIterationSolution(
      qval::Matrix{Float64},
      stategrid::RectangleGrid,
      actiongrid::RectangleGrid,
      cputime::Float64,
      niter::Int64,
      finaltol::Float64) =
    new(qval, stategrid, actiongrid, cputime, niter, finaltol)

end

function discretize_statevariable!(vi::ValueIteration, varname::String, step::Real)
  if haskey(vi.statemap, varname)
    warn(string(
      "state variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.statemap[varname] = LazyDiscrete(varname, float64(step))
end

function discretize_actionvariable!(vi::ValueIteration, varname::String, step::Real)
  if haskey(vi.actionmap, varname)
    warn(string(
      "action variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.actionmap[varname] = LazyDiscrete(varname, float64(step))
end

function getpolicy(mdp::MDP, solution::ValueIterationSolution)

  statedim = length(mdp.statemap)
  actiondim = length(mdp.actionmap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]
  nactions, nstates = size(solution.qval)

  function indexify(statevec::Vector)
    stateidxvec = zeros(statedim)
    for idim in 1:statedim
      statevar = mdp.statemap[stateargs[idim]]
      if isa(statevar, RangeVar)
        stateidxvec[idim] = statevec[idim]
      elseif isa(statevar, ValuesVar)
        stateidxvec[idim] = findfirst(statevar.values, statevec[idim])
      else
        error(string(
          "unknown state variable definition type for ", statevar))
      end
    end
    return stateidxvec
  end

  function policy(state...)

    statevec = [stateelem for stateelem in state]
    stateidxvec = indexify(statevec)
    stateidxs, wts = interpolants(solution.stategrid, stateidxvec)

    iaction_best = 0
    vaction_best = -Inf

    for iaction in 1:nactions
      vaction = 0.0
      for i in 1:length(stateidxs)
        vaction += wts[i] * solution.qval[iaction, stateidxs[i]]
      end

      if vaction > vaction_best
        iaction_best = iaction
        vaction_best = vaction
      end
    end

    rawaction = ind2x(solution.actiongrid, iaction_best)
    action = Array(Any, actiondim)

    for i in 1:actiondim
      actionvar = mdp.actionmap[actionargs[i]]
      if isa(actionvar, RangeVar)
        action[i] = rawaction[i]
      elseif isa(actionvar, ValuesVar)
        action[i] = actionvar.values[rawaction[i]]
      else
        error(string(
          "unknown action variable definition type for ", actionvar))
      end
    end

    return action

  end

  return policy

end

include("valueiteration_checks.jl")
include("valueiteration_solver.jl")
