using GridInterpolations

export
  SerialValueIteration,
  ParallelValueIteration,
  discretize_statevariable!

const MaxIter = 1000
const Tol = 1e-4

abstract type ValueIteration <: Solver

type SerialValueIteration <: ValueIteration

  maxiter::Int64
  tol::Float64
  verbose::Bool

  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  SerialValueIteration(
      maxiter::Int64=MaxIter,
      tol::Float64=Tol,
      verbose::Bool=false) =
    new(
      maxiter,
      tol,
      verbose,
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

end

type ParallelValueIteration <: ValueIteration

  maxiter::Int64
  tol::Float64
  verbose::Bool

  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  ParallelValueIteration(
      maxiter::Int64=MaxIter,
      tol::Float64=Tol,
      verbose::Bool=false) =
    new(
      maxiter,
      tol,
      verbose,
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

end

type LazyDiscrete

  varname::String
  step::Float64

  LazyDiscrete(varname::String, step::Float64) = new(varname, step)

end

type ValueIterationSolution <: Solution

  qval::Matrix{Float64}
  cputime::Float64
  niter::Int64
  finaltol::Float64

  ValueIterationSolution(
      qval::Matrix{Float64},
      cputime::Float64,
      niter::Int64,
      finaltol::Float64) =
    new(qval, cputime, niter, finaltol)

end

function discretize_statevariable!(vi::ValueIteration, varname::String, step::Float64)
  if haskey(vi.statemap, varname)
    warn(string(
      "state variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.statemap[varname] = LazyDiscrete(varname, step)
end

function discretize_actionvariable!(vi::ValueIteration, varname::String, step::Float64)
  if haskey(vi.actionmap, varname)
    warn(string(
      "action variable ", varname, " already discretized in ValueIteration object, ",
      "replacing existing discretization scheme"))
  end
  vi.actionmap[varname] = LazyDiscrete(varname, step)
end

include("valueiteration_checks.jl")
include("valueiteration_solver.jl")
