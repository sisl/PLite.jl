using GridInterpolations

export
  SerialValueIteration,
  ParallelValueIteration,
  discretize_statevariable!

const MaxIter = 1000
const Tol = 1e-4
const Discount = 0.99

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

  SerialValueIteration(
      verbose::Bool=true,
      maxiter::Int64=MaxIter,
      tol::Float64=Tol,
      discount::Float64=Discount) =
    new(
      verbose,
      maxiter,
      tol,
      discount,
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

end

type ParallelValueIteration <: ValueIteration

  verbose::Bool

  maxiter::Int64
  tol::Float64
  discount::Float64

  statemap::Dict{String, LazyDiscrete}
  actionmap::Dict{String, LazyDiscrete}
  stategrid::RectangleGrid
  actiongrid::RectangleGrid

  ParallelValueIteration(
      verbose::Bool=true,
      maxiter::Int64=MaxIter,
      tol::Float64=Tol,
      discount::Float64=Discount) =
    new(
      verbose,
      maxiter,
      tol,
      discount,
      Dict{String, LazyDiscrete}(),
      Dict{String, LazyDiscrete}(),
      RectangleGrid(),
      RectangleGrid())

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

include("valueiteration_checks.jl")
include("valueiteration_solver.jl")
