module ExampleModule

export
  mdp,
  solver,
  solve,
  getpolicy

using PLite

# constants
const MinX = 0
const MaxX = 100
const StepX = 20

# mdp definition
mdp = MDP()

statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "goal", ["no", "yes"])  # discrete

actionvariable!(mdp, "move", ["W", "E", "stop"])  # discrete

transition!(mdp,
  ["x", "goal", "move"],
  function mytransition(x::Float64, goal::AbstractString, move::AbstractString)
    function isgoal(x::Float64)
      if abs(x - MaxX / 2) < StepX
        return "yes"
      else
        return "no"
      end
    end

    if isgoal(x) == "yes" && goal == "yes"
      return [([x, isgoal(x)], 1.0)]
    end

    if move == "E"
      if x >= MaxX
        return [
          ([x, isgoal(x)], 0.9),
          ([x - StepX, isgoal(x - StepX)], 0.1)]
      elseif x <= MinX
        return [
          ([x, isgoal(x)], 0.2),
          ([x + StepX, isgoal(x + StepX)], 0.8)]
      else
        return [
          ([x, isgoal(x)], 0.1),
          ([x - StepX, isgoal(x - StepX)], 0.1),
          ([x + StepX, isgoal(x + StepX)], 0.8)]
      end
    elseif move == "W"
      if x >= MaxX
        return [
          ([x, isgoal(x)], 0.1),
          ([x - StepX, isgoal(x - StepX)], 0.9)]
      elseif x <= MinX
        return [
        ([x, isgoal(x)], 0.9),
        ([x + StepX, isgoal(x + StepX)], 0.1)]
      else
        return [
          ([x, isgoal(x)], 0.1),
          ([x - StepX, isgoal(x - StepX)], 0.8),
          ([x + StepX, isgoal(x + StepX)], 0.1)]
      end
    elseif move == "stop"
      return [([x, isgoal(x)], 1.0)]
    end
  end
)

reward!(mdp,
  ["x", "goal", "move"],
  function myreward(x::Float64, goal::AbstractString, move::AbstractString)
    if goal == "yes" && move == "stop"
      return 1
    else
      return 0
    end
  end
)

# solver options
solver = ParallelValueIteration()
discretize_statevariable!(solver, "x", StepX)

end