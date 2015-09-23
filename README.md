# PLite.jl

// todo

## Problem definition

### MDP

An MDP is typically defined by the tuple (*S*, *A*, *T*, *R*), where

* *S* is the state space
* *A* is the action space
* *T*(*s*, *a*, *s*') is the transition function that gives the probability of reaching state *s*' from state s by taking action *a*
* *R*(*s*, *a*) is the reward function that gives a scalar value for taking action *a* at state *s*.

#### State space

// todo

#### Action space

// todo

#### Transition

// todo

#### Action

// todo

### POMDP

// todo

## Solver selection

### Value iteration

// todo

## Results

// todo

## Example

### MDP with *T*(*s*, *a*, *s*') type transition

```
using PLite

# constants
const MinX = 0
const MaxX = 100
const StepX = 2

const MinY = 0
const MaxY = 100
const StepY = 2

# mdp definition
mdp = MDP()

statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "y", MinY, MaxY)  # continuous
statevariable!(mdp, "goal", ["yes", "no"])  # discrete

actionvariable!(mdp, "move", ["N", "S", "E", "W", "stop"])  # discrete

transition!(mdp,
  ["x", "y", "goal", "move", "x", "y", "goal"],  # note |xp| is an "x" variable
  function mytransition(
      x::Float64,
      y::Float64,
      goal::String,
      move::String,
      xp::Float64,
      yp::Float64,
      goalp::String)

    if (move == "N" && xp - x == 0 && yp - y == 1) ||
      (move == "S" && xp - x == 0 && yp - y == -1) ||
      (move == "E" && xp - x == 1 && yp - y == 0) ||
      (move == "W" && xp - x == -1 && yp - y == 0) ||
      (move == "stop" && xp - x == 0 && yp - y == 0)

      prob = 0.6
    elseif abs(xp - x) <= 1 && abs(yp - y) <= 1
      prob = 0.1
    else
      return 0
    end

    if xp == MaxX && yp == MaxY && goalp == "yes"
      return prob
    else
      return 0
    end
  end
)

reward!(mdp,
  ["x", "y", "goal", "move"],
  function myreward(
      x::Float64,
      y::Float64,
      goal::String,
      move::String)

    if (goal == "yes")
      return 1
    else
      return 0
    end
  end
)

# solver options
solver = ParallelValueIteration()
discretize_statevariable!(solver, "x", StepX)
discretize_statevariable!(solver, "y", StepY)

# generate results
solve!(mdp, solver)
```

### MDP with *T*(*s*, *a*) type transition

```
using PLite

# constants
const MinX = 0
const MaxX = 100
const StepX = 2

# mdp definition
mdp = MDP()

statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "goal", ["yes", "no"])  # discrete

actionvariable!(mdp, "move", ["E", "W", "stop"])  # discrete

transition!(mdp,
  ["x", "goal", "move"],
  function mytransition(x::Float64, goal::String, move::String)
    function isgoal(x::Float64)
      if x == MaxX
        return "yes"
      else
        return "no"
      end
    end

    if goal == "yes"
      return [(x, goal, 1.0)]
    end

    if move == "E"
      return [
        ([x, goal], 0.2),
        ([x - 1, isgoal(x - 1)], 0.2),
        ([x + 1, isgoal(x + 1)], 0.6)
    elseif move == "W"
      return [
        ([x, goal], 0.2),
        ([x - 1, isgoal(x - 1)], 0.6),
        ([x + 1, isgoal(x + 1)], 0.2)]
    end
  end
)

reward!(mdp,
  ["x", "goal", "move"],
  function myreward(x::Float64, goal::String, move::String)
    if (goal == "yes")
      return 1
    else
      return 0
    end
  end
)

# solver options
solver = SerialValueIteration()
discretize_statevariable!(solver, "x", StepX)

# generate results
solve!(mdp, solver)
```

### POMDP example

// todo

## Under the hood

* lazy evaluation of state and action into internal type representations when `solver!` is called
* bound and set element validity checks for state and action
* automatically figures out which transition function type is given
* validity checks on transition function by reporting if probability returned is outside of [0,1] range
* bound checks on state and next state for *T*(*s*, *a*) type transition function
* checks that discretization of state and action variables is valid
