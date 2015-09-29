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
const StepX = 20

# mdp definition
mdp = MDP()

statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "goal", ["yes", "no"])  # discrete

actionvariable!(mdp, "move", ["W", "E", "stop"])  # discrete

transition!(mdp,
  ["x", "goal", "move", "x", "goal"],  # note |xp| is an "x" variable
                                       # note (s,a,s') order
  function mytransition(
      x::Float64,
      goal::String,
      move::String,
      xp::Float64,
      goalp::String)

    function internaltransition(x::Float64, goal::String, move::String)
      function isgoal(x::Float64)
        if abs(x - MaxX / 2) < StepX
          return "yes"
        else
          return "no"
        end
      end

      if goal == "yes"
        return [([x, goal], 1.0)]
      end

      if move == "E"
        return [
          ([x, goal], 0.2),
          ([x - StepX, isgoal(x - StepX)], 0.2),
          ([x + StepX, isgoal(x + StepX)], 0.6)]
      elseif move == "W"
        return [
          ([x, goal], 0.2),
          ([x - StepX, isgoal(x - StepX)], 0.6),
          ([x + StepX, isgoal(x + StepX)], 0.2)]
      elseif move == "stop"
        return [
          ([x, goal], 0.6),
          ([x - StepX, isgoal(x - StepX)], 0.2),
          ([x + StepX, isgoal(x + StepX)], 0.2)]
      end
    end

    statepprobs = internaltransition(x, goal, move)
    for statepprob in statepprobs
      if xp == statepprob[1][1] && goalp == statepprob[1][2]
        return statepprob[2]
      end
    end
    return 0

  end
)

reward!(mdp,
  ["x", "goal", "move"],  # note (s,a) order
                          # note consistency of variables order with transition
  function myreward(x::Float64, goal::String, move::String)
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

# generate results
solve!(mdp, solver)
```

### MDP with *T*(*s*, *a*) type transition

```
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
  function mytransition(x::Float64, goal::String, move::String)
    function isgoal(x::Float64)
      if abs(x - MaxX / 2) < StepX
        return "yes"
      else
        return "no"
      end
    end

    if goal == "yes"
      return [([x, goal], 1.0)]
    end

    if move == "E"
      return [
        ([x, goal], 0.2),
        ([x - StepX, isgoal(x - StepX)], 0.2),
        ([x + StepX, isgoal(x + StepX)], 0.6)]
    elseif move == "W"
      return [
        ([x, goal], 0.2),
        ([x - StepX, isgoal(x - StepX)], 0.6),
        ([x + StepX, isgoal(x + StepX)], 0.2)]
    elseif move == "stop"
      return [
        ([x, goal], 0.6),
        ([x - StepX, isgoal(x - StepX)], 0.2),
        ([x + StepX, isgoal(x + StepX)], 0.2)]
    end
  end
)

reward!(mdp,
  ["x", "goal", "move"],
  function myreward(x::Float64, goal::String, move::String)
    if goal == "yes" && move == "stop"
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
