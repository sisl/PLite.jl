# PLite.jl

PLite, pronounced "polite,"<sup>[1](#myfootnote1)</sup> is a Julia-based modeling system for Markov decision processes. PLite turns Julia into a modeling language, allowing the stochastic problem to be specified using standard Julia expression syntax.

## Problem definition

### MDP

In a Markov decision process (MDP), an agent chooses action *a* based on observing state *s*. The agent then receives a reward *r*. The state evolves stochastically based on the current state and action taken by the agent. Note that the next state depends only on the current state and action, and not on any prior state or action. This assumption is known as the Markov assumption.

An MDP is typically defined by the tuple (*S*, *A*, *T*, *R*), where

* *S* is the state space
* *A* is the action space
* *T*(*s*, *a*, *s*') is the transition function that gives the probability of reaching state *s*' from state s by taking action *a*
* *R*(*s*, *a*) is the reward function that gives a scalar value for taking action *a* at state *s*.

To define an MDP, simply type the following.

```julia
mdp = MDP()
```

#### State space

The state space *S* is the set of all states, and it can be continuous and thus infinite. PLite provides the ability to define states simply as a set of all states or in a factored representation.

To define *S* the latter way, simply define the range or the array of possible values for the state variable. For instance, to define range of values for the continuous state variable *x* from 0 to 100, we simply type the following.

```julia
statevariable!(mdp, "x", 0, 100)
```

Note that we stringified *x* as `"x"`. Internally, this syntax allows `mdp` to keep an internal representation of the variable.

At this point, we may be tempted to discretize the variable and use something like value iteration to solve the MDP. The reason we don't provide the discretization option is because discretization is an approximation technique that should be considered together with the solution method. Yes, you may decide to discretize it an input the discretized values as an array input (see below). But we emphasize that at this point, we're only worried about defining a mathematical formulation of the problem.

To define a set of discrete values for a discrete state variable *direction* that can take on the values *north*, *south*, *east*, and *west*, we simply type the following.

```julia
statevariable!(mdp, "direction", ["north", "south", "east", "weast"])
```

If you made a mistake, as we did above with the spelling of *west*, we can override the existing definition of the state variable *direction* by redefining it as follows.

```julia
statevariable!(mdp, "direction", ["north", "south", "east", "west"])
```

PLite will provide a warning whenever you redefine a previously named variable.

To define *S* as a set of all states, we can simply define a single discrete state variable and type the following.

```julia
statevariable!(mdp, "S", ["v1", "v2", "v3", "v4", "v5", "v6", "v7"])
```

Note that `"S"` is not a special keyword. It's just a state variable like any other, and you'll have to take care to treat it as a state variable in defining the transition and reward functions. The nice thing is PLite allows you to work with both factored and unfactored MDP state space representations.

#### Action space

The action space *A* is the set of all actions. Like *S*, it can be continuous and thus infinite. The definition of action variables follow that of state variables, which means that PLite allows both factored and unfactored action space definitions.

To define a continuous and a discrete action variable *bankangle* and *move*, respectively, we type

```julia
actionvariable!(mdp, "bankangle", -180, 180)
actionvariable!(mdp, "move", ["up", "down", "left", "right"])
```

#### Transition

PLite provides two ways to define a transition model. Depending on the problem, it may be easier to define the transition model one way or another.

##### *T*(*s*, *a*, *s*') type transition

The first method follows the standard definition that returns a probability value between 0 and 1. Suppose we have the following definition of an MDP.

```julia
# constants
const MinX = 0
const MaxX = 100

# mdp definition
mdp = MDP()

statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "goal", ["yes", "no"])  # discrete

actionvariable!(mdp, "move", ["W", "E", "stop"])  # discrete
```

We want to define a transition function that takes a state-action-next state triplet (*s*, *a*, *s*') and returns the probability of starting in state *s*, taking action *a*, and ending up in state *s*'. Internally, PLite needs to match the defined state and action variables with the corresponding arguments for the transition function. To do this, we need to pass in an array of the argument names in the order they are to be input to the defined transition function.

An example of a *T*(*s*, *a*, *s*') type transition function is as follows. Here, the state variables are named `"x"` and `"goal"`, and the action variable is named `"move"`. Note that although *s*' is a different variable from *s*, they share the variable names `"x"` and `"goal"`. So even though the `mytransition` function signature is

```julia
function mytransition(
      x::Float64,
      goal::String,
      move::String,
      xp::Float64,
      goalp::String)
```

the array of (ordered) argument names is `["x", "goal", "move", "x", "goal"]` rather than `["x", "goal", "move", "xp", "goalp"]`. Below is the full listing that defines the transition for `mdp`.

```julia
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
```

##### *T*(*s*, *a*) type transition

The second way to define a transition model is to take in a state-action pair and return the set of all possible next states with their corresponding probabilities. Again, we need to pass an array of argument names in the order the (*s*, *a*) pair is defined to the transition function. Below is the full listing that defines the transition this way. It is mathematically equivalent to the *T*(*s*, *a*, *s*') type transition defined above.

```julia
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
```

#### Reward

The reward function takes in a state-action pair (*s*, *a*) and returns a scalar value indicating the expected reward received when executing action *a* from state *s*. We assume that the reward function is a deterministic function of *s* and *a*.

The process of defining the reward function is similar to that for the *T*(*s*, *a*) type transition function. We need to pass in an ordered array of variable names for PLite's internal housekeeping.

```julia
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
```

## Solver selection

PLite is intended to provide several solvers for any MDP problem, ranging from classic dynamic programming methods such as value iteration, policy iteration, and policy evaluation to approximate, online, and direct policy search methods. Eventually, we'll include learning algorithms, including cooler ones like [deep reinforcement learning with double Q-learning](http://arxiv.org/abs/1509.06461) with support for [distributed computing](http://arxiv.org/abs/1507.04296).

Until then, we just have to play with good o' value iteration in both its serial and parallel flavors.

### Value iteration

PLite implements the value iteration algorithm for infinite horizon problems with discount, but we demonstrate how to hack the existing algorithm to solve finite horizon problems with no discount.

#### Infinite horizon with discount

To initialize the serial value iteration solver, simply type the following.

```julia
solver = SerialValueIteration()
```

In PLite, value iteration requires all variables to be discretized. In the above problem, we need to discretize `"x"`, so we write

```julia
const StepX = 20
discretize_statevariable!(solver, "x", StepX)
```

Note that the solver uses the `GridInterpolations.jl` package for multilinear interpolation to approximate the values between the discretized state variable values if the *T*(*s*, *a*) type transition is defined. In the *T*(*s*, *a*, *s*') type transition, PLite assumes that for any (*s*, *a*, *s*') triplet the transition function will return a valid probability. In this case, the user is assumed to have defined a consistent MDP problem and no approximation is done on the part of PLite.

In any case, to solve the problem, simply pass both `mdp` and `solver` to the `solve!` function.
```julia
solve!(mdp, solver)
```

To use the parallel value iteration algorithm, simply initialize `solver` to `ParallelValueIteration(NThreads)`, with the additional argument indicating how many processor cores you would like to use.

```julia
const NThreads = int(CPU_CORES / 2)
solver = ParallelValueIteration(nthreads=NThreads)
```

Note that `CPU_CORES` is a Julia standard library constant, and it defaults to the number of CPU cores in your system. But the number of cores given usually includes virtual cores (e.g., Intel processors), so we divide by two to obtain the number of physical cores. There isn't an issue with increasing the number of cores. But since we have the same number of cores doing the same number of work, there won't be an increase in efficiency. In fact, with greater number of threads there may be more overhead and runtime processes. As such, we recommend using as many threads as there are physical cores on the machine.

As in the serial solver, PLite needs a definition of the discretization scheme.

There are three parameters for both the `SerialValueIteration` and `ParallelValueIteration` solvers: `maxiter`, `tol`, and `discount`. As their names suggest, these parameters correspond to the maximum number of iterations the value iteration algorithm will run before timing out, the L-infinity tolerance for Q-value convergence between iterations, and the discount factor, respectively.

The default parameters for both solvers are

* `maxiter = 1000`
* `tol = 1e-4`
* `discount = 0.99`
* `verbose = true`.

As suggested above, `ParallelValueIteration` has an additional `nthreads` keyword argument. The default value is `CPU_CORES / 2`.

To change these parameters, we use keyword arguments when instantiating the solver object. For instance, we can define the following.

```julia
solver = SerialValueIteration(
  tol=1e-6,
  maxiter=10000,
  discount=0.999,
  verbose=false)
```

In the case of the parallel solver, we can define

```julia
solver = ParallelValueIteration(
  tol=1e-6,
  maxiter=10000,
  discount=0.999,
  verbose=false,
  nthreads=10)
```

Note that because we're using keyword arguments (i.e., `keyword=value` type arguments), we can input the arguments in any way we want.

#### Finite horizon without discount

Notice that both the serial and parallel value iteration solvers are built with infinite horizon problems in mind. It's easy, however, to modify it to solve finite horizon problems by simply changing the parameters of the solvers.

For an MDP with a horizon of 40 and no discounting, we can define the solver as follows.

```julia
solver = SerialValueIteration(maxiter=40, discount=1)
```

#### Solution

The value iteration solution to the MDP generated by the solver can be extracted in the form of a policy function. The policy function takes as arguments the state variables in the same order defined for the transition and reward functions.

To obtain the value iteration policy for the `mdp` defined above, we call

```julia
policy = getpolicy(mdp)
```

If we want to query the optimal policy to take at the state `stateq = (12, "no")`, we can pass the query to the policy function as follows.

```julia
actionq = policy(stateq...)
```

Above, `actionq` takes on the value `"E"`. This action makes sense since we're to the west of the midpoint goal for the problem, and moving east would bring us closer to the goal.

## Example

### MDP with *T*(*s*, *a*, *s*') type transition

```julia
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

```julia
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

# Todo

## Short-term

* [ ] add support for parallel value iteration
* [ ] add support for pomdps (qmdp and fib)

## Medium-term
* [ ] link with pomdps.jl

<a name="myfootnote1">1</a>: because that's what Hao Yi, the author, aspires to be