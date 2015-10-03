function lazyCheck(mdp::MDP, mcts::MCTS)
  checkArgumentOrder(mdp)
  checkTransition(mdp)
  checkDiscretize(mdp, mcts)
  info(string("mdp and monte-carlo tree search solver passed basic checks"))
end

# Check that all |mdp| RangeVar variables have valid discretization schemes in |mcts|
function checkDiscretize(mdp::MDP, mcts::MCTS)

  for lazyvar in values(mdp.statemap)
    if isa(lazyvar, RangeVar)
      if !haskey(mcts.statemap, lazyvar.varname)
        error(string(
          "state variable ", lazyvar.varname,
          " does not have a discretization scheme"))
      elseif lazyvar.maxval - lazyvar.minval < mcts.statemap[lazyvar.varname].step
        error(string(
          "state variable ", lazyvar.varname,
          " has a discretization step size larger than than its range"))
      end
    end
  end

  for lazyvar in values(mdp.actionmap)
    if isa(lazyvar, RangeVar)
      if !haskey(mcts.actionmap, lazyvar.varname)
        error(string(
          "action variable ", lazyvar.varname,
          " does not have a discretization scheme"))
      elseif lazyvar.maxval - lazyvar.minval < mcts.actionmap[lazyvar.varname].step
        error(string(
          "action variable ", lazyvar.varname,
          " has a discretization step size larger than than its range"))
      end
    end
  end

end
