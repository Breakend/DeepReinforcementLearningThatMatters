# DeepReinforcementLearningThatMatters

Accompanying code for "Deep Reinforcement Learning that Matters"

## Baselines Experiments

<a href="https://github.com/Breakend/baselines"> Our Fork </a>

<a href="https://github.com/openai/baselines"> Current Baselines Code </a>

Our checkpointed version of the baselines code is found in the `baselines` folder. We make several modifications, mostly to allow for passing network structures as arguments to the MuJoCo-related run scripts.

Our only change internally was to the DDPG evaluation code. We do this to allow for comparison against other algorithms. In the DDPG code, evaluation is done across N different policies where N is the number of "epoch_cycles", we did not find this to be consistent for comparison against other methods, so we modify this to match the rllab version of DDPG evaluation. That is, we run on the target policy for 10 full trajectories at the end of an epoch.

## rllab experiments

<a href="https://github.com/rll/rllab"> rllab code </a>

These require the full rllab code, which we do not provide. Instead we provide some run scripts for rllab experiments in the `rllab` folder.

## rllabplusplus experiments

<a href="https://github.com/shaneshixiang/rllabplusplus/"> rllabplusplus (Q-Prop) code</a>

This is the code provided with QPROP, we only provide a checkpointed version of the DDPG code which we use for evaluation here. This is under the rllabplusplus folder.

## modular_rl experiments

<a href="https://github.com/joschu/modular_rl/"> Original TRPO (Modular RL) Code</a>

These are simply run scripts for the modular rl codebase.

## Tools

This contains tools for significance testing which we used. And various associated run scripts.

For bootstrap-based analysis, we use the <a href="https://github.com/facebookincubator/bootstrapped">bootstrapped repo</a>. Tutorials there are a nice introduction to this sort of statistical analysis. 

For t-test and KS test we use the <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html">scipy</a> <a href="https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html">tools</a>.
