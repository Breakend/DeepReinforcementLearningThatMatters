import argparse
import os.path as osp
import pickle

import tensorflow as tf

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc import ext
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.algos.ddpg import DDPG
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
import lasagne.nonlinearities as NL


from sandbox.rocky.tf.misc.tensor_utils import lrelu

parser = argparse.ArgumentParser()
parser.add_argument("env", help="The environment name from OpenAIGym environments")
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--log_dir", default="./data_ddpg/")
parser.add_argument("--reward_scale", default=1.0, type=float)
parser.add_argument("--use_ec2", action="store_true", help="Use your ec2 instances if configured")
parser.add_argument("--dont_terminate_machine", action="store_false", help="Whether to terminate your spot instance or not. Be careful.")
parser.add_argument("--policy_size", default=[100,50,25], type=int, nargs='*')
parser.add_argument("--policy_activation", default="relu", type=str)
parser.add_argument("--vf_size", default=[100,50,25], type=int, nargs='*')
parser.add_argument("--vf_activation", default="relu", type=str)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

stub(globals())
ext.set_seed(args.seed)

gymenv = GymEnv(args.env, force_reset=True, record_video=False, record_log=False)

env = normalize(gymenv)

activation_map = { "relu" : NL.rectify, "tanh" : NL.tanh, "leaky_relu" : NL.LeakyRectify}

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=args.policy_size,
    hidden_nonlinearity=activation_map[args.policy_activation],
)

es = OUStrategy(env_spec=env.spec)

qf = ContinuousMLPQFunction(env_spec=env.spec,
                            hidden_nonlinearity=activation_map[args.vf_activation],
                            hidden_sizes=args.vf_size,)

algo = DDPG(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    batch_size=128,
    max_path_length=env.horizon,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=args.num_epochs,
    discount=0.995,
    scale_reward=args.reward_scale,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4,
    plot=False
)


run_experiment_lite(
    algo.train(),
    log_dir=None if args.use_ec2 else args.log_dir,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_prefix="DDPG_" + args.env,
    seed=args.seed,
    mode="ec2" if args.use_ec2 else "local",
    plot=False,
    # dry=True,
    terminate_machine=args.dont_terminate_machine,
    added_project_directories=[osp.abspath(osp.join(osp.dirname(__file__), '.'))]
)
