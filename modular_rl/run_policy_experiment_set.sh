#!/bin/bash


die() { echo "$@" 1>&2 ; exit 1; }

# TODO: make this proper usage script

# check whether user had supplied -h or --help . If yes display usage
if [[ ( $# == "--help") ||  $# == "-h" ]]
then
  die "Usage: bash $0 num_experiments num_experiments_in_parallel run_script [--all --other --args --to --run_script]"
fi

env=$1


# default (6464tanh)
# activations
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_default run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=100 --timesteps_per_batch=20000 --env=${env} --filter=1 &> ${env}_default.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_vfrelu run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation_vf=relu --n_iter=100 --timesteps_per_batch=20000 --env=${env} --filter=1 &> ${env}_vfrelu.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_polrelu run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=relu --n_iter=100 --timesteps_per_batch=20000 --env=${env} --filter=1 &> ${env}_relu.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh1005025 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=100 --timesteps_per_batch=20000 --hid_sizes "100,50,25" --env=${env} --filter=1 &> ${env}_1005025.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_400300 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=100 --timesteps_per_batch=20000 --hid_sizes "400,300" --env=${env} --filter=1 &> ${env}_400300.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_400300vf run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=100 --timesteps_per_batch=20000 --hid_sizes_vf "400,300" --env=${env} --filter=1 &> ${env}_400300vf.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_1005025vf run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=100 --timesteps_per_batch=20000 --hid_sizes_vf "100,50,25" --env=${env} --filter=1 &> ${env}_1005025vf.log 
