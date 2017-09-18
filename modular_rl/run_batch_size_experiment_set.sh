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
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_defaultbs1024 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=977 --timesteps_per_batch=1024 --env=${env} --filter=1 &> ${env}_default_bs1024.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_default2048 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=488 --timesteps_per_batch=2048 --env=${env} --filter=1 &> ${env}_defaul2048t.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_default4096 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=244 --timesteps_per_batch=4096 --env=${env} --filter=1 &> ${env}_defaul4096t.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_default8192 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=122 --timesteps_per_batch=4096 --env=${env} --filter=1 &> ${env}_default8192.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_default16384 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=61 --timesteps_per_batch=16384 --env=${env} --filter=1 &> ${env}_default16384.log 
KERAS_BACKEND=theano bash run_multiple.sh 5 1 ${env}_tanh_default32768 run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=30 --timesteps_per_batch=32768 --env=${env} --filter=1 &> ${env}_default32768.log 
