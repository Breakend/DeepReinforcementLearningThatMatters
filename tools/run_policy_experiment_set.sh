#!/bin/bash


die() { echo "$@" 1>&2 ; exit 1; }

# TODO: make this proper usage script

# check whether user had supplied -h or --help . If yes display usage
if [[ ( $# == "--help") ||  $# == "-h" ]]
then
  die "Usage: bash $0 num_experiments num_experiments_in_parallel run_script [--all --other --args --to --run_script]"
fi

script=$1
env=$2


# default (6464tanh)
# activations
bash run_multiple.sh 5 1 ${env}_default $script ${env} &> ${env}default.log &
bash run_multiple.sh 5 1 ${env}_vfleakrelu $script ${env} --activation_vf leaky_relu &> ${env}vfleakrelu.log
bash run_multiple.sh 5 1 ${env}_vfrelu $script ${env} --activation_vf relu &> ${env}vfrelu.log &
bash run_multiple.sh 5 1 ${env}_policyrelu $script ${env} --activation_policy relu &> ${env}policyrelu.log &
bash run_multiple.sh 5 1 ${env}_policyleakyrelu $script ${env} --activation_policy leaky_relu &> ${env}policyleakyrelu.log

# 
bash run_multiple.sh 5 1 ${env}_400300 $script $env --policy_size 400 300 &> ${env}_400300tanh.log &
bash run_multiple.sh 5 1 ${env}_1005025 $script $env --policy_size 100 50 25 &> ${env}_1005025tanh.log &
bash run_multiple.sh 5 1 ${env}_vf1005025 $script $env --value_func_size 100 50 25 &> ${env}_vf1005025tanh.log &
bash run_multiple.sh 5 1 ${env}_vf400300 $script $env --value_func_size 400 300 &> ${env}_vf400300tanh.log 
