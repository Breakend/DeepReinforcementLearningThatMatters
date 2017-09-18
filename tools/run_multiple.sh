#!/bin/bash


die() { echo "$@" 1>&2 ; exit 1; }

# TODO: make this proper usage script

if [  $# -le 4 ]
then
  die "Usage: bash $0 num_experiments num_experiments_in_parallel run_script [--all --other --args --to --run-script]"
fi

# check whether user had supplied -h or --help . If yes display usage
if [[ ( $# == "--help") ||  $# == "-h" ]]
then
  die "Usage: bash $0 num_experiments num_experiments_in_parallel run_script [--all --other --args --to --run_script]"
fi

num_experiments=$1
parallel_exps=$2
log_prefix=$3
run_script=$4

pickle_files=()

mkdir -p ./$log_prefix/

trap 'jobs -p | xargs kill' EXIT

for (( c=1; c<=$num_experiments; ))
do
  for (( j=1; j<=$parallel_exps; j++ ))
  do
    echo "Launching experiment $c"
    mkdir -p ./$log_prefix/exp_$c/
    python3 $run_script --seed $c --log_dir ./$log_prefix/exp_$c/ "${@:5}" &> ./$log_prefix/exp_$c.log &
    #pickle_files=("${pickle_files[@]}" "exp_$c.pickle")
    c=$((c+1))
  done
  wait
done

#python create_graphs_from_pickle.py "${pickle_files[@]}"

