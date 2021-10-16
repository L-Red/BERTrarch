#!/bin/bash
FRAMEWORK=$1
LABEL=$2
NUM_NODES=$3
echo $FRAMEWORK $LABEL
#env2lmod
module load gcc/8.2.0 python  eth_proxy
# bsub -n $NUM_NODES -W 10:00 -R "rusage[mem=256000,ngpus_excl_p=1,scratch=16000]" -R "select[gpu_model0==TITANRTX]"  python run_classification.py $FRAMEWORK $LABEL $NUM_NODES bert-svm
bsub -n $NUM_NODES -W 10:00 -R "rusage[mem=256000,ngpus_excl_p=4,scratch=16000]"  python run_classification.py $FRAMEWORK $LABEL $NUM_NODES bert
