#!/bin/bash
FRAMEWORK=$1
LABEL=$2
NUM_NODES=$3
echo $FRAMEWORK $LABEL
#env2lmod
module load new gcc/6.3.0 python/3.7.1 eth_proxy
bsub -n $NUM_NODES -W 08:00 -R "rusage[mem=256000,ngpus_excl_p=4,scratch=16000]" -R "select[gpu_model0==GeForceRTX2080Ti]"  python run_classification.py $FRAMEWORK $LABEL $NUM_NODES
