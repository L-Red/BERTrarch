#!/bin/bash\
#
# if [ -z "$1" ]; then
#   echo "You need to specify a model"
#   exit
# fi
#
bsub -o output_glove -n 16 -W 8:00 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
python /cluster/work/cotterell/liaroth/sub1.py
ENDBSUB
