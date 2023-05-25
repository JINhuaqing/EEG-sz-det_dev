#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long                    # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=R20-%x.out


echo "Running prime number generator program on $SLURM_CPUS_ON_NODE GPU cores"

python -u /data/rajlab1/user_data/jin/MyResearch/gTVDN-NN/python_scripts/EEG_net_test.py

