#!/bin/bash
#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#SBATCH --ntasks=30
#### Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --partition=gpu                    
#### Allocate 1 GPU resource for this job. 
#SBATCH --gres=gpu:teslav100:1   
#SBATCH --output=logs/EEG_%x.out
#SBATCH -J lay2_dis_10step_latent_intervalbds_nomodel


echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

python -u /data/rajlab1/user_data/jin/MyResearch/gTVDN-NN/python_scripts/run_dis.py

