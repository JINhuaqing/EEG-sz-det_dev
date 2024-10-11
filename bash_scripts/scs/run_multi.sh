#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --gres=gpu:1
##SBATCH --nodelist=anahita
#SBATCH --output=scs/logs/RUN_%x_%j.out
#SBATCH -J multi-w5-1000
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/EEG-sz-det_dev/bash_scripts/

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"
source /netopt/rhel7/versions/python/Anaconda3-edge/etc/profile.d/conda.sh
#module load SCS/anaconda/anaconda3
conda activate eeg-sz-det

#python -u ../python_scripts/EEG_sz_train_multi.py --lr 0.0001  --ntrain_batch 1000 --move_steps 1 5 10 --nepoch 50
python -u ../python_scripts/EEG_sz_train_multi.py --lr 0.0001  --ntrain_batch 1000 --aux_loss --move_steps 1 5 10 --aux_loss_weight 5 --nepoch 50



