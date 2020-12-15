#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --job-name=rwc
#SBATCH --output=/work/netthinker/ayush/out/rwc.out

export PYTHONPATH=$WORK/tf-gpu-pkgs
module load singularity

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export NCCL_SOCKET_IFNAME=^docker0,lo
# -------------------------

# random port between 12k and 20k
export MASTER_PORT=$((12000 + RANDOM % 20000))

singularity exec docker://lordvoldemort28/pytorch-opencv:dev python -u $@ --gpus=2 --nodes=1                                                                     
