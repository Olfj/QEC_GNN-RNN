#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525	 	# Project
#SBATCH -p alvis 				# Partition
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -o logs_alvis/logs_%j.out
##SBATCH --gpus-per-node=A100fat:1 # GPUs 256GB of RAM; cost factor 2.2
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0


# Load modules using pre-installed packages from the Alvis module tree
module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate

# send script
python examples/train_nn.py --d "$1" --t "$2" --dt "$3" --batch_size "$4" --n_batches "$5" --n_epochs "$6" --p "$7" \
    --label_mode "${8:-last}" ${9:+--note "$9"} ${10:+--load_path "${10}"} ${11:+--wandb --wandb_project "${11}"}
# sbatch run_training.sh  d  t  dt  batch  nbatch  epochs  p  [mode]  [note]  [load_path]  [wandb_project]
# sbatch run_training.sh  3  49  2  2048   256     200     0.001
# sbatch run_training.sh  3  49  2  2048   256     200     0.001  mpp
# sbatch run_training.sh  3  49  2  2048   256     200     0.001  mpp  baseline
# sbatch run_training.sh  3  49  2  2048   256     200     0.001  mpp  finetune  my_model
# sbatch run_training.sh  3  49  2  2048   256     200     0.001  mpp  baseline  ""  GNN-RNN-mpp