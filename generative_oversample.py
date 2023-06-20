#!/usr/bin/bash
#SBATCH --job-name=oversample
#SBATCH --output=/home/users/%u/out/oversample.%j.out
#SBATCH --error=/home/users/%u/err/oversample.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

module load python/3.9.0
pip3 install transformers
pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
python3 greekOversample.py
