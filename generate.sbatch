#!/usr/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=/home/users/%u/out_llm/generate.%j.out
#SBATCH --error=/home/users/%u/err_llm/generate.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

module load python/3.9.0
module load devel py-pandas/2.0.1_py39
pip3 install transformers
pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
python3 generate.py