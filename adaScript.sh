#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

module load u18/python/3.11.2

## remove venv if it exists
rm -rf env

## create the virtual environment
python3 -m venv env

## Create and acticate venv to run the code in.
source env/bin/activate

pip install -r requirements.txt

python test.ju.py
