#!/bin/bash

#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -p bigmem
#SBATCH -n 28
#SBATCH -A hpc_csbg22
#SBATCH -o /work/gsriva2/BindingAffinityPrediction/BindingDB/EC_Class_mol2vec_100_trees.out

# Accept filename from command-line argument

#python RandomForestRegressionRandomShuffle2.py
python RandomForestRegressionEnzymeClassCrossValidationMol2vec.py
exit 0;
