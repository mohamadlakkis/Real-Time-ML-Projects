#!/bin/bash

## Job name
#SBATCH --job-name=Model-Seq2Seq

## Account to charge
#SBATCH --account=mnl03


## Specify the partition/queue
#SBATCH --partition=normal 
  
## Number of nodes 
#SBATCH --nodes=1
  
## Number of tasks (cores) per node   
#SBATCH --ntasks=1

## Number of CPUs per task  
#SBATCH --cpus-per-task=16  # Adjust this to the number of cores you need    
    
## Memory per node (in MB) 
#SBATCH --mem=16000  # Adjust memory according to your parallel requirements

## Maximum execution time
#SBATCH --time=0-24:00:00 


#SBATCH --output=output.out     # Standard output and error log

## Email notification 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mnl03@mail.aub.edu

## Load Python module (if necessary)
module load python/3  # Adjust this to your cluster's Python version
venv_path="/home/mnl03/virt_TENSORFLOW"

## Create a new virtual environment ( don't need it I have already created it ) 
python3 -m venv $venv_path 

## Activate the virtual environment ( I have already created the virtual environment)
source $venv_path/bin/activate




## Execute the Python script (specify the number of cores via environment variable)
export JOBLIB_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python model_1.py

## Deactivate the virtual environment
deactivate
