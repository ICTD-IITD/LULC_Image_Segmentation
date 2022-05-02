#!/bin/sh
### Set the job name (for your reference)

#PBS -q low
#PBS -N Unet
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends, don't change anything on the below line
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=4:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=10:00:00
#PBS -l software=PYTHON

# After job starts, must goto working directory.
# $PBS_O_WORKDIR is the directory from where the job is fired.
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

# module () {
#         eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
# }

module load compiler/intel/2020u4/intelpython3.7

#module load apps/anaconda/3EnvCreation
echo "Starting training"
python3 main_unet.py 
echo "Testing"
python3 test_unet.py