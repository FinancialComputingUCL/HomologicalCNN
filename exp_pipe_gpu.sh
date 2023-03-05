#$ -l tmem=5G
#$ -l h_rt=48:00:00
#$ -l gpu=true
#$ -R y

#$ -cwd

#$ -S /bin/bash
#$ -j y
#$ -N LR_ONLINE_3

hostname

source /share/apps/source_files/python/python-3.9.5.source
source /SAN/fca/DRL_HFT_Investigations/HCNN/LOCAL_ENV/HEnv/bin/activate
source /share/apps/source_files/cuda/cuda-11.4.source

date
python3 -W ignore incremental_execution_cluster_rolling.py
date
