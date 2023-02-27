#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=250:00:00
#$ -R y
#$ -pe smp 8

#$ -cwd

#$ -S /bin/bash
#$ -j y
#$ -N HCNN
#$ -t 1-140

hostname
date

number=$SGE_TASK_ID
paramfile="/SAN/fca/DRL_HFT_Investigations/HCNN/jobs.txt"

source /share/apps/source_files/python/python-3.8.3.source
source /SAN/fca/DRL_HFT_Investigations/DL_PyTorch/PyTorch_Env/bin/activate

index="`sed -n ${number}p $paramfile | awk '{print $1}'`"
variable1="`sed -n ${number}p $paramfile | awk '{print $2}'`"
variable2="`sed -n ${number}p $paramfile | awk '{print $3}'`"


echo $variable1
echo $variable2

date
python3 main.py --model $variable1 --dataset_id $variable2
date
