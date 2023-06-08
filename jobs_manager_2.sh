#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=48:00:00
#$ -R y
#$ -pe smp 8

#$ -cwd

#$ -S /bin/bash
#$ -j y
#$ -N HCNN
#$ -t 1-69

hostname
date

number=$SGE_TASK_ID
paramfile="/SAN/fca/DRL_HFT_Investigations/HCNN/jobs_2.txt"

source /share/apps/source_files/python/python-3.9.5.source
source /SAN/fca/DRL_HFT_Investigations/HCNN/LOCAL_ENV/HEnv/bin/activate

index="`sed -n ${number}p $paramfile | awk '{print $1}'`"
variable1="`sed -n ${number}p $paramfile | awk '{print $2}'`"
variable2="`sed -n ${number}p $paramfile | awk '{print $3}'`"
variable3="`sed -n ${number}p $paramfile | awk '{print $4}'`"


echo $variable1
echo $variable2
echo $variable3

date
python3 main.py --model $variable1 --dataset_id $variable2 --seed $variable3
date
