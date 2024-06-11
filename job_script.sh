#$ -S /bin/bash
#$ -l tmem=2G
#$ -l h_rt=1:0:0
#$ -wd /home/users/ashcherb/masters_project/1_testing
#$ -j y
#$ -N Test gpu job
#$ -o test_job.log
#$ -l gpu=true
#$ -pe gpu 2 #
#$ -R y

hostname

date
sleep 20
date