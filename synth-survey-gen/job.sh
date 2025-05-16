#!bin/bash
#PBS -N synth_survey_job
#PBS -l nodes=1:ppn6
#PBS -l mem=64gb

module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

module load ollama
ollama serve > /dev/null 2>&1 &

python main.py configs/Chicago