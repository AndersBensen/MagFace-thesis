#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J pipeline
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o logs/Output_arcface%J.out

#BSUB -e logs/Error_arcface%J.err

set -o errexit
set -o nounset

module load ninja/1.10.1
module load python3/3.7.11
module load cuda/11.2.2
module load cudnn/v8.1.1.33-prod-cuda-11.2
# module load opencv/3.4.16-python-3.7.11-cuda-11.1
# module load cmake/3.20.1

#DATASET_NAME='4k_full_pipeline'
DATASET_NAME='200_full_pipeline'
BASE_PATH='/work3/s210521'

echo "Running script..."
echo "Extraction features from database ........"

# # Make and activate venv
python3 -m venv /work3/s210521/venvs/arcface_venv_deepface
source /work3/s210521/venvs/arcface_venv_deepface/bin/activate

# pip3 install tensorflow-gpu
# pip3 install -r recognition/arcface_torch/requirement_deepface.txt
# pip3 install retina-face

# cd detection/retinaface/
# make 
# cd ../..
cd recognition/arcface_torch
python3 detect_recognize.py \
    --recognizer_path "/zhome/ab/6/160488/github/insightface-torch-thesis/recognition/arcface_torch/models/ms1mv3_arcface_r100_fp16/backbone.pth" \
    --db_root "$BASE_PATH/$DATASET_NAME" \
    --folders "accepted,age_group_4,age_group_3,age_group_2,age_group_1,age_group_0" \
    --subfolders "references,probes"

echo "Finished"