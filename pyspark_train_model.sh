#!/bin/bash

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate my_env

# set project id
gcloud config set project "xenon-bivouac-298214"
gcloud config set account "jack.boylan25@mail.dcu.ie"

# set environment variables
export CLOUDSDK_PYTHON=python3 
export PYSPARK_PYTHON=python3
export PYTHONHASHSEED=0
export CLUSTER="my-cluster"
export REGION="europe-west2"
export BUCKET_NAME="ca4022-files"


# run pyspark job pointing to python 3 drivers
gcloud dataproc jobs submit pyspark train_model_sparkify.py \
    --cluster=${CLUSTER} \
    --region=${REGION} \
    --properties="spark.pyspark.python=python3.8,spark.pyspark.driver.python=python3.8" \
    -- gs://${BUCKET_NAME}/input/ gs://${BUCKET_NAME}/output/



