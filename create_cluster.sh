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


# create cluster with required packages
gcloud dataproc clusters create my-cluster \
    --image-version=preview-ubuntu18 \
    --worker-machine-type n1-standard-2 \
    --master-machine-type n1-standard-2 \
    --master-boot-disk-size=200GB \
    --worker-boot-disk-size=200GB \
    --num-workers 2 \
    --region=${REGION} \
    --metadata='PIP_PACKAGES=pandas boto3 pyspark findspark matplotlib numpy networkx' \
    --initialization-actions gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh

