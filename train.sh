#!/bin/sh

batch=${1}
batch=${batch:-2}

epoch=${2}
epoch=${epoch:-1000}

model=${3}
model=${model:-model}

timestamp=$(date +%s)

echo $python

echo "Args: btach=${batch} epoch=${epoch} model=${model}"
python fmri603d_srcgan_embeds.py $batch $epoch $model > "${timestamp}.txt"
echo "output logged to ${timestamp}.txt"
