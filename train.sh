#!/bin/sh

batch=${1}
batch=${batch:-2}

epoch=${2}
epoch=${epoch:-1000}

model=${3}
model=${model:-model}

participant=${4}
participant=${participant:-1}

timestamp=$(date +%s)

echo "Args: btach=${batch} epoch=${epoch} model=${model} participant=${participant}"
python fmri603d_srcgan_embeds.py $batch $epoch $model $participant > "${timestamp}.txt"
echo "output logged to ${timestamp}.txt"
