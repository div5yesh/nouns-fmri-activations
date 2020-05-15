#!/bin/sh

model=${3}
model=${model:-model}

echo "Args: model=${model}"
python fmri603d_srcgan_embeds.py $model