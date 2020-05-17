#!/bin/sh

model=${1}
model=${model:-model}

echo "Args: model=${model}"
python test.py $model
