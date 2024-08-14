#!/bin/bash

cd "$(dirname "$0")" || { echo "Faild to move directory"; exit 1; }

cd .. || { echo "Faild to move to high directory"; exit 1; }

python test.py \
    --mode test \
    --config_file configs/noHue_0.50.5_large_re_4.yaml \
    --model /home/yoojinoh/Others/PR/data/outputs/
    --OUTPUT_DIR /home/yoojinoh/Others/PR/data/outputs &

wait

echo "All jobs are done."