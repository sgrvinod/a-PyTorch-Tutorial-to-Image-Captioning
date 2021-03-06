#!/bin/bash
INPUT_IMAGE="$1/$3/input_$4.png"
MASK_IMAGE="$1/$3/mask_$4.png"
OUTPUT_IMAGE="$2/$3/output_$4.png"
CHECKPOINT="logs/$5"
echo $1
echo $INPUT_IMAGE
echo $MASK_IMAGE
echo $OUTPUT_IMAGE

# cd ..

python test.py --image $INPUT_IMAGE --mask $MASK_IMAGE --output $OUTPUT_IMAGE --checkpoint_dir $CHECKPOINT
