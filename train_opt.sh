#!/usr/bin/env bash
DEVICE=0
FFPP_FACES_DF=./processed/ffpp-df/faces_df.pkl
FFPP_FACES_DIR=./processed/ffpp-faces

# python3 train_entire_coatnet.py \
# --model_type CoatNetv2 \
# --traindb ff-c23-720-140-140 \
# --valdb ff-c23-720-140-140 \
# --ffpp_faces_df_path $FFPP_FACES_DF \
# --ffpp_faces_dir $FFPP_FACES_DIR \
# --face scale \
# --size 224 \
# --batch 32 \
# --lr 1e-5 \
# --valint 500 \
# --patience 10 \
# --maxiter 30000 \
# --seed 41 \
# --attention \
# --device $DEVICE

python3 train_binclass_mlp.py \
--net EfficientNetFIA \
--traindb ff-c23-720-140-140 \
--valdb ff-c23-720-140-140 \
--ffpp_faces_df_path $FFPP_FACES_DF \
--ffpp_faces_dir $FFPP_FACES_DIR \
--face scale \
--size 224 \
--batch 2 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE
