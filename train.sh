#!/bin/bash
OUTDIR=lstm7
DUELING=0

# #rm $OUTDIR/train.log
# #nohup CUDA_VISIBLE_DEVICES="0,1" python3 -u src/train.py --outdir "./output_env1"  > output_env1/train.log 2>&1 | tail -F output_env1/train.log &
# #CUDA_VISIBLE_DEVICES="-1" python3 -u src2/train.py --outdir $OUTDIR --dueling $DUELING  >> $OUTDIR/train.log 2>&1 | tail -F $OUTDIR/train.log
# CUDA_VISIBLE_DEVICES="0" python3 -u src/train.py --outdir $OUTDIR --dueling $DUELING >> $OUTDIR/train.log 2>&1 &
# tail -F $OUTDIR/train.log
# #CUDA_VISIBLE_DEVICES="0" python3 src/train.py --outdir $OUTDIR --dueling $DUELING

##############################
if [ ! -d "$OUTDIR" ]; then
    # 如果不存在，则创建文件夹
    mkdir -p "$OUTDIR"
    echo " " > model.log
    echo "Folder created: $OUTDIR"
else
    echo "Folder already exists: $OUTDIR"
fi
nohup /home/liting/anaconda3/envs/dqn/bin/python -u src/train.py --outdir  $OUTDIR --dueling $DUELING  >> $OUTDIR/train.log 2>&1 &