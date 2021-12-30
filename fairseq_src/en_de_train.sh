#!/usr/bin/env bash
set -x
set -e
#---------------------------------------

src=en
tgt=de

DATA_PATH=data-bin/wmt17_en_de/
MODEL_PATH=results/en2de/standard/rdrop_0/
mkdir -p $MODEL_PATH
nvidia-smi

python -c "import torch; print(torch.__version__)"

export CUDA_VISIBLE_DEVICES=0

fairseq-train $DATA_PATH \
    --user-dir translation_rdrop_src \
    --task rdrop_translation \
    --arch transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings \
    --optimizer adam --lr 0.001 -s $src -t $tgt \
    --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
    --max-tokens 4096 \
    --update-freq 16 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0 \
    --criterion reg_label_smoothed_cross_entropy \
    --reg-alpha 5 \
    --seed 0 \
    --fp16 \
    --max-update 300000 --warmup-updates 6000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $MODEL_PATH | tee -a $MODEL_PATH/train.log \
