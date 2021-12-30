#!/usr/bin/env bash

TEXT=data-bin/wmt17_en_de
fairseq-preprocess --source-lang en --target-lang de \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0  \
    --workers 20
