#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

max_or_min=min
dump_base_dir=dump
keyword_dataset=s2m

models="tskim-50 tskim"

for model in ${models}; do
    ./enh_kws.sh --dumpdir ${dump_base_dir} \
                --train_set train-${keyword_dataset}-${max_or_min} \
                --valid_set dev-${keyword_dataset}-${max_or_min} \
                --test_sets "test-${keyword_dataset}-${max_or_min} test-l2m-${max_or_min}" \
                --enh_config conf/tuning/tpdt-${model}.yaml \
                --enh_exp exp/${keyword_dataset}/"${model}_${max_or_min}" \
                --expdir exp/${keyword_dataset}/"${model}_${max_or_min}" \
                --kws_enh_task true \
                --ngpu 8 --num_nodes 1 \
                --use_noise_ref false --lang en \
                --fs 16k --audio_format wav \
                --stage 6 --stop_stage 6 "$@"; exit $?
done