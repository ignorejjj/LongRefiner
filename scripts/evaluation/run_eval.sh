#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set datasets
DATASET_NAME="nq"
SPLIT="test"

# set generator model path
GENERATOR_MODEL="llama3.1-8B-instruct"
GENERATOR_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# set refiner model path
BASE_REFINER_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
QUERY_ANALYSIS_MODULE="jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct"
DOC_STRUCTURING_MODULE="jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct"
GLOBAL_SELECTION_MODULE="jinjiajie/Global-Selection-Qwen2.5-3B-Instruct"
SCORE_MODEL="bge-reranker-v2-m3"
SCORE_MODEL_PATH="BAAI/bge-reranker-v2-m3"

# set save directory
SAVE_DIR="results/"

# set retrieval result
RETRIEVAL_RESULT="sample_docs.json"

# run script
python scripts/evaluation/run_eval.py \
    --dataset_name ${DATASET_NAME} \
    --split ${SPLIT} \
    --generator_model ${GENERATOR_MODEL} \
    --generator_model_path ${GENERATOR_MODEL_PATH} \
    --base_refiner_model_path ${BASE_REFINER_MODEL_PATH} \
    --query_analysis_module_lora_path ${QUERY_ANALYSIS_MODULE} \
    --doc_structuring_module_lora_path ${DOC_STRUCTURING_MODULE} \
    --global_selection_module_lora_path ${GLOBAL_SELECTION_MODULE} \
    --score_model_name ${SCORE_MODEL} \
    --score_model_path ${SCORE_MODEL_PATH} \
    --gpu_id "0,1,2,3,4,5,6,7" \
    --save_dir ${SAVE_DIR} \
    --retrieval_result_path ${RETRIEVAL_RESULT} \
    --framework "vllm" \
    --gpu_memory_utilization 0.85 \
    --generator_max_input_len 15000 \
    --max_tokens 512 \
    --test_sample_num 1000 \
    --save_note "test_experiment"
