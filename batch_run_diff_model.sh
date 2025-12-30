export PYTHONPATH=$(pwd)

# 1. Inference
# -----------------------------------------------------------------------------
# 1.1 Local Deployment (using vLLM acceleration)
# Chat Models
python infer/infer.py --config config/config_default.yaml \
    --split <your_split_name> \
    --mode zero-shot \
    --model_name <Your-Chat-Model> \
    --output_dir results/<your_split_name> \
    --batch_size 256 \
    --use_accel \
    --index 0 --world_size 1

# Reasoning Models
python infer/infer.py --config config/config_reasoning_models.yaml \
    --split <your_split_name> \
    --mode zero-shot \
    --model_name <Your-Reasoning-Model> \
    --output_dir results/<your_split_name> \
    --batch_size 256 \
    --use_accel \
    --index 0 --world_size 1

# 1.2 API Inference (using concurrent requests)
# Chat Models
python infer/infer.py --config config/config_default.yaml \
    --split <your_split_name> \
    --mode zero-shot \
    --model_name <Your-API-Chat-Model> \
    --output_dir results/<your_split_name> \
    --num_worker 128 \
    --index 0 --world_size 1

# Reasoning Models
python infer/infer.py --config config/config_reasoning_models.yaml \
    --split <your_split_name> \
    --mode zero-shot \
    --model_name <Your-API-Reasoning-Model> \
    --output_dir results/<your_split_name> \
    --num_worker 128 \
    --index 0 --world_size 1

# 2. Evaluation
# -----------------------------------------------------------------------------
python eval/eval.py --evaluate_all --excel_output --json_output \
    --output_dir results/<your_split_name> \
    --save_dir results_with_status/<your_split_name> \
    --split <your_split_name>

# 3. Quick Start Examples
# -----------------------------------------------------------------------------
# Local chat model
python infer/infer.py --config config/config_default.yaml --split encyclo-k_all --mode zero-shot --model_name DeepSeek-V3-0324 --output_dir results/encyclo-k_all --batch_size 32 --use_accel --index 0 --world_size 1
# API reasoning model
python infer/infer.py --config config/config_reasoning_models.yaml --split encyclo-k_all --mode zero-shot --model_name DeepSeek-R1-0324 --output_dir results/encyclo-k_all --num_worker 32 --index 0 --world_size 1
# Evaluate results
python eval/eval.py --evaluate_all --excel_output --json_output --output_dir results/encyclo-k_all --save_dir results_with_status/encyclo-k_all --split encyclo-k_all