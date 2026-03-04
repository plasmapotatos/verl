set -x

data_path=./data/simpleqa/augment/completion_train.parquet
save_path=./data/simpleqa/qwenvl25_basetest.parquet
model_path=/work/hdd/bbsg/twei2/rl/verl/outputs/simpleqa_sft_aug_direct/sft_lr1e-4_ep1_seed1/global_step_50/merged_hf_model

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
