
### multi gpu ###
### ***NOTE***: if set --main_process_port=0 (according to accelerate documentation, it will automatically choose a port number. but it seems not well implemented in deepspeed, if set 0, process hangs). so we need to specify a port number manually.
TORCH_DISTRIBUTED_DEBUG=DETAIL ACCELERATE_DEBUG_VERBOSITY="debug" CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --main_process_port=29919 --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=4 --use_deepspeed Benchmark_comparison_with_LoRA.py 

### single gpu ###
# CUDA_VISIBLE_DEVICES="4" accelerate launch --main_process_port=29919 --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=1 --use_deepspeed finetune_distributed.py 
