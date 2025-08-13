import torch
import json
import datetime
import os
import matplotlib.pyplot as plt

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial

from util.logutil import init_logger, get_logger

from accelerate import Accelerator, DeepSpeedPlugin

print("Init deepspeed plugin...")
# Create a DeepSpeedPlugin configuration object to customize DeepSpeed integration settings。
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,   # Enable ZeRO (Zero Redundancy Optimizer) stage 3 optimization
                    # ZeRO stages: 
                    # 0 - disabled
                    # 1 - optimizer state partitioning
                    # 2 - optimizer state + gradient partitioning
                    # 3 - optimizer state + gradient + parameter partitioning (most memory efficient)
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps before optimization
    zero3_save_16bit_model=True,    # Save models in 16-bit precision when using ZeRO stage 3
                                    # Reduces model checkpoint size by 50% while maintaining model quality
    offload_optimizer_device="cpu", # Offload optimizer computation to CPU to drastically reduce GPU memory usage
    offload_param_device="cpu"      # Offload model parameters to CPU to further decrease GPU memory consumption
)
print("Init deepspeed plugin done")
# Initialize the Hugging Face Accelerator with DeepSpeed integration
# Accelerator provides a unified interface for distributed training across various backends
# (TPU, multi-GPU, DeepSpeed, etc.) while maintaining compatibility with PyTorch code
print("Init accelerator...")
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
print("Init accelerator done")

'''
With the above configuration, when launching the script with below command:
$TORCH_DISTRIBUTED_DEBUG=DETAIL ACCELERATE_DEBUG_VERBOSITY="debug" CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --main_process_port=29919 --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=4 --use_deepspeed finetune_distributed.py 

The final DeepSpeed configuration required will be generated during the subsequent execution of accelerator.prepare(). The configuration details are as follows:

json = {
    "train_batch_size": 8, 
    "train_micro_batch_size_per_gpu": 1, 
    "gradient_accumulation_steps": 2, 
    "zero_optimization": {
        "stage": 3, 
        "offload_optimizer": {
            "device": "cpu", 
            "nvme_path": null
        }, 
        "offload_param": {
            "device": "cpu", 
            "nvme_path": null
        }, 
        "stage3_gather_16bit_weights_on_model_save": true
    }, 
    "gradient_clipping": 1.0, 
    "steps_per_print": inf, 
    "bf16": {
        "enabled": true
    }, 
    "fp16": {
        "enabled": false
    }, 
    "zero_allow_untested_optimizer": true
}
'''

'''
Attention: 
In DeepSpeed, fp16 and bf16 are generally indicative of mixed precision training. 
The half-precision is used for forward and backward computations, while fp32 is used for optimizer computation.
'''

device = accelerator.device
output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'

if accelerator.is_local_main_process:
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    logger = get_logger()


class ToyDataSet(Dataset): # for toy demo, for train_data/data.json
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]

    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def write_chat_template(processor, output_dir):
    '''
    ***Note**

    We should have not had this function, as normal processor.save_pretrained(output_dir) would save chat_template.json file.
    However, on 2024/09/05, I think a commit introduced a bug to "huggingface/transformers", which caused the chat_template.json file not to be saved. 
    See the below commit, src/transformers/processing_utils.py line 393, this commit avoided chat_template.json to be saved.
    https://github.com/huggingface/transformers/commit/43df47d8e78238021a4273746fc469336f948314#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356

    To walk around that bug, we need manually save the chat_template.json file.

    I hope this bug will be fixed soon and I can remove this function then.
    '''
    
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")


def setup_lora_config():
    """
    Setup LoRA configuration for the model.
    
    Returns:
        LoraConfig: LoRA configuration object
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha parameter
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none",
        use_rslora=False,
        use_dora=False,
    )
    
    return lora_config


def apply_lora_to_model(model):
    """
    Apply LoRA to the model.
    
    Args:
        model: The base model

    Returns:
        model: Model with LoRA applied
    """
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    return model
        
    


def plot_training_loss(loss_history, epoch_losses, output_dir, training_type="LoRA"):
    """
    Plot training loss history and save to file.
    
    Args:
        loss_history (list): List of loss values for each training step
        epoch_losses (list): List of average loss values for each epoch
        output_dir (str): Directory to save the plot
        training_type (str): Type of training ("LoRA" or "Full")
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss over all steps
    ax1.plot(loss_history, label=f'{training_type} Training Loss', color='blue', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{training_type} Training Loss Over All Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Average loss per epoch
    if epoch_losses:
        ax2.plot(range(1, len(epoch_losses) + 1), epoch_losses, 
                label=f'{training_type} Average Epoch Loss', color='red', marker='o', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Loss')
        ax2.set_title(f'{training_type} Average Training Loss Per Epoch')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{training_type.lower()}_loss_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save loss data to JSON for further analysis
    loss_data = {
        'training_type': training_type,
        'step_losses': loss_history,
        'epoch_losses': epoch_losses
    }
    json_path = os.path.join(output_dir, f'{training_type.lower()}_loss_data.json')
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)


def compare_training_methods(lora_loss_data, full_loss_data, output_dir):
    """
    Compare LoRA and full fine-tuning loss curves.
    
    Args:
        lora_loss_data (dict): LoRA training loss data
        full_loss_data (dict): Full fine-tuning loss data
        output_dir (str): Directory to save comparison plots
    """

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Step-wise loss comparison
    lora_steps = lora_loss_data.get('step_losses', [])
    full_steps = full_loss_data.get('step_losses', [])
    
    if lora_steps and full_steps:
        # Normalize to same length for fair comparison
        max_steps = max(len(lora_steps), len(full_steps))
        if len(lora_steps) < max_steps:
            lora_steps.extend([lora_steps[-1]] * (max_steps - len(lora_steps)))
        if len(full_steps) < max_steps:
            full_steps.extend([full_steps[-1]] * (max_steps - len(full_steps)))
        
        ax1.plot(lora_steps, label='LoRA Training', color='blue', alpha=0.7, linewidth=2)
        ax1.plot(full_steps, label='Full Fine-tuning', color='red', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('LoRA vs Full Fine-tuning Loss Comparison (Steps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Epoch-wise loss comparison
    lora_epochs = lora_loss_data.get('epoch_losses', [])
    full_epochs = full_loss_data.get('epoch_losses', [])
    
    if lora_epochs and full_epochs:
        # Normalize to same length for fair comparison
        max_epochs = max(len(lora_epochs), len(full_epochs))
        if len(lora_epochs) < max_epochs:
            lora_epochs.extend([lora_epochs[-1]] * (max_epochs - len(lora_epochs)))
        if len(full_epochs) < max_epochs:
            full_epochs.extend([full_epochs[-1]] * (max_epochs - len(full_epochs)))
        
        epochs_range = range(1, max_epochs + 1)
        ax2.plot(epochs_range, lora_epochs, label='LoRA Training', color='blue', marker='o', linewidth=2)
        ax2.plot(epochs_range, full_epochs, label='Full Fine-tuning', color='red', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('LoRA vs Full Fine-tuning Loss Comparison (Epochs)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    comparison_data = {
        'lora_loss_data': lora_loss_data,
        'full_loss_data': full_loss_data,
        'comparison_metrics': {
            'lora_final_loss': lora_steps[-1] if lora_steps else None,
            'full_final_loss': full_steps[-1] if full_steps else None,
            'lora_final_epoch_loss': lora_epochs[-1] if lora_epochs else None,
            'full_final_epoch_loss': full_epochs[-1] if full_epochs else None,
        }
    }
    
    comparison_json_path = os.path.join(output_dir, 'training_comparison.json')
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)

def train(use_lora=True):
    """
    Train the model with LoRA or full fine-tuning.
    
    Args:
        use_lora (bool): If True, use LoRA fine-tuning. If False, use full fine-tuning.
    """
    # Load the model on the available device(s)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="bfloat16"
    )

    # Apply LoRA to the model based on parameter
    if use_lora:
        if accelerator.is_local_main_process:
            logger.info("Applying LoRA to the model...")
        model = apply_lora_to_model(model)
    else:
        if accelerator.is_local_main_process:
            logger.info("Using full fine-tuning (LoRA disabled)")
        model = model  # No LoRA applied

    # Load processor. 
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28

    # **Note:** About padding_side parameter, it default value is "left", here we set it as "right".
    # For why, read below.
    # Typically, in training, when batch size of training dataloader is > 1, it is often we need pad shorter inputs to the same length.
    # To pad, we often add "padding_token_id" to the right side of shorter inputs to make them the same length and set 0 in attention_mask for those padding_token_id.
    # BTW, in batching inference, we must use "padding_side" left, as generation usually uses the last token of output list of tokens.
    # 
    # If you like to read more, here are more discussions about padding and padding side:
    # https://github.com/huggingface/transformers/pull/26572
    # https://github.com/pytorch/pytorch/issues/110213
    # transformers/models/qwen2_vl/modeling_qwen2_vl.py: causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=128*28*28, max_pixels=256*28*28, padding_side="right")
    train_loader = DataLoader(
        ToyDataSet("train_data/data.json"),
        batch_size=1,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    model.train()
    epochs = 10
    
    # Use different learning rate for LoRA vs full fine-tuning
    if use_lora:
        assert hasattr(model, 'peft_config'), "Model should have peft_config when using LoRA"
        # Higher learning rate for LoRA training
        lr = 2e-4
        if accelerator.is_local_main_process:
            logger.info(f"Using LoRA training with learning rate: {lr}")
    else:
        # Standard learning rate for full fine-tuning
        lr = 1e-5
        if accelerator.is_local_main_process:
            logger.info(f"Using full fine-tuning with learning rate: {lr}")
    
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # Initialize loss tracking
    loss_history = []
    epoch_losses = []
    
    for epoch in range(epochs):
        steps = 0
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        for batch in train_loader:
            steps += 1
            with accelerator.accumulate(model):
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                # If use deepseed,`accelerator.backward(loss)` is doing that automatically. Therefore, this function will not work. 
                # For detail, see https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py , DeepSpeedOptimizerWrapper.step is an "pass" function.
                optimizer.step() 
                # If use deepseed,`accelerator.backward(loss)` is doing that automatically. Therefore, this function will not work.
                # For detail, see https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py , DeepSpeedOptimizerWrapper.zero_grad is an "pass" function.
                optimizer.zero_grad() 
                
                if accelerator.is_local_main_process:
                    current_loss = loss.item()
                    loss_history.append(current_loss)
                    epoch_loss_sum += current_loss
                    epoch_steps += 1
                    logger.info(f"Batch {steps} of epoch {epoch + 1}/{epochs}, training loss : {current_loss:.10f}")
        
        # Calculate average loss for this epoch
        if accelerator.is_local_main_process and epoch_steps > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_steps
            epoch_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{epochs} completed. Average loss: {avg_epoch_loss:.10f}")

    # Synchronize all processes to ensure training completion before saving the model.
    accelerator.wait_for_everyone()
    # Unwrap the model from distributed training wrappers
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Save the model using HuggingFace's pretrained format
    if use_lora and hasattr(unwrapped_model, 'peft_config'):
        # Save LoRA model
        if accelerator.is_local_main_process:
            logger.info("Saving LoRA model...")
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
    else:
        # Save full fine-tuned model
        if accelerator.is_local_main_process:
            logger.info("Saving full fine-tuned model...")
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            max_shard_size="20GB",
            state_dict=accelerator.get_state_dict(model),
        )
    
    if accelerator.is_local_main_process:
        processor.save_pretrained(output_dir)
        write_chat_template(processor, output_dir)

    # Plot loss history
    if accelerator.is_local_main_process:
        training_type = "LoRA" if use_lora else "Full"
        plot_training_loss(loss_history, epoch_losses, output_dir, logger, training_type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL model with LoRA or full fine-tuning')
    parser.add_argument('--use_lora', type=bool, default=True,
                       help='Use LoRA (True) or full fine-tuning (False). Default: True')

    args = parser.parse_args()    
    train(use_lora=args.use_lora)

