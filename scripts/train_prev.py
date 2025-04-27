import os, glob, argparse, sys
import random
from PIL import Image
from transformers import AutoTokenizer
import torch, math
from torch import nn
import numpy as np
import torch.nn.functional as F

from datasets import load_dataset

from molmo_video.memory import MolmoForCausalLM
from molmo_video.memory_preprocessor import MultiModalPreprocessor
from trl import (
    SFTTrainer,
    SFTConfig
)

from utils import get_points_in_xml_format, compute_mse_points, plot_metric, extract_caption, \
                    pil_to_np, PROMPT_TEMPLATES

from optimizer import build_optimizer

parser = argparse.ArgumentParser(description="Training script for VideoMolmo")
parser.add_argument('--method', type=str, default='memory_mean')
parser.add_argument('--model_id', type=str, default="allenai/Molmo-7B-D-0924", help='Model version to train')
parser.add_argument('--num_frames', type=int, default=4, choices=[0, 1, 2, 3, 4], help='number of previous frames to use')
parser.add_argument('--base_data_dir', type=str, default='/l/users/salman.khan/molmo/pointing_dataset', help='path to the dataset directory')
parser.add_argument('--output_dir', type=str, default=f'/l/users/salman.khan/workspace_pointing_lmm/models', help='path to the output directory')
parser.add_argument('--annotation_dir', type=str, default="/l/users/salman.khan/workspace_pointing_lmm/datasets/annotations" , help='Path to the annotation directory')
parser.add_argument('--bfloat16', action='store_true', help='whether to use bfloat16 or not')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')

args = parser.parse_args()
args.output_dir = os.path.join(args.output_dir, args.method)
annotation_files = glob.glob(f"{args.annotation_dir}/*.jsonl") 

dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")

def random_augmentation(batch):
    new_batch = {"images": [], "frame_idxs": [], "points": [], "question": [], "answer": [], "prev_frames": []}
    for i in range(len(batch["video"])):
        video_dir = batch["video"][i]
        frame_idxs = batch["frame_idxs"][i]    # e.g. [0, 1, 2, ...]
        points = batch["points"][i]
        caption = batch["caption"][i]

        selected_indices = sorted(random.sample(range(len(frame_idxs)), 1))
        selected_frame_idxs = [frame_idxs[j] for j in selected_indices]
        images = []
        for j in selected_indices:
            frame_filename = f"{frame_idxs[j]:05d}.jpg"
            frame_path = os.path.join(args.base_data_dir, video_dir, frame_filename)
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                image = None
            images.append(image)
            
        w, h = images[-1].size
        idx = selected_indices[0]
        prev_frames = []
        for j in range(max(0, idx - 4), idx):
            frame_filename = f"{frame_idxs[j]:05d}.jpg"
            frame_path = os.path.join(args.base_data_dir, video_dir, frame_filename)
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                image = None
            prev_frames.append(image.resize((w, h)))
        
        black = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
        prev_frames = [black for i in range(args.num_frames - len(prev_frames))] + prev_frames

        assert len(prev_frames) == args.num_frames, f"Expected {args.num_frames} frames, but got {len(prev_frames)}"

        new_batch["prev_frames"].append(prev_frames)

        selected_points = {int(k): points[k] for k in selected_frame_idxs[-1:]}
        new_batch["images"].append(images[-1:])
        new_batch["frame_idxs"].append(selected_frame_idxs)
        new_batch["points"].append(selected_points)
        caption = extract_caption(caption)
        question = random.choice(PROMPT_TEMPLATES).format(label=caption)
        new_batch["question"].append(question)
        answer = get_points_in_xml_format(selected_points, caption)
        new_batch["answer"].append(answer)

    return new_batch

augmented_dataset = dataset.with_transform(random_augmentation)

def collate_fn(examples):
    batch_outputs = []
    for example in examples:
        question = example["question"]
        answer = example["answer"]
        images = example["images"]
        prev_frames = example['prev_frames']
        
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt = tokenizer.apply_chat_template(
            conversation,
            chat_template=tokenizer.chat_template,
            add_generation_prompt=False,
            tokenize=False,
            return_dict=False
        )
        split_text = prompt.split("Assistant:")
        messages = [" " + split_text[0].strip() + " Assistant:", " " + split_text[1].strip()]
        images = pil_to_np(images)
        prev_frames = pil_to_np(prev_frames)
        if images:
            example_inputs = processor(
                images,
                messages,
                prev_frames,
                is_training=True
            )
        else:
            example_inputs = processor(
                text=prompt,
                is_training=True
            )
        
        example_inputs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in example_inputs.items()}
        batch_outputs.append(example_inputs)
    
    # Now, collate the list of dictionaries into a single batch dictionary.
    # (Assuming each key in example_inputs is a tensor of the same shape across examples.)
    def _collate(tensors, max_sequence_length=None, dtype=None, pad_value=-1):
        max_len = max_sequence_length
        tensor = [x for x in tensors if x is not None][0]
        arr = np.full([len(tensors), max_len] + list(tensor.shape[1:]), pad_value,
                    dtype=dtype or tensor.dtype)

        for ix, tensor in enumerate(tensors):
            if tensor is not None:
                arr[ix, :len(tensor)] = tensor[:max_len]
        return torch.from_numpy(arr)

    padded_inputs = {}

    TEXT_KEYS = ["input_tokens", "loss_masks", "position_ids", "target_tokens", "position_ids"]
    IMAGE_KEYS = ["images", "image_input_idx", "prev_frames", "image_masks"]

    for key in TEXT_KEYS:
        dtype = np.float32 if key == "loss_masks" else np.int64
        if key == 'input_tokens':
            padded_inputs['input_ids'] = _collate([ex.get(key).squeeze(0) for ex in batch_outputs], max_sequence_length=1300, dtype=dtype)
        elif key == 'target_tokens':
            padded_inputs['labels'] = _collate([ex.get(key).squeeze(0) for ex in batch_outputs], max_sequence_length=1300, dtype=dtype)
        else:
            padded_inputs[key] = _collate([ex.get(key).squeeze(0) for ex in batch_outputs], max_sequence_length=1300, dtype=dtype)

    for key in IMAGE_KEYS:
        dtype = np.float32 if key == "loss_masks" else np.int64
        if key == "image_input_idx":
            padded_inputs[key] = _collate([ex.get(key).squeeze(0) for ex in batch_outputs], max_sequence_length=13, dtype=dtype)
        else:
            padded_inputs[key] = _collate([ex.get(key).squeeze(0) for ex in batch_outputs], max_sequence_length=13, dtype=np.float32)

    loss_masks = padded_inputs['loss_masks']*(padded_inputs['loss_masks'] > 0)
    padded_inputs['labels'].masked_fill_(~(loss_masks > 0), -100)

    return padded_inputs


tokenizer = AutoTokenizer.from_pretrained(args.model_id)
processor = MultiModalPreprocessor(tokenizer=tokenizer)

model = MolmoForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16 if args.bfloat16 else torch.float32,
    device_map='auto',
)

# initialize the memory module weights
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wq.weight)
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wk.weight)
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wv.weight)
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wo.weight)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable percentage {trainable_params * 100 / total_params:.2f}")


gradient_accumulation_steps = 256 // args.batch_size

ds_config = {
    "zero_optimization": {
      "stage": 3,
      "offload_param": {
        "device": "cpu"
      },
      "offload_optimizer": {
        "device": "cpu"
      }
    },
    "gradient_accumulation_steps": gradient_accumulation_steps,
    
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    
    "gradient_clipping": "auto",
    
    "bf16": {
      "enabled": True
    }
  }

training_args = SFTConfig(
    output_dir=f"{args.output_dir}/checkpoints",  
    num_train_epochs=4,  
    per_device_train_batch_size=args.batch_size,  
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=False,  
    optim="paged_adamw_32bit",
    learning_rate=5e-6,
    lr_scheduler_type="cosine", 
    logging_steps=1, 
    eval_strategy= "no",
    save_strategy="steps",
    save_steps=300, 
    bf16=args.bfloat16, 
    tf32=True,  
    max_grad_norm=1.0, 
    warmup_ratio=0.0067,  
    push_to_hub=False, 
    report_to='wandb',  
    dataset_text_field="", 
    dataset_kwargs={"skip_prepare_dataset": True},  
    dataloader_pin_memory=False,
    resume_from_checkpoint=False,
    ignore_data_skip=False,
    # deepspeed=ds_config,
)

training_args.remove_unused_columns = False  

per_step_batch = training_args.per_device_train_batch_size
grad_accum    = training_args.gradient_accumulation_steps
steps_per_epoch = math.ceil(len(augmented_dataset) / per_step_batch / grad_accum)
num_training_steps = steps_per_epoch * training_args.num_train_epochs
num_warmup_steps = int(training_args.warmup_ratio * num_training_steps)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=augmented_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
    optimizers=build_optimizer(model, num_warmup_steps, num_training_steps)
)

trainer.train()

trainer.save_model(args.output_dir)