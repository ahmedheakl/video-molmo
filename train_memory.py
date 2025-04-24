import os, glob
import random
from PIL import Image
from transformers import AutoTokenizer
from molmo_video.memory_preprocessor import MultiModalPreprocessor
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from functools import partial
from datasets import concatenate_datasets

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

model_id = "allenai/Molmo-7B-D-0924"

from molmo_video.memory import MolmoForCausalLM
from molmo_video.preprocessor import (FRAME_START_TOKEN, FRAME_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader

from trl import (
    SFTTrainer,
    SFTConfig,
)

from datasets import load_dataset

from utils import compute_mse_points, plot_metric, extract_caption, \
                    pil_to_np

num_frames = 4 #TODO: 

method = 'memory_mean'
base_data_dir = '/share/data/drive_1/heakl/data'
output_dir = f'/share/data/drive_1/heakl/models/{method}'
annotation_dir = "/share/data/drive_1/heakl/data/annotations" 
annotation_files = glob.glob(f"{annotation_dir}/*.jsonl") 

dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")

PROMPT_TEMPLATES = [
        "Point to {label}",
        "Point to {label}\nPlease say 'This isn't in the image.' if it is not in the image.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the image",
        "Point to any {label} in the image.",
        "Point: Where are {label}",
        "Show me where {label} are",
        "Can you show me where {label} are?",
        "Show me where {label} are",
        "Show me where {label} is",
        "Show me where {label} is.",
        "If there are any {label} in the image? Show me where they are.",
        "Where are {label}?",
        "Generate a list of points showing where {label} are.",
        "Find \"{label}\".",
        "Find \"{label}\".",
        "Locate all {label}.",
        "Locate {label}.",
        "Locate {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find {label}",
        "Find any {label}",
        "Point to {label}",
        "Look for {label} in the image and show me where they are.",
        "Help me find an object in the image by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the image?",
        "Can you see any {label} in the image? Point to them.",
        "Point out each {label} in the image.",
        "Point out every {label} in the image.",
        "Point to {label} in the image.",
        "Locate each {label} in the image.",
        "Can you point out all {label} in this image?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is {label} present, indicate its positions.",
        "show me all visible {label}",
    ]

def get_points_in_xml_format(points, caption):   
    for t, xy_list in points.items():
        xy_list = sorted(xy_list, key=lambda p: (p["x"], p["y"]))
        if len(xy_list) == 1:
            lines = ["<point"]
        else:
            lines = ["<points"] 
        line = ''
        for i, xy in enumerate(xy_list):
            if len(xy_list) == 1:  
                x, y = xy["x"], xy["y"]
                if x==-1.0 and y==-1.0:
                    return f' There are none.'
                line += f' x="{x:.1f}" y="{y:.1f}"'
            else:
                x, y = xy["x"], xy["y"]
                line += f' x{i+1}="{x:.1f}" y{i+1}="{y:.1f}"'
        lines.append(line)
        if len(xy_list) == 1:
            lines.append(f' alt="{caption}">{caption}</point>')
        else:
            lines.append(f' alt="{caption}">{caption}</points>') 
    output = "".join(lines)
    return output

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
            frame_path = os.path.join(base_data_dir, video_dir, frame_filename)
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
            frame_path = os.path.join(base_data_dir, video_dir, frame_filename)
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                image = None
            prev_frames.append(image.resize((w, h)))
        
        black = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
        prev_frames = [black for i in range(num_frames - len(prev_frames))] + prev_frames

        assert len(prev_frames) == num_frames, f"Expected {num_frames} frames, but got {len(prev_frames)}"

        new_batch["prev_frames"].append(prev_frames)

        selected_points = {int(k): points[k] for k in selected_frame_idxs[-1:]}
        new_batch["images"].append(images[-1:])
        new_batch["frame_idxs"].append(selected_frame_idxs)
        new_batch["points"].append(selected_points)
        caption = extract_caption(caption)
        question = np.random.choice(PROMPT_TEMPLATES).format(label=caption)
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

    batch_inputs = {}
    for key in batch_outputs[0]:
        batch_inputs[key] = [ex[key] for ex in batch_outputs]

    padded_inputs = tokenizer.pad({"input_ids": [ex[0] for ex in batch_inputs['input_tokens']],},
        return_tensors="pt",
    )
    
    labels = torch.cat([example for example in batch_inputs["target_tokens"]], 0)
    loss_masks = torch.cat([example for example in batch_inputs["loss_masks"]], 0)
    loss_masks = loss_masks * (loss_masks > 0)    
    labels.masked_fill_(~(loss_masks > 0), -100)

    padded_inputs['labels'] = labels
    padded_inputs["loss_masks"] = loss_masks
    padded_inputs["images"] = torch.cat([example for example in batch_inputs["images"]], 0)
    padded_inputs["prev_frames"] = torch.cat([example for example in batch_inputs["prev_frames"]], 0)
    padded_inputs["image_input_idx"] = torch.cat([example for example in batch_inputs["image_input_idx"]], 0)
    padded_inputs["position_ids"] = torch.cat([example for example in batch_inputs["position_ids"]], 0)

    return padded_inputs

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = MultiModalPreprocessor(tokenizer=tokenizer)

model = MolmoForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

nn.init.xavier_normal_(model.model.vision_backbone.adapter.wq.weight)
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wk.weight)
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wv.weight)
nn.init.xavier_normal_(model.model.vision_backbone.adapter.wo.weight)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable percentage {trainable_params * 100 / total_params:.2f}")

gradient_accumulation_steps = 256

training_args = SFTConfig(
    output_dir=f"{output_dir}/checkpoints",  
    num_train_epochs=4,  
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=False,  
    optim="paged_adamw_32bit",
    learning_rate=5e-6,
    lr_scheduler_type="cosine", 
    logging_steps=1, 
    eval_strategy= "no",
    save_strategy="steps",
    save_steps=200, 
    bf16=True, 
    tf32=True,  
    max_grad_norm=0.7, 
    warmup_ratio=0.03,  
    push_to_hub=False, 
    report_to='wandb',  
    dataset_text_field="", 
    dataset_kwargs={"skip_prepare_dataset": True},  
    dataloader_pin_memory=False,
    resume_from_checkpoint=True,
    ignore_data_skip=False
)

training_args.remove_unused_columns = False  

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=augmented_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

trainer.train()

trainer.save_model(output_dir)