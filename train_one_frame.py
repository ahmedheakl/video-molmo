import os, re
import random
from PIL import Image
from transformers import AutoTokenizer
from molmo_video.preprocessor import MolmoProcessor
from molmo_video.model_preprocessor import MultiModalPreprocessor
import torch
import numpy as np
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

model_id = "allenai/Molmo-7B-D-0924"

from molmo_video.model import MolmoForCausalLM
from molmo_video.preprocessor import (FRAME_START_TOKEN, FRAME_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)
from transformers import BitsAndBytesConfig


from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    SFTConfig
)

from datasets import load_dataset
import glob
from peft import LoraConfig, get_peft_model

from utils import SYSTEM_PROMPT, plot_metric, extract_caption

num_frames = 1 #TODO: change to 8

base_data_dir = '/l/users/salman.khan/molmo/pointing_dataset'
output_dir = '/l/users/salman.khan/workspace_pointing_lmm/models/single_frame'
annotation_dir = "/l/users/salman.khan/workspace_pointing_lmm/datasets/annotations" 
annotation_files = glob.glob(f"{annotation_dir}/*.jsonl") 

dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")

PROMPT_TEMPLATES = [
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

def compute_mse_points(prediction, gt):
    pred_coords = get_coords(prediction)
    gt_coords = get_coords(gt)
    mse = 0.0
    for p, g in zip(pred_coords, gt_coords):
        mse += ((p[0] - g[0])**2 + (p[1] - g[1])**2)/2.0
    if len(gt_coords) != 0:
        mse /= len(gt_coords)
    return mse

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

def get_coords(output_string):
    coordinates = []

    def safe_float_conversion(value):
        try:
            return float(value)
        except ValueError:
            return None
        
    if 'points' in output_string:
        # Handle multiple coordinates
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        for _, x_val, _, y_val in matches:
            x = safe_float_conversion(x_val)
            y = safe_float_conversion(y_val)
            if x is not None and y is not None:  # Only append valid pairs
                coordinates.append((int(x), int(y)))
    else:
        # Handle single coordinate
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            x = safe_float_conversion(match.group(1))
            y = safe_float_conversion(match.group(2))
            if x is not None and y is not None:
                coordinates.append((int(x), int(y)))
    
    return coordinates


def random_augmentation(batch):
    new_batch = {"images": [], "frame_idxs": [], "points": [], "question": [], "answer": []}
    for i in range(len(batch["video"])):
        video_dir = batch["video"][i]
        frame_idxs = batch["frame_idxs"][i]    # e.g. [0, 1, 2, ...]
        points = batch["points"][i]
        caption = batch["caption"][i]
        if len(frame_idxs) <= num_frames:
            selected_indices = list(range(len(frame_idxs)))
        else:
            selected_indices = sorted(random.sample(range(len(frame_idxs)), num_frames))

        selected_frame_idxs = [frame_idxs[j] for j in selected_indices]
        selected_points = {int(k): points[k] for k in selected_frame_idxs}
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

        new_batch["images"].append(images)
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
        image_arrays = []
        for image in images:
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
                image_arrays.append(np.array(image))
        images = image_arrays
        if images:
            example_inputs = processor(
                images,
                messages,
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

# for name, param in model.named_parameters():
#     if "att_proj" in name or "ff_proj" in name or "image_projector" in name or "image_pooling_2d" in name:
#         param.requires_grad_(True)
#     else:
#         param.requires_grad_(False)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable percentage {trainable_params * 100 / total_params:.2f}")

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        labels = inputs.get("labels")
        mask = labels!=-100
        accuracy = (preds[mask].contiguous() == labels[mask].contiguous()).float().mean().item()
        predicted_text = self.processing_class.decode(preds[mask], skip_special_tokens=True)
        gt_text = self.processing_class.decode(labels[mask], skip_special_tokens=True)
        point_loss = compute_mse_points(predicted_text, gt_text)
        
        self.point_loss += point_loss        
        self.loss += loss.detach().item()
        self.accuracy += accuracy
        self.counter += 1
        if self.counter == self.args.gradient_accumulation_steps:
            avg_loss = self.loss / self.args.gradient_accumulation_steps
            avg_accuracy = self.accuracy / self.args.gradient_accumulation_steps
            avg_point_loss = self.point_loss / self.args.gradient_accumulation_steps
            print(f"Loss: {avg_loss}, Mean token accuracy: {avg_accuracy} Point Loss: {avg_point_loss}")
            print(f"pred text: {predicted_text}\n GT: {gt_text}")
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)
            self.point_loss_history.append(avg_point_loss)
            # Reset accumulators
            self.loss = 0.0
            self.accuracy = 0.0
            self.counter = 0
            self.point_loss = 0.0

        if self.state.global_step % 50 == 1:
            plot_metric(self.loss_history, "Loss", folder='single_frame_plots')
            plot_metric(self.accuracy_history, "Token Accuracy", folder='single_frame_plots')
            plot_metric(self.point_loss_history, "Point Loss", folder='single_frame_plots')

        return (loss, outputs) if return_outputs else loss

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
    logging_steps=10, 
    eval_strategy= "no",
    save_strategy="steps",
    save_steps=500, 
    bf16=True, 
    tf32=True,  
    max_grad_norm=0.7, 
    report_to='wandb',
    warmup_ratio=0.03,  
    push_to_hub=False, 
    dataset_text_field="", 
    dataset_kwargs={"skip_prepare_dataset": True},  
    dataloader_pin_memory=False,
)

training_args.remove_unused_columns = False  

trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=augmented_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model(output_dir)