import os
import random
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from molmo_video.preprocessor import MolmoProcessor
import torch
import torch.nn.functional as F

from datasets import load_dataset
import glob

# model_id = 'allenai/MolmoE-1B-0924'
model_id = "allenai/Molmo-7B-D-0924"
tokenizer = AutoTokenizer.from_pretrained(model_id)

preprocessor_config_path = "/home/salman.khan/shehan/workspace_pointing_lmm/MolmoVideo/molmo_video/hf_configs_Molmo-7B-D-0924/preprocessor_config.json"
processor = MolmoProcessor(tokenizer=tokenizer, 
                           preprocessor_config_path=preprocessor_config_path)

from molmo_video.model import MolmoForCausalLM
from molmo_video.preprocessor import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT, IMAGE_SEPARATOR_TOKEN)
from transformers import BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding


from trl import SFTConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from peft import LoraConfig, get_peft_model

num_frames = 4 #TODO: make it work for 8
RESIZE_DIM = 336
max_sequence_length, max_crops = 500*num_frames, 2*num_frames #TODO: adjust them to fit into memory

base_data_dir = '/l/users/salman.khan/molmo/pointing_dataset'
output_dir = '/l/users/salman.khan/workspace_pointing_lmm/models/lora/connector'
annotation_dir = "/l/users/salman.khan/workspace_pointing_lmm/datasets/annotations" 
annotation_files = glob.glob(f"{annotation_dir}/*.jsonl")  # Adjust extension if needed

# annotation_files = annotation_files[1:2] # TODO remove this

# Load the dataset from the JSONL file
dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")

SYSTEM_PROMPT = "You are given {num_frames} frames from a video. The frame indices are {selected_frame_idxs}. "
PROMPT_TEMPLATES = [
    'Point to {label}\nPlease say "This isn\'t in the video." if it is not in the video.',
    'Point to all occurrences of {label}.',
    'Point to any {label} in the video.',
    'Point to any {label} in the video.',
    'Point: Where are the {label}.',
    'Show me where the {label} are.',
    'Can you show me where the {label} are?',
    'Show me where the {label} are',
    'Show me where a {label} is',
    'Show me where a {label} is.',
    'If there are any {label} in the video? Show me where they are.',
    'Where are the {label}?',
    'Generate a list of points showing where the {label} are.',
    'Find the {label}.',
    'Find a {label}.',
    'Locate all {label}.',
    'Locate an {label}.',
    'Locate a {label}.',
    'Locate every {label}.',
    'Locate {label}.',
    'Locate the {label}.',
    'Object: {label}\nInstruction: Point to the object.',
    'find {label}',
    'find {label}.',
    'Point to every {label}',
    'find any {label} in the picture',
    'Find the {label}',
    'Find any {label}',
    'Point to a {label}',
    'Point to an {label}',
    'Look for {label} in the video and show me where they are.',
    'Help me find an object in the video by pointing to them.\nObject: {label}.',
    'I am looking for {label}, where can they be found in the video?',
    'Can you see any {label} in the video? Point to them.',
    'Point out each {label} in the video.',
    'Point out every {label} in the video.',
    'Point to the {label} in the video.',
    'Locate each {label} in the video.',
    'Can you point out all {label} in this video?',
    'Please find {label} and show me where they are.',
    'If there are any {label} present, indicate their positions.',
    'If there is a {label} present, indicate its positions.',
    'show me all visible {label}.',
]

# <points/ t1, x1, y1, t2, x2, y2, t3, x3, y3, t4, x4, y4 alt=caption </point>
def get_points_in_xml_format(points, caption):
    """
    Convert a dictionary of {frame: [{x, y}]} points to a string in XML format.
    """
    lines = ["<points"]
    
    for t, xy_list in points.items():
        for xy in xy_list:  # Handle multiple points per frame
            x, y = xy["x"], xy["y"]
            if x == -1 and y == -1:
                lines.append(f' t="{int(t)}" There are none.')
            else:
                lines.append(f' t="{int(t)}" x="{x:.1f}" y="{y:.1f}"')

    # Append the alt attribute and the element's text content on the final line
    lines.append(f' alt="{caption}">{caption}</points>')
    output = "".join(lines)
    # print(f"Output: {output}")
    return output

def random_augmentation(batch):
    """
    Batch-wise random augmentation.
    For each example in the batch:
      - Randomly selects `num_frames` indices (or all if there are fewer).
      - Loads only the images corresponding to the selected frame indices.
      - Subsets the 'frame_idxs', 'timestamps', and 'points' to only those indices.
    Assumes that each example in the batch has keys:
      'video'      : the directory containing the frames,
      'frame_idxs' : a list of frame indices,
    #   'timestamps' : a list of timestamps,
      'points'     : a list of (t, x, y) values.
    """
    new_batch = {"images": [], "frame_idxs": [], "points": [], "question": [], "answer": []}

    # Process each example in the batch
    for i in range(len(batch["video"])):
        video_dir = batch["video"][i]
        frame_idxs = batch["frame_idxs"][i]    # e.g. [0, 1, 2, ...]
        # timestamps = batch["timestamps"][i]
        points = batch["points"][i]
        caption = batch["caption"][i]

        # Determine the indices to select
        if len(frame_idxs) <= num_frames:
            selected_indices = list(range(len(frame_idxs)))
        else:
            # Select a sorted list of unique random indices to maintain temporal order
            selected_indices = sorted(random.sample(range(len(frame_idxs)), num_frames))

        # Subset the metadata based on selected indices
        selected_frame_idxs = [frame_idxs[j] for j in selected_indices]
        selected_points = {int(k): points[k] for k in selected_frame_idxs}        

        # Load images corresponding to the selected frame indices
        images = []
        for j in selected_indices:
            # Construct file path: assuming images are named as "00000.jpg", "00001.jpg", etc.
            frame_filename = f"{frame_idxs[j]:05d}.jpg"
            frame_path = os.path.join(base_data_dir, video_dir, frame_filename)
            try:
                image = Image.open(frame_path).convert("RGB")
                image = image.resize((RESIZE_DIM, RESIZE_DIM))
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                image = None
            images.append(image)

        # Add the selected data to the new batch
        new_batch["images"].append(images)
        new_batch["frame_idxs"].append(selected_frame_idxs)
        # new_batch["timestamps"].append(selected_timestamps)
        new_batch["points"].append(selected_points)
        
        question = SYSTEM_PROMPT.format(num_frames=num_frames, selected_frame_idxs=selected_frame_idxs) + random.choice(PROMPT_TEMPLATES).format(label=caption)
        new_batch["question"].append(question)
        
        answer = get_points_in_xml_format(selected_points, caption)
        new_batch["answer"].append(answer)
    return new_batch


# Now, apply the transform to your dataset:
# dataset = your_huggingface_dataset.with_transform(random_augmentation)
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
        if images:
            example_inputs = processor.process(
                images=images,
                text=prompt,
                for_training=True
            )
        else:
            example_inputs = processor.process(
                text=prompt,
                for_training=True
            )

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
    TEXT_KEYS = ["input_ids", "loss_masks", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks", "image_input_idx",]

    for key in TEXT_KEYS:
        dtype = np.float32 if key == "loss_masks" else np.int64
        padded_inputs[key] = _collate([ex.get(key) for ex in batch_outputs], max_sequence_length, dtype)
    for key in IMAGE_KEYS:
        padded_inputs[key] = _collate([ex.get(key) for ex in batch_outputs], max_crops)
    
    labels = padded_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    # Set the special tokens to -100
    labels[labels == processor.special_token_ids[IMAGE_PROMPT]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IM_START_TOKEN]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IM_END_TOKEN]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IM_COL_TOKEN]] = -100
    labels[labels == processor.special_token_ids[IMAGE_SEPARATOR_TOKEN]] = -100

    # mask user tokens and image tokens to -100 as they don't contribute to the CE Loss
    loss_masks = padded_inputs["loss_masks"] * (padded_inputs["loss_masks"] > 0)
    loss_masks = loss_masks.to(labels.device)
    labels.masked_fill_(~(loss_masks > 0), -100)

    padded_inputs["labels"] = labels
    attention_mask = (padded_inputs['input_ids'] != tokenizer.pad_token_id).long()
    padded_inputs['attention_mask'] = attention_mask

    for key in padded_inputs:
        if key in ["input_ids", "labels", "attention_mask", "image_input_idx", "image_masks", "position_ids", "loss_masks"]:
            # Integer tensors stay as integers
            padded_inputs[key] = padded_inputs[key].to("cuda")
        else:
            padded_inputs[key] = padded_inputs[key].to(torch.bfloat16).to("cuda")

    return padded_inputs

gradient_accumulation_steps = 16

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
    output_dir=output_dir, 
    num_train_epochs=2, 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=gradient_accumulation_steps,  
    gradient_checkpointing=False, 
    optim="adamw_torch", 
    learning_rate=1e-4,  
    lr_scheduler_type="cosine",  
    logging_steps=10,  
    eval_strategy= "no", 
    save_strategy="steps", 
    save_steps=1000, 
    bf16=True, 
    tf32=True,  
    max_grad_norm=0.3,  
    warmup_ratio=0.03, 
    push_to_hub=False, 
    dataset_text_field="",  
    dataset_kwargs={"skip_prepare_dataset": True},
    dataloader_pin_memory=False,
    deepspeed=ds_config, 
)

training_args.remove_unused_columns = False 

model = MolmoForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",  
    r=8,                    
    lora_alpha=16,          
    lora_dropout=0.1,    
    target_modules=["attn_proj", "ff_proj"]
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=augmented_dataset,
    peft_config=peft_config,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

trainer.train()

trainer.save_model(output_dir)
