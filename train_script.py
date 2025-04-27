# # %%
# %load_ext autoreload
# %autoreload 2

# %%
from transformers import AutoTokenizer
from molmo_video.preprocessor import MolmoProcessor

tokenizer = AutoTokenizer.from_pretrained("allenai/Molmo-7B-D-0924")


processor = MolmoProcessor(tokenizer=tokenizer, 
                           preprocessor_config_path="/share/users/shehan/workspace_pointing_lmm/MolmoVideo/molmo_video/hf_configs_Molmo-7B-D-0924/preprocessor_config.json")

# %%
from molmo_video.preprocessor import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)

# %%
SYSTEM_PROMPT = "You are given {num_frames} frames from a video. The frame indices are {selected_frame_idxs}. "
PROMPT_TEMPLATES = [
        "Point to {label}\nPlease say 'This isn't in the video.' if it is not in the video.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the video",
        "Point to any {label} in the video.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the video? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the video and show me where they are.",
        "Help me find an object in the video by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the video?",
        "Can you see any {label} in the video? Point to them.",
        "Point out each {label} in the video.",
        "Point out every {label} in the video.",
        "Point to the {label} in the video.",
        "Locate each {label} in the video.",
        "Can you point out all {label} in this video?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
    ]

# %%
from datasets import load_dataset

# Load the dataset from the JSONL file
dataset = load_dataset("json", 
                       data_files="/share/users/shehan/workspace_pointing_lmm/DataPrep/annotations_davis17_train.jsonl",
                       split="train")

# Print a sample to verify
print(dataset[0])


# %%
dataset




# %%
dataset[0]

# %%
dataset[0]['points']

# %%
def get_points_in_xml_format(points, caption):
    """
    Convert a list of (t, x, y) points to a string in XML format.
    """
    lines = ["<point"]
    # for t, x, y in points:
    for t, x_y in points.items():
        if x_y is None:
            continue
        x, y = x_y
        # Append each point's attributes on a new, indented line
        lines.append(f' t="{int(t)}" x="{x:.1f}" y="{y:.1f}"')
    # Append the alt attribute and the element's text content on the final line
    lines.append(f' alt="{caption}">{caption}</point>')
    formatted_output = "".join(lines)
    
    return formatted_output

# %%
import os
import random
from PIL import Image

base_data_dir = '/share/users/shehan/workspace_pointing_lmm/DataPrep'

num_frames = 4 #TODO: change to 4

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
      'timestamps' : a list of timestamps,
      'points'     : a list of (t, x, y) values.
    """
    new_batch = {"images": [], "frame_idxs": [], "timestamps": [], "points": [], "question": [], "answer": []}

    # Process each example in the batch
    for i in range(len(batch["video"])):
        video_dir = batch["video"][i]
        frame_idxs = batch["frame_idxs"][i]    # e.g. [0, 1, 2, ...]
        timestamps = batch["timestamps"][i]
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
        selected_timestamps = [timestamps[j] for j in selected_indices]
        # selected_points = [points[j] for j in selected_indices]
        # selected_points = [points[j] for j in selected_indices if j in points]
        # print('points', points)
        selected_points = {int(k): v for k, v in points.items() if int(k) in selected_frame_idxs}
        # print('selected_points', selected_points)
        

        # Load images corresponding to the selected frame indices
        images = []
        for j in selected_indices:
            # Construct file path: assuming images are named as "00000.jpg", "00001.jpg", etc.
            frame_filename = f"{frame_idxs[j]:05d}.jpg"
            frame_path = os.path.join(base_data_dir, video_dir, frame_filename)
            try:
                image = Image.open(frame_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")
                image = None
            images.append(image)

        # Add the selected data to the new batch
        new_batch["images"].append(images)
        new_batch["frame_idxs"].append(selected_frame_idxs)
        new_batch["timestamps"].append(selected_timestamps)
        new_batch["points"].append(selected_points)
        
        question = SYSTEM_PROMPT.format(num_frames=num_frames, selected_frame_idxs=selected_frame_idxs) + random.choice(PROMPT_TEMPLATES).format(label=caption)
        new_batch["question"].append(question)
        
        answer = get_points_in_xml_format(selected_points, caption)
        new_batch["answer"].append(answer)

    return new_batch

# Now, apply the transform to your dataset:
# dataset = your_huggingface_dataset.with_transform(random_augmentation)
augmented_dataset = dataset.with_transform(random_augmentation)


# %%


# %%
augmented_dataset

# %%
augmented_dataset[0:2]

# %%
import matplotlib.pyplot as plt

# %%
import torch

def collate_fn(examples):
    batch_outputs = []
    # Iterate over each example in the batch.
    # for question, answer, images in zip(examples["question"], examples["answer"], examples["image"]):
    
    # print('len(examples["question"])', len(examples["question"]))
    # print('examples["question"]', examples["question"])
    # print('examples["answer"]', examples["answer"])
    
    # print('len(examples)', len(examples))
    # print('examples', examples)
    
    # for n in range(len(examples["question"])):
    for example in examples:
        question = example["question"]
        answer = example["answer"]
        images = example["images"]
        
        # print('>>>>>>len(images)', len(images))
        # print('>>>>>>images', images[0].size) #dimensions
        
        # if image size is (640, 270), plot
        if images[0].size == (640, 270):
            plt.imshow(images[0])
        
        # question = examples["question"][n]
        # answer = examples["answer"][n]
        # images = examples["images"][n]
        
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
        
        # Move to cuda and add a batch dimension.
        example_inputs = {k: v.to("cuda").unsqueeze(0) for k, v in example_inputs.items()}
        
        batch_outputs.append(example_inputs)
    
    # Now, collate the list of dictionaries into a single batch dictionary.
    # (Assuming each key in example_inputs is a tensor of the same shape across examples.)
    batch_inputs = {}
    for key in batch_outputs[0]:
        # print('>>> key', key)
        # for bt in batch_outputs:
        #     print('bt[key].shape', bt[key].shape)
        batch_inputs[key] = [ex[key] for ex in batch_outputs] #torch.cat([ex[key] for ex in batch_outputs], dim=0)
    
    # return batch_inputs
    
    # batch_inputs has keys: input_ids', 'images', 'image_input_idx', 'image_masks', 'labels'
    
    # print('>> batch_inputs', batch_inputs)
    
    
    # padded_inputs = tokenizer.pad({"input_ids": [ex["input_ids"][0] for ex in batch_inputs],},
    #     padding=True, return_tensors="pt",
    # )
    
    padded_inputs = tokenizer.pad({"input_ids": [ex[0] for ex in batch_inputs['input_ids']],},
        # padding=True, 
        padding="max_length", max_length=1600,
        return_tensors="pt",
    )
    
    print('>>padded_inputs["input_ids"]', padded_inputs["input_ids"].shape)
    
    # Create the labels and set the special tokens to -100
    labels = padded_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Set the special tokens to -100
    labels[labels == processor.special_token_ids[IMAGE_PROMPT]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IM_START_TOKEN]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IM_END_TOKEN]] = -100
    labels[labels == processor.special_token_ids[DEFAULT_IM_COL_TOKEN]] = -100
    
    padded_inputs["labels"] = labels
    
    # padded_inputs["images"] = torch.cat([example["images"] for example in batch_inputs], 0)
    # padded_inputs["image_input_idx"] = torch.cat([example["image_input_idx"] for example in batch_inputs], 0)
    # padded_inputs["image_masks"] = torch.cat([example["image_masks"] for example in batch_inputs], 0)
    
    # for ex in batch_inputs['images']:
    #     print('ex.shape', ex.shape)
    
    padded_inputs["images"] = torch.cat([example for example in batch_inputs["images"]], 0)
    padded_inputs["image_input_idx"] = torch.cat([example for example in batch_inputs["image_input_idx"]], 0)
    padded_inputs["image_masks"] = torch.cat([example for example in batch_inputs["image_masks"]], 0)
    
    padded_inputs = padded_inputs.to(torch.bfloat16).to("cuda")
    
    return padded_inputs


# %% [markdown]
# # Model

# %%
import os; os.environ['LD_LIBRARY_PATH'] = "/share/softwares/cuda_cudnn/cuda-12.1/lib64:/lib/x86_64-linux-gnu:" + os.environ.get('LD_LIBRARY_PATH', '')


# %%

import os


# %%
import torch

from molmo_video.model import MolmoForCausalLM


# %%
model_id = "allenai/Molmo-7B-D-0924"

# %%
USE_QLORA = True

from transformers import BitsAndBytesConfig


# load the model
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = MolmoForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        # conda_map="auto",
    )
else:
    model = MolmoForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # device_map="auto",
    )

# %%
from peft import LoraConfig, get_peft_model

# Define LoRA configuration. By setting target_modules to ["att_proj", "ff_proj"],
# only the transformer (LLM) portion will have LoRA adapters.
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # for causal language modeling
    r=8,                    # LoRA rank (adjust as needed)
    lora_alpha=8,          # scaling factor
    lora_dropout=0.1,       # dropout probability for LoRA layers
    target_modules=["att_proj", "ff_proj"]
)

# Wrap the model with LoRA (this only affects modules matching the target names)
model = get_peft_model(model, lora_config)

# Optional: print out trainable parameters to verify that only the LLM part is modified.
model.print_trainable_parameters()


# %%
gradient_accumulation_steps = 10

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



# %%
from trl import SFTConfig

# Configure training arguments
training_args = SFTConfig(
    output_dir="molmo-video",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    # per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=gradient_accumulation_steps,  # Steps to accumulate gradients
    
    gradient_checkpointing=False,  # Enable gradient checkpointing for memory efficiency
    
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    
    eval_strategy= "no", # "steps",  # Strategy for evaluation
    save_strategy="no", #steps",  # Strategy for saving the model
    
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    #max_seq_length=1024  # Maximum sequence length for input
    
    deepspeed=ds_config,  # DeepSpeed configuration
    dataloader_pin_memory=False
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset

# %%
# from transformers import Trainer
# trainer = Trainer(
#         model=model,
#         train_dataset=augmented_dataset,
#         data_collator=collate_fn,
#         args=training_args,
#         )

# trainer.train()

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=augmented_dataset,
    data_collator=collate_fn,
    peft_config=lora_config,
    tokenizer=processor.tokenizer,
)


# %%
trainer.train()

# %%
trainer.save_model("path_to_save_model")



