import os
import random
from PIL import Image
from transformers import AutoTokenizer
from molmo_video.preprocessor import MolmoProcessor
import torch
import torch.nn.functional as F
from transformers import EvalPrediction

model_id = "allenai/Molmo-7B-D-0924"
tokenizer = AutoTokenizer.from_pretrained(model_id)

preprocessor_config_path = "/home/salman.khan/shehan/workspace_pointing_lmm/MolmoVideo/molmo_video/hf_configs_Molmo-7B-D-0924/preprocessor_config.json"
processor = MolmoProcessor(tokenizer=tokenizer, 
                           preprocessor_config_path=preprocessor_config_path)



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

num_frames = 2 #TODO: change to 8

base_data_dir = '/l/users/salman.khan/molmo/pointing_dataset'
output_dir = '/l/users/salman.khan/workspace_pointing_lmm/models/llm_connector'
annotation_dir = "/l/users/salman.khan/workspace_pointing_lmm/datasets/annotations" 
annotation_files = glob.glob(f"{annotation_dir}/*.jsonl")  

dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")

SYSTEM_PROMPT = "You are given {num_frames} frames from a video. The frame indices are {selected_frame_idxs}. "
PROMPT_TEMPLATES = [
    'Point to {label}\nPlease say "This isn\'t in the video." if it is not in the video.',
    'Point to all occurrences of {label}',
    'Point to any {label} in the video',
    'Point to any {label} in the video.',
    'Point: Where are the {label}',
    'Show me where the {label} are',
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
    'show me all visible {label}',
]

def get_points_in_xml_format(points, caption):
    lines = ["<points"]
    
    for t, xy_list in points.items():
        for xy in xy_list:  
            x, y = xy["x"], xy["y"]
            if x == -1 and y == -1:
                lines.append(f' t="{int(t)}" There are none.')
                continue
            lines.append(f' t="{int(t)}" x="{x:.1f}" y="{y:.1f}"')

    lines.append(f' alt="{caption}">{caption}</points>')
    output = "".join(lines)
    return output

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
        question = SYSTEM_PROMPT.format(num_frames=num_frames, selected_frame_idxs=selected_frame_idxs) + random.choice(PROMPT_TEMPLATES).format(label=caption)
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
        
        device = torch.device("cuda")
        example_inputs = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in example_inputs.items()}
        
        batch_outputs.append(example_inputs)

    batch_inputs = {}
    for key in batch_outputs[0]:
        batch_inputs[key] = [ex[key] for ex in batch_outputs]
    padded_inputs = tokenizer.pad({"input_ids": [ex[0] for ex in batch_inputs['input_ids']],},
        return_tensors="pt",
    )
    padded_inputs["labels"] = torch.cat([example for example in batch_inputs["labels"]], 0)
    padded_inputs["loss_masks"] = torch.cat([example for example in batch_inputs["loss_masks"]], 0)
    padded_inputs["images"] = torch.cat([example for example in batch_inputs["images"]], 0)
    padded_inputs["image_input_idx"] = torch.cat([example for example in batch_inputs["image_input_idx"]], 0)
    padded_inputs["image_masks"] = torch.cat([example for example in batch_inputs["image_masks"]], 0)
    padded_inputs["position_ids"] = torch.cat([example for example in batch_inputs["position_ids"]], 0)

    return padded_inputs

model = MolmoForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

for name, param in model.named_parameters():
    if "att_proj" in name or "ff_proj" in name or "image_projector" in name or "image_pooling_2d" in name:
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

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
    learning_rate=1e-4,
    lr_scheduler_type="cosine", 
    logging_steps=10, 
    eval_strategy= "no",
    save_strategy="steps",
    save_steps=1000, 
    bf16=True, 
    tf32=True,  
    max_grad_norm=1.0, 
    warmup_ratio=0.03,  
    push_to_hub=False, 
    report_to=None,  
    dataset_text_field="", 
    dataset_kwargs={"skip_prepare_dataset": True},  
    dataloader_pin_memory=False,
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