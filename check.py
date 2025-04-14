import os
import random
from PIL import Image
from transformers import AutoTokenizer
from molmo_video.model_preprocessor import MultiModalPreprocessor
import torch
import numpy as np
import torch.nn.functional as F
from transformers import EvalPrediction

model_id = "allenai/Molmo-7B-D-0924"
tokenizer = AutoTokenizer.from_pretrained(model_id)
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)

# preprocessor_config_path = "/home/salman.khan/shehan/workspace_pointing_lmm/MolmoVideo/molmo_video/hf_configs_Molmo-7B-D-0924/preprocessor_config.json"
# processor = MolmoProcessor(tokenizer=tokenizer, 
#                            preprocessor_config_path=preprocessor_config_path)


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

from utils import SYSTEM_PROMPT, PROMPT_TEMPLATES

from datasets import load_dataset
from torch.utils.data import DataLoader
import glob
from peft import LoraConfig, get_peft_model

num_frames = 1 #TODO: change to 8

base_data_dir = '/l/users/salman.khan/molmo/pointing_dataset'
output_dir = '/l/users/salman.khan/workspace_pointing_lmm/models/llm_connector'
annotation_dir = "/l/users/salman.khan/workspace_pointing_lmm/datasets/annotations" 
annotation_files = glob.glob(f"{annotation_dir}/*.jsonl")  
annotation_files = annotation_files[1:2]

dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")
device = 'cuda'

def get_points_in_xml_format(points, caption):
    lines = ["<point"]
    
    for t, xy_list in points.items():
        xy_list = sorted(xy_list, key=lambda p: (p["x"], p["y"]))
        for xy in xy_list:  
            x, y = xy["x"], xy["y"]
            if x == -1 and y == -1:
                lines.append(f' There are none.')
                continue
            lines.append(f' x="{x:.1f}" y="{y:.1f}"')

    lines.append(f' alt="{caption}">{caption}</point>')
    output = "".join(lines)
    return output



def get_output(images, prompt, model):
    # process the image and text
    conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        chat_template=tokenizer.chat_template,
        add_generation_prompt=False,
        tokenize=False,
        return_dict=False
    )
    split_text = prompt.split("Assistant:")
    messages = [" " + split_text[0].strip() + " Assistant:", "" + split_text[1].strip()]
    image_arrays = []
    for image in images:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image_arrays.append(np.array(image))
    images = image_arrays

    inputs = processor(
        images=images,
        messages=messages,
    )
    # move inputs to the correct device and make a batch of size 1
    inputs = {k: torch.from_numpy(v).to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs['input_ids'] = inputs['input_tokens']
    # generate output; maximum 200 new tokens; stop generation when 
        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    with torch.autocast(device_type=str(device), enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings='<|endoftext|>'),
            tokenizer=processor.tokenizer
        )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def random_augmentation(batch):
    new_batch = {"images": [], "frame_idxs": [], "points": [], "question": [], "answer": [], "caption": [], "video_dir": []}
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
        question = "Point to {caption}"
        new_batch["question"].append(question)
        answer = get_points_in_xml_format(selected_points, caption)
        new_batch["answer"].append(answer)
        new_batch["caption"].append(caption)
        new_batch["video_dir"].append(video_dir)
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
                for_training=False
            )
        else:
            example_inputs = processor.process(
                text=prompt,
                for_training=False
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
    device_map='auto',
)

augmented_dataset = dataset.with_transform(random_augmentation)
dataloader = DataLoader(augmented_dataset, batch_size=1, shuffle=True)

for batch in augmented_dataset:
    images = batch['images']
    pred = get_output(images, prompt=f'Point to {batch["caption"]}', model=model)
    print(f"video dir: {batch['video_dir']}, frame_idxs: {batch['frame_idxs']}")
    print(f"Pred: {pred}")
    print(f"GT: {batch['answer']}")