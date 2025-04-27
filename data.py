import os
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


from molmo_video.memory import MolmoForCausalLM
from molmo_video.preprocessor import (FRAME_START_TOKEN, FRAME_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader

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

from utils import get_coords, compute_mse_points, plot_metric, extract_caption, \
                    pil_to_np

num_frames = 4 #TODO: 

method = 'memory_mean'
base_data_dir = '/l/users/salman.khan/molmo/pointing_dataset'
output_dir = f'/l/users/salman.khan/workspace_pointing_lmm/models/{method}'
annotation_dir = "/l/users/salman.khan/workspace_pointing_lmm/datasets/annotations" 
annotation_files = glob.glob(f"{annotation_dir}/*.jsonl")
# annotation_files = annotation_files[2:3]
print(f'annotation_files: {annotation_files}')

dataset = load_dataset("json", 
                       data_files=annotation_files,
                       split="train")

print(f"Total Number of Samples: {len(dataset)}")

for idx in range(len(dataset)):
    print(f"{dataset[idx]['video']}")
    frames_dir = os.path.join(base_data_dir, dataset[idx]['video'])
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
    print(f"total frames: {len(dataset[idx]['points'])}, {len(frame_files)}")

    assert len(frame_files) == len(dataset[idx]['frame_idxs']), f"Expected {len(frame_files)} frames, but got {len(dataset[idx]['frame_idxs'])}"
    assert len(frame_files) == len(dataset[idx]['points'])