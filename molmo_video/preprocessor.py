"""
Processor class for Molmo.
"""

from typing import Optional
import re
import PIL
from PIL import ImageOps
# from PIL.Image import Image
from PIL import Image

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import torch


from transformers import AutoTokenizer

from .image_processor import MolmoImageProcessor
from transformers.image_utils import ImageInput


DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"
FRAME_START_TOKEN = f"<frame_start>"
FRAME_END_TOKEN = f"<frame_end>"
EXTRA_TOKENS = (FRAME_START_TOKEN, FRAME_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)


def get_special_token_ids(tokenizer):
    ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in EXTRA_TOKENS]
    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


##############################################################################################################

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class MolmoImagesKwargs:
    """
    Holds configuration values for image processing.
    """
    max_crops: int = 12
    overlap_margins: List[int] = field(default_factory=lambda: [4, 4])
    base_image_input_size: List[int] = field(default_factory=lambda: [336, 336])
    image_token_length_w: int = 12
    image_token_length_h: int = 12
    image_patch_size: int = 14
    image_padding_mask: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> "MolmoImagesKwargs":
        """
        Load configuration from a JSON file and return a MolmoImagesKwargs instance.
        Only keys matching the class annotations are used.
        """
        with open(json_path, "r") as f:
            config = json.load(f)
        # Filter out only the keys that are relevant to the class
        valid_keys = set(cls.__annotations__.keys())
        filtered_config: Dict[str, Any] = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)

@dataclass
class MolmoTextKwargs:
    """
    Holds configuration values for text processing.

    Default values:
        {
            "style": "long_caption",
            "system_prompt": "none",
            "message_format": "role",
            "always_start_with_space": True,
            "sequence_length": 1536,
            "padding": False,
        }
    """
    style: str = "long_caption"
    system_prompt: str = "none"
    message_format: str = "role"
    always_start_with_space: bool = True
    sequence_length: int = 1536
    padding: bool = False

    @classmethod
    def from_json(cls, json_path: str) -> "MolmoTextKwargs":
        """
        Load configuration from a JSON file and return a MolmoTextKwargs instance.
        Only keys matching the class attributes are used.
        """
        with open(json_path, "r") as f:
            config = json.load(f)
        valid_keys = set(cls.__annotations__.keys())
        filtered_config: Dict[str, Any] = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered_config)
    
##############################################################################################################


class MolmoProcessor():

    def __init__(self, tokenizer : AutoTokenizer = None,
                 preprocessor_config_path: str = "preprocessor_config.json"):
        
        # 
        self._special_tokens = None
        
        
        # Load kwargs  for image processing
        images_kwargs_instance = MolmoImagesKwargs.from_json(preprocessor_config_path)
        self.images_kwargs_dict = images_kwargs_instance.__dict__
        # Load kwargs  for text processing
        text_kwargs_instance = MolmoTextKwargs() #.from_json("text_preprocessor_config.json")
        self.text_kwargs_dict = text_kwargs_instance.__dict__
        
        # 
        self.image_processor = MolmoImageProcessor(**self.images_kwargs_dict)
        self.tokenizer = tokenizer

    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens

    def get_tokens_input(self, prompt, message_format, always_start_with_space):
        if message_format == "none" or message_format is None:
            pass
        elif message_format == "role": #NOTE: this is run by default
            prompt = "User: " + prompt + " Assistant:"
        else:
            raise NotImplementedError(f"Message format {message_format} not implemented")

        if always_start_with_space: #NOTE: this is run by default
            prompt = " " + prompt

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        return tokens
    
    def get_tokens_input_for_training(self, train_prompt, message_format, always_start_with_space, for_training=True):
        if message_format == "none" or message_format is None:
            pass
        elif message_format == "role": #NOTE: this is run by default
            prompt = train_prompt
        else:
            raise NotImplementedError(f"Message format {message_format} not implemented")
        if not for_training:
            tokens, loss_mask = [], []
            message_ids = self.tokenizer.encode(train_prompt)
            message_ids.append(self.tokenizer.eos_token_id)
            tokens += message_ids
            loss_mask += [True]*len(message_ids)
            tokens = np.array(tokens, dtype=np.int32)
            return tokens, loss_mask

        split_text = train_prompt.split("Assistant:")

        # Formatting into the required list format
        messages = [" " + split_text[0].strip() + " Assistant:", " " + split_text[1].strip()]
        tokens, loss_mask = [], []
        message_ids = self.tokenizer.encode(messages[0])
        tokens += message_ids
        loss_mask += [False]*len(message_ids)
        message_ids = self.tokenizer.encode(messages[1])
        message_ids.append(self.tokenizer.eos_token_id)
        loss_mask += [True]*len(message_ids)
        tokens += message_ids
        tokens = np.array(tokens, dtype=np.int32)
        return tokens, loss_mask

    def process(
        self,
        text = None, # str : question # TextInput
        images: ImageInput = None, # List of PIL images
        *,
        tokens = None, # Optional[PreTokenizedInput]
        for_training = False, # bool
    ):
        if tokens is None:
            if not for_training:
               tokens = self.get_tokens_input(
                    text,
                    self.text_kwargs_dict["message_format"],
                    self.text_kwargs_dict["always_start_with_space"],
                )
            else:
                tokens, loss_mask = self.get_tokens_input_for_training(
                    text,
                    self.text_kwargs_dict["message_format"],
                    self.text_kwargs_dict["always_start_with_space"],
                )

        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            image_arrays = []
            for image in images:
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            
            images = image_arrays # list of np arrays # dimensions: (num_images, height, width, channels)
            
            # For now only support inserting images at the start
            image_idx = [-1]*len(images) # ??
        else:
            image_idx = None

        # sequence_length = output_kwargs["text_kwargs"]["sequence_length"]
        sequence_length = self.text_kwargs_dict["sequence_length"]

        image_patch_token_id = self.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
        image_col_token_id = self.special_token_ids[DEFAULT_IM_COL_TOKEN]
        image_start_token_id = self.special_token_ids[DEFAULT_IM_START_TOKEN]
        image_end_token_id = self.special_token_ids[DEFAULT_IM_END_TOKEN]
        frame_start_token_id = self.special_token_ids[FRAME_START_TOKEN]
        frame_end_token_id = self.special_token_ids[FRAME_END_TOKEN]

        out = self.image_processor.multimodal_preprocess(
            images=images, #TODO: edit multimodal_preprocess 
            image_idx=image_idx, # ??
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=sequence_length,
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
            frame_start_token_id=frame_start_token_id,
            frame_end_token_id=frame_end_token_id,
            loss_mask = loss_mask,
            **self.images_kwargs_dict
        )
        
        return out