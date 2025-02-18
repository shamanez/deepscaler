from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import torch
import numpy as np
from datasets import load_dataset
from typing import List, Dict


class AppWorldDataset(Dataset):
    """Dataset for loading tasks from Hugging Face hub."""
    
    def __init__(self, 
                 dataset_name: str,
                 tokenizer: PreTrainedTokenizer,
                 prompt_key: str = 'prompt',
                 max_prompt_length: int = 1024,
                 num_tasks: int = None,
                 filter_prompts: bool = True,
                 cache_dir: str = '~/.cache/verl/rlhf',
                 chat_template_func = None,
                 return_raw_chat: bool = False,
                 truncation: str = 'error',
                 split: str = "train",
                ):
        # Load dataset from Hugging Face hub
        self.dataset = load_dataset(dataset_name, split=split)
        if num_tasks:
            self.dataset = self.dataset.select(range(num_tasks))
            
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts
        self.return_raw_chat = return_raw_chat
        self.truncation = truncation

        # Apply chat template to all examples
        self.dataset = self.dataset.map(self._apply_chat_template)
        
        # Filter long prompts if needed
        if filter_prompts:
            self.dataset = self.dataset.filter(self._filter_long_prompts)
            print(f'Dataset size after filtering: {len(self.dataset)}')
    
    def _apply_chat_template(self, example):
        """Apply chat template to messages"""
        prompt_with_template = self.tokenizer.apply_chat_template(
            example['chat_messages'],
            add_generation_prompt=True,
            tokenize=False
        )
        return {'prompt_with_template': prompt_with_template, **example}
    
    def _filter_long_prompts(self, example):
        """Filter examples that exceed max_prompt_length"""
        return len(self.tokenizer.encode(example['prompt_with_template'])) <= self.max_prompt_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get item with pre-processed prompt"""
        example = self.dataset[idx]
        
        # Tokenize and process the pre-templated prompt
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=example['prompt_with_template'],
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        output = {
            "task_id": example['task_id'],
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "reward_model": example['reward_model'],
            "index": example['extra_info']['index'],
            "history": []  # Store episode trajectory
        }

        if self.return_raw_chat:
            output['raw_chat'] = example['chat_messages']

        return output

def collate_fn_appworld(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    # Special handling for multi-step episodes
    output = {}
    output.update(tensors)
    output.update(non_tensors)

    # Store episode history
    output["history"] = [data["history"] for data in data_list]

    return output
