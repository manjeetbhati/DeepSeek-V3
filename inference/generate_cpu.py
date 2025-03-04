import os
import json
from argparse import ArgumentParser
from typing import List

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

def sample(logits, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.multinomial(1).squeeze(-1)

@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, "Prompt length exceeds model max sequence length"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cpu")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
    
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cpu")
    prompt_mask = tokens != -1
    
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = sample(logits, temperature) if temperature > 0 else logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    torch.set_num_threads(os.cpu_count())  # Utilize available CPU cores
    
    with open(config, "r") as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = 1
    model = Transformer(model_args)
    load_model(model, ckpt_path, device="cpu")
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
    eos_id = tokenizer.eos_token_id
    
    if interactive:
        while True:
            try:
                prompt = input("Input: ")
                if prompt.strip().lower() == "exit":
                    break
                prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.tolist()
                outputs = generate(model, prompt_tokens, max_new_tokens, eos_id, temperature)
                for output in outputs:
                    print(tokenizer.decode(output))
            except KeyboardInterrupt:
                break
    else:
        with open(input_file, "r") as f:
            prompts = f.readlines()
        for prompt in prompts:
            prompt_tokens = tokenizer(prompt.strip(), return_tensors="pt").input_ids.tolist()
            outputs = generate(model, prompt_tokens, max_new_tokens, eos_id, temperature)
            for output in outputs:
                print(tokenizer.decode(output))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--input-file", type=str, default="", help="Path to a file containing input prompts.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature value for sampling.")
    args = parser.parse_args()
    main(
        ckpt_path=args.ckpt_path,
        config=args.config,
        input_file=args.input_file,
        interactive=args.interactive,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

