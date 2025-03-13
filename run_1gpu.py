# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    MyCollator,
)
from tqdm import tqdm
import os
import yaml
import json
import argparse
from utils import Config, set_seed

def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    
    print("Config:", config_dict)
    configs = Config(config_dict)
    set_seed(configs.seed)

    save_dir = os.path.join(configs.save_path, configs.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = AutoModelForCausalLM.from_pretrained(configs.model_id).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if configs.load_model_path != "None":
        saved_weights = torch.load(configs.load_model_path, map_location='cuda')
        model.load_state_dict(saved_weights, strict=False)

    model.resize_token_embeddings(len(tokenizer))

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    scheduled_stage = 0 if (configs.cot or configs.no_cot) else 0

    dataset_gen_val = get_question_latent_dataset(
        scheduled_stage,
        base_dataset_valid,
        configs,
        start_id,
        latent_id,
        end_id,
        no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
    )

    valid_gen_dataloader = torch.utils.data.DataLoader(
        dataset_gen_val,
        num_workers=1,
        pin_memory=True,
        batch_size=1,
        collate_fn=collator,
    )

    model.eval()
    cor, cor_cot, total = 0, 0, 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_gen_dataloader, desc="Evaluating")):
            test_idx = batch["idx"][0]
            batch = {k: v.to('cuda') for k, v in batch.items() if v is not None and k not in ["idx", "position_ids"]}

            answer = answers_val[test_idx]
            answer_cot = cot_val[test_idx]
            question = question_val[test_idx]

            outputs = model.generate(**batch, max_new_tokens=max_new_tokens)
            
            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_output = text_output.split("#")[-1].replace(",", "").strip()
            cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

            if idx < 5:
                print(f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'")
                print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                print(f"Extracted Output: '{answer_output}'")

            cor += answer_output == answer
            cor_cot += cot_output == answer_cot
            total += 1

    print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
    print(f"CoT match on validation set: {cor_cot} / {total} = {cor_cot/total}")

if __name__ == "__main__":
    main()

