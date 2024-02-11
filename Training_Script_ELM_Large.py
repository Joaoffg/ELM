import os

from datasets import load_dataset
import re
import os
import torch
from transformers import LlamaTokenizer, LlamaConfig, LlamaModel, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

base_path='filtered_text_files'

dataset = load_dataset("text",
                       data_dir=base_path, streaming = True)


tokenizer = LlamaTokenizer.from_pretrained('erasmian-lm')
tokenizer.pad_token = "<pad>"

def chunk_examples(examples,chunk_lenght=256, min_chunk_lenght = 25):
    chunks = []
    for text in examples["text"]:
            tokenized = tokenizer(text,add_special_tokens=False)
        #if len(tokenized.input_ids) > min_chunk_lenght:
            input_ids = [tokenizer.bos_token_id] + tokenized.input_ids + [tokenizer.eos_token_id]
            attention_mask = [1] + tokenized.attention_mask + [1]
            inputs=[]
            mask=[]
            labels=[]

            for i in range(0, len(tokenized.input_ids), chunk_lenght):
                cunk_input_ids = input_ids[i:i + chunk_lenght]
                cunk_att_mask = attention_mask[i:i + chunk_lenght]
                cur_chunk_len = len(cunk_input_ids)

                if  cur_chunk_len < chunk_lenght:
                    cunk_input_ids = cunk_input_ids + [tokenizer.pad_token_id]*(chunk_lenght - cur_chunk_len)
                    cunk_att_mask = cunk_att_mask + [0]*(chunk_lenght - cur_chunk_len)

                inputs.append(cunk_input_ids)
                mask.append(cunk_att_mask)
                labels.append(cunk_input_ids)

                return {"input_ids": inputs, "attention_mask": mask, "labels":labels}


chunked_dataset = dataset['train'].map(chunk_examples, batched=True, batch_size= 1, remove_columns=['text'])


config = LlamaConfig( # around 1B parameter LlaMa
    vocab_size = 32000,
    hidden_size= int(2048),
    intermediate_size = int(5120),
    num_hidden_layers = int(16),
    num_attention_heads = int(32),
    max_position_embeddings = 2048 ,
    rms_norm_eps = 1e-12
)

model = LlamaForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"LlaMa Model Size: {model_size/1000**2:.1f}M parameters")

data_collator= DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler


dataset = chunked_dataset.with_format("torch")
dataloader = DataLoader(dataset, collate_fn=data_collator)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train().to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, weight_decay=0.1)
lr_scheduler = get_scheduler(
      "cosine",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=10255294
  )

# steps 10255294

for epoch in range(1):
    dataset.set_epoch(epoch)
    running_loss = 0
    for i, batch in enumerate(tqdm(dataloader, total=10255294)):
        if i == 10255294:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(True):
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        if i % 100 == 0:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0
        if i % 20000 == 0:
            model.save_pretrained("erasmian-lm/loopv1/checkpoint"+"_"+str(i), from_pt=True)

