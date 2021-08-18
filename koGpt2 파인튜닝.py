from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
from transformers import PreTrainedTokenizerFast
import os
from sklearn.model_selection import train_test_split
import json


class SongLyrics(Dataset):
    def __init__(self, series, truncate=False, gpt2_type="gpt2", max_length=1010):

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                                 bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                                 pad_token='<pad>', mask_token='<mask>')
        self.lyrics = []

        for row in tqdm(series):
            self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"</s>{row[:max_length]}</s>")
            ))
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]


dataset = SongLyrics(trainset, gpt2_type="gpt2")

# Get the tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# Accumulated batch size (since GPT2 is so big)


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


writer = SummaryWriter('scalar/')


def train(
    dataset, model, tokenizer,
    batch_size=8, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False, save_model_on_epoch=False,
):
    acc_steps = 100
    device = torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in enumerate(tqdm(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(
                entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )

    writer.close()
    return model


model = train(dataset, model, tokenizer,
              save_model_on_epoch=True, output_dir="./gptPaper2")
