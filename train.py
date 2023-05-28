from accelerate import Accelerator
from MEGABYTE_pytorch import MEGABYTE
import bitsandbytes as bnb

import random
import signal
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from bitsandbytes.optim.adam import Adam8bit
from sophiag import SophiaG

# constants
TOTAL_BATCHES = 15000000 # approximately 15M btaches of 8192 for 1 epoch of wikipedia
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 8
NUM_BATCHES = TOTAL_BATCHES // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY
LEARNING_RATE = 4e-4
VALIDATE_EVERY  = 1600 // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY
GENERATE_EVERY  = 8000 // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY
PRIME_LEN = 100
SEQ_LEN = 8192

# helpers

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable: {100 * trainable_params / all_param}")

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


class WrappedDataset(IterableDataset):
    def __init__(self, huggingface_dataset, seq_len):
        self.huggingface_dataset = huggingface_dataset
        self.seq_len = seq_len

    def __iter__(self):
        buffer = torch.tensor([], dtype=torch.long)
        while True:  # Infinite loop over the dataset
            for row in self.huggingface_dataset:
                formatted_text = f"=={row['title']}==\n{row['text']}"
                x = np.frombuffer(formatted_text.encode(), dtype=np.uint8)
                buffer = torch.cat((buffer, torch.from_numpy(x)), dim=0)
                while len(buffer) >= self.seq_len:
                    yield buffer[:self.seq_len].long()
                    buffer = buffer[self.seq_len:]

def main():
    accelerator = Accelerator()

    wikipedia_ds = load_dataset("lsb/enwiki20230101")

    # instantiate GPT-like decoder model

    device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

    if device_properties.major == 8 and device_properties.minor == 0:
        flash_attn = False  # set to false if using A100
    else:
        flash_attn = True

    # global: D: 1024, L: 14
    # local: D: 1024, L: 18
    model = MEGABYTE(
        num_tokens = 8192,
        dim = 512,
        depth = (6, 4),
        max_seq_len = (2048, 1024),
        flash_attn = flash_attn
    ).to(accelerator.device, dtype=torch.bfloat16)
    # model = MEGABYTE(
    #     num_tokens = 8192,
    #     dim = (768, 1024),  # embeddings dimenstion -> d_head
    #     max_seq_len = (2048, 1024),  # number of embeddings -> d_model
    #     depth = (14, 18),  # numof layers => #L
    #     dim_head = 92,
    #     heads = 16,
    #     ff_dropout=0.1,
    #     attn_dropout=0.1,
    #     flash_attn = True,
    # ).to(accelerator.device, dtype=torch.bfloat16)

    print_trainable_parameters(model)
    # prepare enwik8 data

    ds = wikipedia_ds["train"].train_test_split(test_size=0.001)
    train_dataset = WrappedDataset(ds["train"], SEQ_LEN)
    val_dataset = WrappedDataset(ds["test"], SEQ_LEN)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    # optimizer

    # optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = Adam8bit(model.parameters(), lr=LEARNING_RATE)
    optimizer = SophiaG(model.parameters(), lr=LEARNING_RATE)

    # training
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    signal.signal(
        signal.SIGINT,
        lambda signal, frame: (torch.save(model.state_dict(), 'model_out_sigint.pt'), exit(0)),
    )

    pbar = tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training')
    for i in pbar:
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader), return_loss = True)
            accelerator.backward(loss)
            # loss.backward()

        pbar.set_description(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader), return_loss = True)
                pbar.set_description(f'validation loss: {loss.item()}')
            torch.save(model.state_dict(), 'path_to_save_your_model.pt')

    torch.save(model.state_dict(), 'model_out.pt')


if __name__ == "__main__":
    main()
