import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import logging
from dataset import get_dataset, SystemTokens, text_to_tokens, tokens_to_text
from model import Transformer
import time
import argparse
import json
from inp_out import read_file, write_file, log_to_file
from io import BytesIO
import boto3



# Train the transformer
def train(epoch, n_epochs, transformer, dataloader, src_vocab_size, max_seq_len, criterion, optimizer, device, logger, log_every, checkpoint, bucket=None, log_file=None):
    for epoch in range(epoch, n_epochs):
        transformer.train()
        total_loss = 0
        total_samples = 0
        start_time = time.time()
        for src, tgt in dataloader:
            if src.size(1) > max_seq_len:
                src = src[:, -max_seq_len:]
            if tgt.size(1) > max_seq_len + 1:
                tgt = tgt[:, :max_seq_len + 1]
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = transformer(src, tgt[:, :-1], device=device)
            loss = criterion(output.contiguous().view(-1, src_vocab_size), tgt[:, 1:].contiguous().view(-1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += src.size(0)
            # Log the loss every LOG_EVERY seconds
            if time.time() - start_time > log_every:
                start_time = time.time()
                mean_loss = total_loss / total_samples
                buff = BytesIO()
                torch.save((transformer.state_dict(), optimizer.state_dict(), epoch), buff)
                write_file(checkpoint, buff.getvalue(), 'wb', bucket)
                log_to_file(f"Train\t-> Epoch: {epoch}, Loss: {mean_loss:.4f}", logger, bucket, log_file)
                print(f"Train\t-> Epoch: {epoch}, Loss: {mean_loss:.4f}")
        # Log the loss at the end of the epoch
        mean_loss = total_loss / total_samples
        buff = BytesIO()
        torch.save((transformer.state_dict(), optimizer.state_dict(), epoch + 1), buff)
        write_file(checkpoint, buff.getvalue(), 'wb', bucket)
        log_to_file(f"Train\t-> Epoch: {epoch+1}, Loss: {mean_loss:.4f}", logger, bucket, log_file)
        print(f"Train\t-> Epoch: {epoch+1}, Loss: {mean_loss:.4f}")


# Use the transformer to predict the next values
def predict(transformer, src, max_seq_len, temp, device):
    transformer = transformer.to(device)
    transformer.eval()
    src = src.to(device)
    tgt = torch.full((src.size(0), max_seq_len), SystemTokens.PAD, dtype=torch.long, requires_grad=False).to(device)
    with torch.no_grad():
        tgt[:, 0] = SystemTokens.SOA
        for i in range(max_seq_len - 1):
            output = transformer(src, tgt[:, :-1], device=device) # (1, max_seq_len, vocab_size)
            if temp > 1e-3:
                tokens = output / temp
                tokens = torch.softmax(tokens, dim=-1)
                next_token = torch.multinomial(tokens[:, i], 1)
                next_token = torch.squeeze(next_token, dim=-1)  # Remove extra dimension
            else:
                tokens = output.argmax(dim=-1) # (max_seq_len,)
                next_token = tokens[:, i]  # next token
            tgt[:, i+1] = next_token    # update the target 
            if np.all(next_token.cpu().numpy() == SystemTokens.EOA):
                break
    return tgt


def test(transformer, vocabulary, inv_vocabulary, max_seq_len, temp, device):
    context = []
    while True:
        message = input("You: ")
        if message == "exit":
            break
        if message == "clear":
            context = []
            continue
        # Tokenize the input message
        src_tokens = text_to_tokens(message, vocabulary)
        # Add special tokens
        src_tokens = [SystemTokens.SOQ] + src_tokens + [SystemTokens.EOQ]
        # Add to the context
        context += src_tokens
        # padding
        if len(context) < max_seq_len:
            src = context + [SystemTokens.PAD] * (max_seq_len - len(src_tokens))
        else:
            src = context[-max_seq_len:]
        src = torch.tensor([src], dtype=torch.long)
        # Transofomer prediction
        tgt = predict(transformer, src, max_seq_len, temp, device)
        tgt = tgt[0].cpu().numpy().tolist()
        response = tokens_to_text(tgt, inv_vocabulary)
        print(f"Bot: {response}")
        # Add special tokens
        tgt = [SystemTokens.SOA] + tgt + [SystemTokens.EOA]
        # Add to the context
        context += tgt


def main(args):

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Initiate AWS S3 client
    if args.s3_bucket is not None:
        with open('./AWS_CREDENTIALS.json', 'r') as f:
            aws_credentials = json.load(f)
        boto3.setup_default_session(region_name='eu-central-1',
                                    aws_access_key_id=aws_credentials['AWS_ACCESS_KEY_ID'],
                                    aws_secret_access_key=aws_credentials['AWS_SECRET_ACCESS_KEY'])

    # Load the dataset 
    dataset, vocabulary, inv_vocabulary = get_dataset(args.conv_file, args.vocab_file, args.dataset_file, args.max_vocab_size - SystemTokens.NUM_OF, args.s3_bucket)
    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)


    # Create the model
    transformer = Transformer(args.max_vocab_size, args.max_vocab_size, args.d_model, args.num_heads, args.num_layers, args.d_ff, args.max_seq_len, args.dropout)
    epoch = 0
    opt_state_dict = None
    file = read_file(args.checkpoint, 'rb', args.s3_bucket)
    if file is not None:
        with BytesIO(file) as buff:
            state_dict, opt_state_dict, epoch = torch.load(buff)
            transformer.load_state_dict(state_dict)
            log_to_file(f"Loaded model from {args.checkpoint}", logger, args.s3_bucket, args.log_file)
    transformer = transformer.to(args.train_device)


    # Train the model
    criterion = nn.CrossEntropyLoss(ignore_index=SystemTokens.PAD) # ignore padding
    optimizer = Adam(transformer.parameters(), lr=args.opt_lr, betas=(args.opt_beta_1, args.opt_beta_2), eps=args.opt_eps)
    if opt_state_dict is not None:
        optimizer.load_state_dict(opt_state_dict)
    train(epoch, args.num_epochs, transformer, train_loader, args.max_vocab_size, args.max_seq_len, criterion, optimizer, args.train_device, logger, args.log_every, args.checkpoint, args.s3_bucket, args.log_file)

    # Test the model
    if args.chat is True:
        test(transformer, vocabulary, inv_vocabulary, args.max_seq_len, args.temp, args.test_device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", type=str, default="")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--checkpoint", type=str, default='transformer.pt')
    parser.add_argument("--conv_file", type=str, default='conversations.json')
    parser.add_argument("--vocab_file", type=str, default='vocabulary.json')
    parser.add_argument("--dataset_file", type=str, default='dataset.npz')
    parser.add_argument("--max_vocab_size", type=int, default=16000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--opt_lr", type=float, default=0.0001)
    parser.add_argument("--opt_beta_1", type=float, default=0.9)
    parser.add_argument("--opt_beta_2", type=float, default=0.98)
    parser.add_argument("--opt_eps", type=float, default=1e-9)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train_device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--log_file", type=str, default='transformer.log')
    parser.add_argument("--log_every", type=int, default=60)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--test_device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--chat", type=bool, default=False)
    args = parser.parse_args()

    main(args)
    
