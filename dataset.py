import json
from enum import IntEnum, auto
import regex as re
from collections import Counter
import numpy as np
import os
import torch
import torch.utils.data as data
from inp_out import read_file, write_file
from io import BytesIO


# System tokens
class SystemTokens(IntEnum):
    PAD = 0         # Padding token
    SOQ = auto()    # Start of question
    EOQ = auto()    # End of question
    SOA = auto()    # Start of answer
    EOA = auto()    # End of answer
    OOV = auto()    # Out of vocabulary
    NUM_OF = auto() # Number of tokens


# Pytorch dataset
class Dataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# Load conversations
def load_conversations(file, bucket=None):
    json_data = json.loads(read_file(file, 'r', bucket))
    conversations = []
    for key in json_data.keys():
        conv = []
        for message in json_data[key]["content"]:
            conv.append({
                "turn": True if message["agent"] == "agent_1" else False,
                "message": message["message"]
            })
        conversations.append(conv)
    return conversations


# Make vocabulary from conversations
def make_vocabulary(conversations, keep, file=None, bucket=None):
    msgs = []
    for conv in conversations:
        for message in conv:
            msgs.append(message["message"])
    counter = Counter()
    for message in msgs:
        # Find all sequences of Unicode letters (words) including accented characters
        words = re.findall(r'\p{L}+', message.lower(), flags=re.UNICODE)
        counter.update(words)
        # Find and count all characters that are not Unicode letters individually excluding spaces
        non_letters = re.findall(r'[^\p{L}\s]', message, flags=re.UNICODE)
        counter.update(non_letters)
    # Get the most common elements
    most_common = counter.most_common(keep)
    # Create a dictionary with {word: index} with index starting from the last system token
    vocabulary = {word: index + SystemTokens.NUM_OF for index, (word, _) in enumerate(most_common)}
    # Save the vocabulary to a json file
    if file is not None:
        write_file(file, json.dumps(vocabulary, indent=4), 'w', bucket)
    return vocabulary


# Tokenize a message
def text_to_tokens(text, vocabulary):
    # Tokenize the text while maintaining the order of words and symbols, excluding spaces
    tokens = re.findall(r'\p{L}+|[^\p{L}\s]', text.lower(), flags=re.UNICODE)
    # Convert tokens to indices, using the vocabulary or a default value for unknown tokens
    token_indices = [vocabulary.get(token, SystemTokens.OOV) for token in tokens]
    return token_indices


# Detokenize a message
def tokens_to_text(tokens, index_to_token):
    # Initialize an empty string to hold the reconstructed text
    reconstructed_text = ''
    for i, token in enumerate(tokens):
        # Skip system tokens
        if token < SystemTokens.NUM_OF and token != SystemTokens.OOV:
            continue
        # Get the string representation of the token
        token_str = index_to_token.get(token, '<UNK>')
        # Always add a space before a word except at the beginning
        if token_str.isalpha():
            if i > 0:  # Add a space before the word if it's not the first token
                reconstructed_text += ' ' + token_str
            else:  # Do not add a space if it's the first token
                reconstructed_text += token_str
        else:
             # Add a spce before a number if the last token is a letter
            if token_str.isdigit() and i > 0 and reconstructed_text[-1].isalpha():
                reconstructed_text += ' '
            reconstructed_text += token_str
    return reconstructed_text


# Create a dataset
def create_dataset(conversations, vocabulary, file=None, bucket=None):
    # Initialize lists to hold the input and output sequences
    X = []
    Y = []
    max_len = 0
    # Iterate over the conversations
    for conv in conversations:
        context_1 = []
        context_2 = []
        # Iterate over the source messages in the conversation
        for i in range(len(conv) - 1):
            # Src message
            src_msg = conv[i]["message"]
            turn = conv[i]["turn"]
            src_tokens = text_to_tokens(src_msg, vocabulary)
            if turn:
                context_1 += [SystemTokens.SOQ] + src_tokens + [SystemTokens.EOQ]
                context_2 += [SystemTokens.SOA] + src_tokens + [SystemTokens.EOA]
                X.append(np.array(context_1, dtype=np.int32))
            else:
                context_1 += [SystemTokens.SOA] + src_tokens + [SystemTokens.EOA]
                context_2 += [SystemTokens.SOQ] + src_tokens + [SystemTokens.EOQ]
                X.append(np.array(context_2, dtype=np.int32))
            # Tgt message
            tgt_msg = conv[i + 1]["message"]
            tgt_tokens = text_to_tokens(tgt_msg, vocabulary)
            if turn:
                target_1 = [SystemTokens.SOA] + tgt_tokens + [SystemTokens.EOA]
                target_2 = [SystemTokens.SOQ] + tgt_tokens + [SystemTokens.EOQ]
                Y.append(np.array(target_1, dtype=np.int32))
            else:
                target_1 = [SystemTokens.SOQ] + tgt_tokens + [SystemTokens.EOQ]
                target_2 = [SystemTokens.SOA] + tgt_tokens + [SystemTokens.EOA]
                Y.append(np.array(target_2, dtype=np.int32))
            # Update the maximum sequence length
            max_len = max(max_len, len(context_1), len(context_2), len(target_1), len(target_2))
    # Pad the sequences to the same length
    for i in range(len(X)):
        if len(X[i]) < max_len:
            X[i] = np.pad(X[i], (0, max_len - len(X[i])), mode='constant', constant_values=SystemTokens.PAD)
        if len(Y[i]) < max_len:
            Y[i] = np.pad(Y[i], (0, max_len - len(Y[i])), mode='constant', constant_values=SystemTokens.PAD)
    # Convert the lists to numpy arrays
    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    # Save the dataset to a numpy file
    if file is not None:
        buff = BytesIO()
        np.savez_compressed(buff, X=X, Y=Y)
        write_file(file, buff.getvalue(), 'wb', bucket)   
    return X, Y


def get_dataset(conv_file, vocab_file, dataset_file, max_vocab_size, bucket=None):
    file = read_file(dataset_file, 'rb', bucket)
    if file is not None:
        with BytesIO(file) as buff:
            npz = np.load(buff)
            X = npz['X']
            Y = npz['Y']
        vocabulary = json.loads(read_file(vocab_file, 'r', bucket))
    else:
        # Load conversations
        conversations = load_conversations(conv_file, bucket)
        # Make vocabulary
        vocabulary = make_vocabulary(conversations, max_vocab_size, vocab_file, bucket)
        # Create dataset
        X, Y = create_dataset(conversations, vocabulary, dataset_file, bucket)
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    # Pytorch dataset
    dataset = Dataset(torch.from_numpy(X).long(), torch.from_numpy(Y).long())
    return dataset, vocabulary, inv_vocabulary
