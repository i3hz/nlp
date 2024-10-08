import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import nltk
import spacy
import string
import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast

import pandas as pd

# Replace 'datas.csv' and 'output2.csv' with the actual file names
input_file = 'QA_dataset.csv'

# Read the input CSV file into a pandas DataFrame with the correct delimiter
df = pd.read_csv(input_file)
df = df.sample(frac = 1)
df = df.reset_index(drop=True)

# Initialize an empty DataFrame for transformed data
data = pd.DataFrame(columns=['context', 'question', 'answer'])

frames = []

# Iterate through each row and transform the data
for _, row in df.iterrows():
    paragraph = row['Paragraphs']
    temp_df = pd.DataFrame({
        'context': [paragraph] * 3,
        'question': [row[f'Question{i}'] for i in range(1, 4)],
        'answer': [row[f'Answer{i}'] for i in range(1, 4)]
    })
    
    frames.append(temp_df)

# Concatenate the list of DataFrames
data = pd.concat(frames, ignore_index=True)
#print(data)

Q_LEN = 256   # Question Length
T_LEN = 32    # Target Length
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10

TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
OPTIMIZER = AdamW(MODEL.parameters(), lr=0.00001)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)
print(DEVICE)