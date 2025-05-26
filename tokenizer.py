# tokenizer.py - Dashun Feng
# Produces tokenized_dataset based on articles.csv in Data

import pandas as pd
from datasets import Dataset
from transformers import LongformerTokenizer

# Load dataframe from articles.csv and cut for content, bias and ID
article_dataframe = pd.read_csv("./Data/articles.csv")
article_dataframe = article_dataframe[['content', 'bias', 'id']].dropna()

# Convert article_dataframe to Hugging Face dataset
article_dataset = Dataset.from_pandas(article_dataframe)

# Define tokenization function
def tokenize(batch):
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    return tokenizer(batch['content'], truncation=True, padding="max_length", max_length=4096)

# Tokenize the dataset and split into train and test sets (80% train - 20% test)
tokenized_dataset = article_dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)