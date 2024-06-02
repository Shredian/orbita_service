import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import re


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=10)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def mean_target_encoding(df, category_col, target_col):
    means = df.groupby(category_col)[target_col].mean()
    return df[category_col].map(means)



df = pd.read_excel("data/processed/data_last.csv")


# Очистка текста и получение эмбеддингов
df['cleaned_issue'] = df['issue'].apply(clean_text)
df['issue_embedding'] = df['cleaned_issue'].apply(get_bert_embedding)

# Mean target encoding для брендов
brand_means = mean_target_encoding(df, 'brand', 'repair_cost').to_dict()
df['brand_encoded'] = df['brand'].map(brand_means)


df['order_year'] = df['order_date'].apply(lambda x: int(x.split('-')[0]))
df['inflation_factor'] = df['order_year'].map(inflation_dict)

# Преобразование данных в вектор для модели
def prepare_input_vector(row):
    master_factor = 1 if row['master'] == 'experienced' else 0
    input_vector = np.concatenate((
        [row['inflation_factor']],
        [row['device_price']],
        [master_factor],
        [row['brand_encoded']],
        row['issue_embedding']
    ))
    return input_vector

df['input_vector'] = df.apply(prepare_input_vector, axis=1)

# Разделение данных на обучающую и тестовую выборки
X = np.vstack(df['input_vector'].values)
y = df['repair_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
