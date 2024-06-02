import re
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')



def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=10)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def prepare_input_data(order_date, brand, model, issue, master, device_price, brand_means):

    cleaned_issue = clean_text(issue)

    issue_embedding = get_bert_embedding(cleaned_issue)

    order_year = int(order_date.split('-')[0])
    inflation_factor = inflation_dict.get(order_year, 1)
    master_factor = 1 if master == 'experienced' else 0
    brand_encoded = brand_means.get(brand.lower(), 0)

    input_vector = np.concatenate((
        [inflation_factor],
        [device_price],
        [master_factor],
        [brand_encoded],
        issue_embedding
    ))

    return input_vector.reshape(1, -1)


def predict_repair_cost(order_date, brand, model, issue, master, device_price, model_path, brand_means):

    catboost_model = CatBoostRegressor()
    catboost_model.load_model(model_path)

    input_data = prepare_input_data(order_date, brand, model, issue, master, device_price, brand_means)

    predicted_cost = catboost_model.predict(input_data)
    return predicted_cost[0]



if __name__ == '__main__':

    model_path = 'models/catboost_model.cbm'

    # Предсказание стоимости ремонта
    predicted_cost = predict_repair_cost(order_date, brand, model, issue, master, device_price, model_path, brand_means)
    print(f'Предсказанная стоимость ремонта: {predicted_cost}')
