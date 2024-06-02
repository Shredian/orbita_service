import pandas as pd
from fuzzywuzzy import process, fuzz


my_table = pd.read_excel("data/raw/data1217.csv")

internet_table = pd.read_csv("data/external/electronics_products_pricing.csv")

my_table['brand'] = my_table['brand'].str.lower()
my_table['model'] = my_table['model'].str.lower()
internet_table['brand'] = internet_table['brand'].str.lower()
internet_table['model'] = internet_table['model'].str.lower()

threshold = 80


def find_price(row, internet_df, threshold):
    brand = row['brand']
    model = row['model']

    filtered_df = internet_df[internet_df['brand'] == brand]
    if filtered_df.empty:
        return None

    # Нечеткое сопоставление модели
    matches = process.extract(model, filtered_df['model'], scorer=fuzz.token_set_ratio, limit=1)
    best_match, score = matches[0] if matches else (None, 0)


    if score >= threshold:
        price = filtered_df[filtered_df['model'] == best_match]['price'].values[0]
        return price
    return None


my_table['internet_price'] = my_table.apply(find_price, axis=1, internet_df=internet_table, threshold=threshold)


print(my_table)
