import requests
from bs4 import BeautifulSoup
import random
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

proxies_list = [

]

def get_random_proxy():
    return {'http': random.choice(proxies_list), 'https': random.choice(proxies_list)}

def get_price_from_example(tech_name, max_retries=5):
    search_url = f'?q={tech_name}'
    retries = 0

    while retries < max_retries:
        try:
            proxies = get_random_proxy()
            logging.info(f"Использование прокси: {proxies}")
            response = requests.get(search_url, proxies=proxies, timeout=10)

            if response.status_code != 200:
                logging.warning(f"Ошибка при выполнении запроса: {response.status_code}")
                retries += 1
                time.sleep(2)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            price_tag = soup.find('span', class_='price')

            if price_tag:
                price = price_tag.get_text(strip=True)
                logging.info(f"Цена для '{tech_name}' найдена: {price}")
                return price
            else:
                logging.warning(f"Цена для '{tech_name}' не найдена.")
                return None

        except requests.RequestException as e:
            logging.error(f"Исключение при запросе: {e}")
            retries += 1
            time.sleep(2)

    logging.error(f"Не удалось получить цену для '{tech_name}' после {max_retries} попыток.")
    return None



if __name__ == '__main__':
    tech_name = 'samsung s20'
    price = get_price_from_example(tech_name)
    if price:
        print(f"Цена для '{tech_name}': {price}")
    else:
        print(f"Не удалось получить цену для '{tech_name}'.")