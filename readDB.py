import pickle
from pprint import pprint 
# Открываем файл в режиме чтения байтов
with open('database.pkl', 'rb') as f:
    data = pickle.load(f)

# Выводим данные
for item in data['documents']:
    pprint(item)