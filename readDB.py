import pickle
from pprint import pprint 
# Open file in byte reading mode 
with open('database.pkl', 'rb') as f:
    data = pickle.load(f)

# Read data
for item in data['documents']:
    pprint(item)