from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def get_embedding(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


# Database of sentences
database = [
    "This is a sample sentence.",
    "This is another example.",
    "Yet another sentence.",
    "This is the last sentence in the database.",
    "This is the last sensdsdfence in the database.",
    "adsThis is the last sentence in the database.",
    "I love watching anime football matches.",
    "The anime series about football is incredibly exciting.",
    "In the anime, the football players have amazing skills.",
    "The main character in the anime is a talented football player.",
    "Anime football matches often have intense action and dramatic moments.",
    "I can't wait for the next episode of the anime football series.",
    "The animation in the anime football scenes is stunning.",
    "Football tournaments in anime are always thrilling to watch.",
    "The anime perfectly captures the spirit and passion of football.",
    "The anime football team is determined to win the championship.",
    "The rivalry between the anime football clubs is fierce.",
    "The coach in the anime series is known for his innovative strategies.",
]


database_embeddings = get_embedding(database).numpy() # Get embeddings for all sentences in the database


def search(query):
    # Get embedding for the query
    query_embedding = get_embedding([query]).numpy()

    # Compute cosine similarity between the query and each sentence in the database
    similarities = cosine_similarity(query_embedding, database_embeddings)

    # Get the indices of the sentences sorted by similarity
    ranked_indices = similarities[0].argsort()[::-1]

    # Get the top 5 most similar sentences
    top_sentences = [database[i] for i in ranked_indices[:5]]

    return top_sentences


query = "I the next episode football clubs" # Assume you have a query


results = search(query) # Call the search function with your query

# Print the results
for sentence in results:
    print(sentence)
