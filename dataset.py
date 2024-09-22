import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents(filename="data.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_encoder(model_name='all-MiniLM-L6-v2'):
    encoder_model = SentenceTransformer(model_name)
    return encoder_model

def encode_docs(corpus_of_documents, encoder_model):
    return encoder_model.encode(corpus_of_documents)

def get_recomendation(query, encoder_model, doc_embeddings, corpus_of_documents, print_results=False):
    query_embedding = encoder_model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)

    indexed = list(enumerate(similarities[0]))
    sorted_index = sorted(indexed, key=lambda x: x[1], reverse=True)

    recommended_documents = []
    for value, score in sorted_index:
        if print_results:
            formatted_score = "{:.2f}".format(score)
            print(f"{formatted_score} => {corpus_of_documents[value]}")
        if score > 0.3:
            recommended_documents.append(corpus_of_documents[value])
            
    return recommended_documents
