import ollama
import numpy as np

def get_embeddings(text):
    # Use the model to generate embeddings
    return ollama.embed(model='nomic-embed-text', input=text).embeddings[0]


def get_similarity(text_a, text_b):
    if isinstance(text_b, list):
        embeddings = [np.array(get_embeddings(t)) for t in text_b]
        similarities = [np.dot(np.array(get_embeddings(text_a)), e) / (np.linalg.norm(np.array(get_embeddings(text_a))) * np.linalg.norm(e)) for e in embeddings]
        return np.mean(similarities)
    else:
        embedding_a = np.array(get_embeddings(text_a))
        embedding_b = np.array(get_embeddings(text_b))
        similarity = np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
        return similarity



if __name__ == "__main__":
    print(get_embeddings("Hello, world!"))
    print(get_similarity("Hello, world!", "Goodbye, world!"))
    print(get_similarity("Hello, Bob!", "Hello, world!"))
    print(get_similarity("Hello, Bob!", "Goodbye, world!"))
    print(get_similarity("Hello, Bob!", ["Hello, world!", "Goodbye, world!"]))
    
