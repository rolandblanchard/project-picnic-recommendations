import json
import logging
import sys
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from mean_embedding_vectorizer import MeanEmbeddingVectorizer
from tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer
logging.basicConfig(filename='recommender_model.log', level=logging.DEBUG)

def get_recommendations(N, scores, recipes):
    """
    Rank scores and output a pandas data frame containing all the details of the top N recipes.
    :param scores: list of cosine similarities
    """
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    recommendation = pd.DataFrame(columns=["recipe_id", "score"])
    count = 0
    for i in top:
        recommendation.at[count, "recipe_id"] = recipes[i]["id"]
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation.to_json()


def get_recs(ingredients, N=5, mean=False):
    """
    Get the top N recipe recommendations.
    :param ingredients: comma-separated string listing ingredients
    :param N: number of recommendations
    :param mean: False if using tfidf weighted embeddings, True if using simple mean
    """
    logging.debug("Loading Word2Vec model...")
    model = Word2Vec.load("model_cbow.bin")

    logging.debug("Loading processed recipes...")
    with open("processed_recipes.json") as f:
        data = json.load(f)

    logging.debug("Processing input ingredients...")
    corpus = [rec["ingredients"] for rec in data]

    logging.debug("Generating document vectors...")
    if mean:
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
    else:
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]

    input = ingredients.split(",")
    logging.debug(f"Calculating embeddings for input: {input}")
    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    logging.debug("Calculating cosine similarities...")
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)

    logging.debug("Getting recipe recommendations...")
    recommendations = get_recommendations(N, scores, data)
    return recommendations


if __name__ == "__main__":
    input_ingredients = sys.argv[1]  # Get input from command-line arguments
    logging.debug(f"Received input ingredients: {input_ingredients}")
    recommendations = get_recs(input_ingredients)
    logging.debug(recommendations)
    print(recommendations)  # Add this line
 
