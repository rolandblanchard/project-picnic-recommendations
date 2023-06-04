import json
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from .mean_embedding_vectorizer import MeanEmbeddingVectorizer
from .tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer


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
    return recommendation


def get_recs(ingredients, N=5, mean=False):
    """
    Get the top N recipe recommendations.
    :param ingredients: comma-separated string listing ingredients
    :param N: number of recommendations
    :param mean: False if using tfidf weighted embeddings, True if using simple mean
    """
    model = Word2Vec.load("model_cbow.bin")
    model.init_sims(replace=True)

    with open("processed_recipes.json") as f:
        data = json.load(f)

    corpus = [rec["ingredients"] for rec in data]

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
    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)

    recommendations = get_recommendations(N, scores, data)
    return recommendations


if __name__ == "__main__":
    input_ingredients = "chicken, onion, rice, seaweed, sesame, shallot, soy, spinach, star, tofu"
    recommendations = get_recs(input_ingredients)
    print(recommendations)
