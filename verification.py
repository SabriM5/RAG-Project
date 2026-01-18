from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
INDEX_NAME = "openfoodfacts"
ES_HOST = "http://localhost:9200"

def inspect_vectors():

    print("Connexion à Elasticsearch...")
    es = Elasticsearch(ES_HOST)
    

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\n Tape un mot-clé (ex: 'tomate', 'chocolat') ou 'q' pour quitter : ")
        if query == 'q':
            break

        query_vector = model.encode(query)

        print("   ... Recherche dans les vecteurs ...")
        response = es.search(
            index=INDEX_NAME,
            body={
                "knn": {
                    "field": "vector_embedding",
                    "query_vector": query_vector.tolist(),
                    "k": 3,  
                    "num_candidates": 100
                },
                "_source": ["product_name", "ingredients_text"]
            }
        )

        print(f"\n--- RÉSULTATS POUR : '{query}' ---")
        hits = response['hits']['hits']
        
        if not hits:
            print(" Aucun résultat trouvé (Vecteurs vides ou index vide ?)")
            continue

        for i, hit in enumerate(hits):
            score = hit['_score']
            name = hit['_source']['product_name']
            ingredients = hit['_source']['ingredients_text']
            
            print(f"#{i+1} [Score: {score:.4f}]")
            print(f"   Produit : {name}")
            print(f"   Chunk (Ingrédients) : {ingredients[:150]}...") 
            print("-" * 40)

if __name__ == "__main__":
    inspect_vectors()