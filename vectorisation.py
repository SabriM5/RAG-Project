import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import glob
import os
import torch

INDEX_NAME = "openfoodfacts"
ES_HOST = "http://localhost:9200"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def run_vectorization():
    try:
        es = Elasticsearch(ES_HOST, request_timeout=60)
        info = es.info()
        print(f"Connecté à Elasticsearch version {info['version']['number']}")
    except Exception as e:
        print(f"ERREUR: Impossible de joindre Elasticsearch. Vérifiez Docker ! ({e})")
        return
    
    if es.indices.exists(index=INDEX_NAME):
        print(f" Suppression de l'ancien index '{INDEX_NAME}' pour reset...")
        es.indices.delete(index=INDEX_NAME)
        print(f"Index '{INDEX_NAME}' supprimé.")

    
    model = SentenceTransformer(MODEL_NAME)
    
    parquet_files = glob.glob("data/staging_data/*.parquet")
    
    if not parquet_files:
        print("ERREUR: Aucun fichier Parquet trouvé dans data/staging_data/ !")
        print("Avez-vous bien lancé le script 01_ingestion.py avant ?")
        return

    print(f"Début de l'indexation de {len(parquet_files)} fichiers...")

    for file_path in parquet_files:
        print(f"Traitement de : {os.path.basename(file_path)}")
        
        df = pd.read_parquet(file_path)
        
        texts = df['ingredients_text'].astype(str).tolist()
        
        print(f"   -> Calcul des vecteurs pour {len(texts)} produits...")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        

        print("   -> Envoi vers la base de données...")
        actions = []
        for i, row in df.iterrows():

            doc = {
                "_index": INDEX_NAME,
                "_source": {
                    "product_name": row['product_name'],
                    "ingredients_text": row['ingredients_text'],
                    "nutriscore_grade": row['nutriscore_grade'],
                    "vector_embedding": embeddings[i].tolist() 
                }
            }
            actions.append(doc)
            
        success, failed = helpers.bulk(es, actions, stats_only=True)
        print(f"   -> Succès : {success} documents indexés.")

    print("\nTERMINE ! Votre moteur de recherche est prêt.")

if __name__ == "__main__":
    run_vectorization()