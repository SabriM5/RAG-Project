import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch
from openai import OpenAI

# --- CONFIGURATION ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "openfoodfacts"

# Modèles (Doivent correspondre à ce que tu as indexé)
# RETRIEVER : Rapide (Doit être celui utilisé dans vectorisation.py)
RETRIEVER_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# RERANKER : Précis (Pour trier le top 50)
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# CONFIGURATION LLM (Ici configuré pour Ollama en local)
LLM_CLIENT = OpenAI(
    base_url='http://localhost:11434/v1', # L'adresse par défaut d'Ollama
    api_key='ollama', # Ollama ne demande pas de vraie clé
)
LLM_MODEL_NAME = "tinyllama" # Assure-toi d'avoir fait 'ollama pull mistral'

# Si tu veux utiliser ChatGPT à la place, décommente ça :
# LLM_CLIENT = OpenAI(api_key="TA-CLE-OPENAI-ICI")
# LLM_MODEL_NAME = "gpt-3.5-turbo"

def get_context(user_query):
    """
    Récupère les documents pertinents et les re-trie.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Connexion et Chargement
    es = Elasticsearch(ES_HOST)
    retriever = SentenceTransformer(RETRIEVER_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    
    # 2. Retrieval (Elasticsearch)
    query_vector = retriever.encode(user_query)
    search_query = {
        "knn": {
            "field": "vector_embedding",
            "query_vector": query_vector.tolist(),
            "k": 50, 
            "num_candidates": 500
        },
        "_source": ["product_name", "ingredients_text", "nutriscore_grade"]
    }
    
    response = es.search(index=INDEX_NAME, body=search_query)
    hits = response['hits']['hits']
    
    if not hits:
        return []

    # 3. Reranking (Cross-Encoder)
    # On prépare les paires [Question, Contenu Produit]
    cross_inp = [[user_query, f"{h['_source']['product_name']} {h['_source']['ingredients_text']}"] for h in hits]
    
    scores = reranker.predict(cross_inp)
    
    # On ajoute le score et on trie
    for i, h in enumerate(hits):
        h['_score_reranked'] = scores[i]
        
    sorted_hits = sorted(hits, key=lambda x: x['_score_reranked'], reverse=True)
    
    # On garde le TOP 5
    return sorted_hits[:5]

def generate_answer(user_query, context_docs):
    if not context_docs:
        return "Désolé, je n'ai trouvé aucun produit correspondant."

    # 1. On prépare le contexte (plus propre)
    context_text = ""
    for doc in context_docs:
        s = doc['_source']
        context_text += f"Product: {s['product_name']} | Nutriscore: {s['nutriscore_grade']} | Ingredients: {s['ingredients_text'][:200]}...\n"

    # 2. Prompt simplifié (Les petits modèles préfèrent les instructions en Anglais)
    # On lui dit explicitement de répondre en Français.
    system_prompt = (
        "You are a helpful nutrition assistant. "
        "Answer the user question based ONLY on the Context provided below. "
        "If the answer is not in the Context, say 'I don't know'. "
        "Answer in French."
    )

    user_message = f"Context:\n{context_text}\n\nQuestion: {user_query}"

    print("Generation de la réponse en cours...")
    
    # 3. Appel
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1 # On baisse la température pour qu'il soit plus factuel/rigoureux
    )
    
    return response.choices[0].message.content

def run_rag():
    print("=== ASSISTANT OPENFOODFACTS ===")
    while True:
        query = input("\nVotre question (ou 'q' pour quitter) : ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        # Etape 1 : Récupération
        print("Recherche des produits...")
        top_docs = get_context(query)
        
        # Etape 2 : Génération
        print("L'IA réfléchit...")
        answer = generate_answer(query, top_docs)
        
        print("\n" + "-"*50)
        print("RÉPONSE :")
        print(answer)
        print("-"*50)

if __name__ == "__main__":
    run_rag()