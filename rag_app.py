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
LLM_MODEL_NAME = "mistral" # Assure-toi d'avoir fait 'ollama pull mistral'

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
    """
    Construit le prompt et interroge le LLM.
    """
    if not context_docs:
        return "Désolé, je n'ai trouvé aucun produit correspondant dans la base."

    # 1. Construction du contexte (le texte que le LLM va lire)
    context_text = ""
    for doc in context_docs:
        source = doc['_source']
        context_text += f"- Produit : {source['product_name']}\n"
        context_text += f"  Nutriscore : {source['nutriscore_grade']}\n"
        context_text += f"  Ingrédients : {source['ingredients_text']}\n\n"

    # 2. Le Prompt Système (Les instructions pour l'IA)
    system_prompt = """
    Tu es un assistant expert en nutrition utilisant les données OpenFoodFacts.
    Ta mission est de répondre à la question de l'utilisateur en utilisant UNIQUEMENT les informations contextuelles fournies ci-dessous.
    
    Règles :
    1. Si la réponse n'est pas dans le contexte, dis "Je ne sais pas basé sur les documents fournis".
    2. Cite le nom des produits que tu recommandes.
    3. Sois synthétique et utile.
    """

    # 3. Appel au LLM
    print("Generation de la réponse en cours...")
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexte:\n{context_text}\n\nQuestion: {user_query}"}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def run_rag():
    print("=== ASSISTANT OPENFOODFACTS ===")
    while True:
        query = input("\nVotre question (ou 'q' pour quitter) : ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        # Etape 1 : Récupération
        print("🔍 Recherche des produits...")
        top_docs = get_context(query)
        
        # Etape 2 : Génération
        print("🤖 L'IA réfléchit...")
        answer = generate_answer(query, top_docs)
        
        print("\n" + "-"*50)
        print("RÉPONSE :")
        print(answer)
        print("-"*50)

if __name__ == "__main__":
    run_rag()