import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch
from openai import OpenAI

# --- CONFIGURATION ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "openfoodfacts"


RETRIEVER_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

LLM_CLIENT = OpenAI(
    base_url='http://localhost:11434/v1', 
    api_key='ollama', 
)
LLM_MODEL_NAME = "qwen2.5:3b" 

# LLM_CLIENT = OpenAI(api_key="TA-CLE-OPENAI-ICI")
# LLM_MODEL_NAME = "gpt-3.5-turbo"

def get_context(user_query):
    """
    Récupère les documents pertinents et les re-trie.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    es = Elasticsearch(ES_HOST)
    retriever = SentenceTransformer(RETRIEVER_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    

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

    
    cross_inp = [[user_query, f"{h['_source']['product_name']} {h['_source']['ingredients_text']}"] for h in hits]
    
    scores = reranker.predict(cross_inp)
    
    for i, h in enumerate(hits):
        h['_score_reranked'] = scores[i]
        
    sorted_hits = sorted(hits, key=lambda x: x['_score_reranked'], reverse=True)
    
    return sorted_hits[:5]

def generate_answer(user_query, context_docs):
    if not context_docs:
        return "Désolé, je n'ai trouvé aucun produit correspondant."

    context_text = ""
    for doc in context_docs:
        s = doc['_source']
        context_text += f"""
Produit:
- Nom: {s['product_name']}
- Nutriscore: {s['nutriscore_grade']}
- Ingrédients: {s['ingredients_text']}
"""
        
    system_prompt = ("""
                     You are a nutrition assistant.
Answer using ONLY the provided context.
Explain briefly how you found the answer.
If the answer is not in the context, say "Je ne sais pas".
Answer in French.
"""
    )

    user_message = f"Context:\n{context_text}\n\nQuestion: {user_query}"

    print("Generation de la réponse en cours...")
    
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1)
    
    return response.choices[0].message.content

def run_rag():
    print("=== ASSISTANT OPENFOODFACTS ===")
    while True:
        query = input("\nVotre question (ou 'q' pour quitter) : ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        print("Recherche des produits...")
        top_docs = get_context(query)
        
        print("L'IA réfléchit...")
        answer = generate_answer(query, top_docs)
        
        print("\n" + "-"*50)
        print("RÉPONSE :")
        print(answer)
        print("-"*50)

if __name__ == "__main__":
    run_rag()