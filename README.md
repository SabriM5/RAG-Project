# Assistant Nutritionnel RAG - OpenFoodFacts

Ce projet implémente un système de **RAG (Retrieval-Augmented Generation)** permettant d'interroger la base de données **OpenFoodFacts** en langage naturel. 

Il utilise une approche "Big Data" avec **Elasticsearch** pour la recherche vectorielle et un **LLM local (Ollama)** pour générer des réponses synthétiques et sourcées.

## Architecture

Le système suit un pipeline RAG classique optimisé pour tourner sur une machine standard (Low Resource) :

1.  **Ingestion (ETL) :** Nettoyage du dataset CSV OpenFoodFacts et conversion en fichiers Parquet optimisés.
2.  **Vectorisation :** Encodage des produits (noms + ingrédients) via le modèle `paraphrase-multilingual-MiniLM-L12-v2`.
3.  **Indexation :** Stockage des vecteurs et métadonnées dans **Elasticsearch**.
4.  **Recherche (Retrieval) :** Recherche sémantique (KNN) pour trouver les produits pertinents.
5.  **Génération (Generation) :** Un LLM (`Qwen 2.5` ou `Phi-3`) répond à l'utilisateur en utilisant uniquement les produits trouvés.

## Prérequis

* **Docker** (pour Elasticsearch & Kibana)
* **Python 3.10+**
* **Ollama** (installé localement pour le LLM)
* **RAM :** Minimum 8 Go recommandé (4 Go possible avec le modèle Qwen Nano).

**Stack Technique**

* **Langage** : Python
* **Base Vectorielle** : Elasticsearch 8.11
* **Embedding Model** : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
* **LLM** : Qwen2.5-0.5b via Ollama (Inférence Locale)
* **Format de données** : Apache Parquet

## Installation

### 1. Cloner le projet et préparer l'environnement

```bash
git clone <votre-repo>
cd ProjetRAG
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Mac/Linux :
source .venv/bin/activate

pip install pandas sentence-transformers elasticsearch openai torch pyarrow

# Créer le réseau
docker network create reseau-rag

# Lancer Elasticsearch (Port 9200)
docker run -d --name es01 --net reseau-rag -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" docker.elastic.co/elasticsearch/elasticsearch:8.11.1

# (Optionnel) Lancer Kibana pour visualiser les données (Port 5601)
docker run -d --name kib01 --net reseau-rag -p 5601:5601 -e "ELASTICSEARCH_HOSTS=http://es01:9200" docker.elastic.co/kibana/kibana:8.11.1

# Pour les PC avec peu de RAM (< 8Go)
ollama pull qwen2.5:0.5b

# Pour les PC plus puissants (> 8Go)
ollama pull phi3
```

**Auteur**
Projet réalisé dans le cadre du cours de Big Data / Langages Fonctionnels.
