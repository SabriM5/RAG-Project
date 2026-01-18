import pandas as pd
import os
import shutil

CSV_PATH = r"C:/Users/sabri/Downloads/ESEO/S9/LangagesFonctionnel/en.openfoodfacts.org.productsfill.csv"

OUTPUT_DIR = "data/staging_data"
CHUNK_SIZE = 50000 

def clean_and_convert():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Lecture du fichier : {CSV_PATH}")
    
    # 2. Lecture par morceaux
    chunk_iterator = pd.read_csv(
        CSV_PATH, 
        sep='\t', 
        chunksize=CHUNK_SIZE, 
        on_bad_lines='skip',
        low_memory=False, 
        dtype={
            "code": str,
            "product_name": str,
            "ingredients_text": str,
            "nutriscore_grade": str
        }
    )

    count = 0
    total_products = 0

    for i, df_chunk in enumerate(chunk_iterator):
        cols_to_keep = ["code", "product_name", "ingredients_text", "nutriscore_grade"]
        
        available_cols = [c for c in cols_to_keep if c in df_chunk.columns]
        df_clean = df_chunk[available_cols].copy() # .copy() évite des bugs de mémoire

        for col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)

        df_clean = df_clean[df_clean["ingredients_text"] != "nan"]
        df_clean = df_clean[df_clean["product_name"] != "nan"]
        
        if "ingredients_text" in df_clean.columns:
            df_clean = df_clean[df_clean["ingredients_text"].str.len() > 20]

        if df_clean.empty:
            continue

        output_file = os.path.join(OUTPUT_DIR, f"part_{i}.parquet")
        df_clean.to_parquet(output_file, index=False)
        
        count += 1
        total_products += len(df_clean)
        print(f"Paquet {i} traité -> {len(df_clean)} produits sauvegardés.")

    print(f"Terminé ! {total_products} produits sauvegardés dans '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    clean_and_convert()