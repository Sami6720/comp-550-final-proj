import requests
from SemEval2017 import preprocessing_EN_data

Success = 0
Failure = 0


def translate_text(text, src_lang, dest_lang):
    url = "http://localhost:5002/translate"  # URL of your Flask API
    payload = {
        "text": text,
        "fromCode": src_lang,
        "toCode": dest_lang
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.text
        else:
            print("Error Return")
            return None  # Return None in case of an error response
    except Exception as e:
        print(e)
        return None  # Return None in case of a translation failure


from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm


def parallel_translate(texts, src_lang, dest_lang):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(translate_text, text, src_lang, dest_lang) for text in texts]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results


def translate_and_save_df(df, src_lang, dest_lang, output_file_path):
    tqdm.pandas(desc="Translating")

    # Split the DataFrame into smaller chunks for parallel processing
    chunk_size = 100  # Adjust the chunk size based on your dataset and system capabilities
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

    # Process each chunk in parallel and collect the translated texts
    translated_texts = []
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        translated_texts.extend(parallel_translate(chunk['text'].tolist(), src_lang, dest_lang))

    # Assign the translated texts back to the DataFrame
    df['text'] = translated_texts

    # Drop rows where translation failed (if any)
    df = df.dropna(subset=['text'])

    # Save the translated DataFrame to a file
    df.to_csv(output_file_path, index=False)
    print(f"Translated DataFrame saved to {output_file_path}")


# Example usage
if __name__ == "__main__":
    SourceLan = "en"
    OutLan = "bn"
    # Load your DataFrame here
    #translate_text("hello", SourceLan, OutLan)

    df = preprocessing_EN_data.preprocess_data_EN("./Sentiment140/Sentiment140.csv", max_rows=70000)
    translate_and_save_df(df, SourceLan, OutLan, f"translated_{SourceLan}_{OutLan}.csv")

