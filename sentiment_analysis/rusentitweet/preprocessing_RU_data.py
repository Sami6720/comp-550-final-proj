from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import emoji
import spacy
from nltk import SnowballStemmer
from spacy.lang.ru.stop_words import STOP_WORDS
import nltk
from nltk.stem.porter import *
from tqdm import tqdm

from SemEval2017.preprocessing_EN_data import preprocess_data_EN
from translate import translate_text

# Download the Russian language model
#spacy.cli.download("ru_core_news_sm")
nlp = spacy.load('ru_core_news_sm')
stemmer = SnowballStemmer("russian")

def stem_russian_df(df):
    """
    Stem Russian text in a specified column of a DataFrame.

    :param df: DataFrame to process.
    :return: DataFrame with stemmed Russian text in the specified column.
    """

    def stem_text(text):
        # Tokenize the text by splitting on whitespace
        tokens = text.split()
        # Stem each token
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    df['text'] = df['text'].apply(stem_text)
    return df

def lemmatize_df(df):
    """
    Lemmatize text in a specified column of a DataFrame.

    :param df: DataFrame to process.
    :return: DataFrame with lemmatized text in the specified column.
    """
    df['text'] = df['text'].apply(lambda text: ' '.join([token.lemma_ for token in nlp(text)]))
    return df


def parallel_translate(texts, src_lang, dest_lang):
    """
    Translates a list of texts in parallel using the translate_text function.

    :param texts: List of texts to be translated.
    :param src_lang: The source language code.
    :param dest_lang: The destination language code.
    :return: List of translated texts.
    """
    with ThreadPoolExecutor() as executor:
        # Start translation tasks and collect futures
        futures = [executor.submit(translate_text, text, src_lang, dest_lang) for text in texts]

        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results

def translate_df(df, source_file_path="../Sentiment140/Sentiment140.csv", dest_lang='ru'):
    tqdm.pandas(desc="Translating")
    # Load and preprocess the English dataset
    translated_df = preprocess_data_EN(source_file_path)

    # Split the DataFrame into smaller chunks for parallel processing
    chunk_size = 100  # You can adjust the chunk size based on your dataset and system capabilities
    chunks = [translated_df[i:i + chunk_size] for i in range(0, translated_df.shape[0], chunk_size)]

    # Process each chunk in parallel
    translated_texts = []
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        translated_texts.extend(parallel_translate(chunk['text'].tolist(), "en", dest_lang))

    # Assign the translated texts back to the DataFrame
    translated_df['text'] = translated_texts

    # Drop rows where translation failed (if any)
    translated_df = translated_df.dropna(subset=['text'])

    # Concatenate the translated dataframe with the original Russian dataframe
    return pd.concat([df, translated_df], ignore_index=True)
def remove_stopwords_df(df):
    """
    Remove stopwords from text in a specified column of a DataFrame.

    :param df: DataFrame to process.
    :return: DataFrame with text having stopwords removed in the specified column.
    """
    df['text'] = df['text'].apply(lambda text: ' '.join([token.text for token in nlp(text) if token.text.lower() not in STOP_WORDS]))
    return df




def remove_emojis_from_df(df):
    """
    Remove all emojis from a specified column in a DataFrame.

    :param df: DataFrame to process.
    :return: DataFrame with emojis removed from the specified column.
    """
    df['text'] = df['text'].apply(lambda text: emoji.replace_emoji(text, replace=''))
    return df


def replace_emojis_in_df(df):
    """
    Replace all emojis in a specified column in a DataFrame with their textual description.

    :param df: DataFrame to process.
    :return: DataFrame with emojis replaced by their textual descriptions in the specified column.
    """
    df['text'] = df['text'].apply(lambda text: emoji.demojize(text, delimiters=("", "")))
    return df


def remove_unwanted_labels(df, labels_to_remove):
    """
    Remove rows from the DataFrame where the label column contains any of the labels specified in labels_to_remove.

    :param df: DataFrame from which to remove rows.
    :param labels_to_remove: List of labels to remove.
    :return: DataFrame with the specified labels removed.
    """
    return df[~df['label'].isin(labels_to_remove)]


def convert_label_to_numeric(df):
    """
    Convert the sentiment labels in the column ''label'' to numeric values.
    Positive becomes 1, Neutral becomes 0, and Negative becomes -1.

    :param df: DataFrame containing the sentiment labels.
    :return: DataFrame with converted numeric sentiment values.
    """
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'speech': 2}
    df['label'] = df['label'].map(label_map)
    return df

def average_words_per_sentence(dataframe, text_column):
    total_words = 0
    max_val=0
    for sentence in dataframe[text_column]:
        words = sentence.split()  # This splits the sentence into words based on spaces.
        max_val=max(len(words),max_val)
        total_words += len(words)
    average_words = total_words / len(dataframe)
    return average_words,max_val


# Example usage:
# Assuming you have a DataFrame 'data' with a column 'label'
# data = convert_label_to_numeric(data, 'label')
def preprocess_data_RU(file_path, remove_emojis=False, replace_emojis=True, include_speech=False, lemmatize=False, remove_stopwords=False,augment_data_translate=False,stemming=False):
    if replace_emojis and remove_emojis:
        raise ValueError("replace_emojis and remove_emojies cannot both be True")
    if lemmatize and stemming:
        raise ValueError("lemmatize and Stemming cannot both be True")
    # Load the data
    data = pd.read_csv(file_path)
    if not include_speech:
        data = remove_unwanted_labels(data, ['speech', 'skip'])
    else:
        data = remove_unwanted_labels(data, ['skip'])
    # Convert the labels to numeric values
    data = convert_label_to_numeric(data)
    if augment_data_translate:
        data = translate_df(data)
    # Replace Emojis
    if replace_emojis:
        data = replace_emojis_in_df(data)
    elif remove_emojis:
        data = remove_emojis_from_df(data)
    # Remove stopwords
    if remove_stopwords:
        data = remove_stopwords_df(data)
    # Lemmatize of words
    if lemmatize:
        data = lemmatize_df(data)
    if stemming:
        data = stem_russian_df(data)

    return data[['text', 'label']]


if __name__ == "__main__":
    gola=preprocess_data_RU("rusentitweet_full.csv",remove_emojis=True,replace_emojis=False,stemming=True)
    print(gola)
    #print(average_words_per_sentence(preprocess_data_RU("rusentitweet_full.csv"),"text"))

