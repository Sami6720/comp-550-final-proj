import nltk
import pandas as pd
import spacy
from nltk import PorterStemmer, SnowballStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from nltk.tokenize import word_tokenize
# Initialize the spaCy model
#spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
stemmer = SnowballStemmer("english")
def stem_df(df):
    """
    Stem English text in a specified column of a DataFrame using Snowball Stemmer.

    :param df: DataFrame to process.
    :return: DataFrame with stemmed text in the specified column.
    """
    # Initialize the English Snowball Stemmer


    tqdm.pandas(desc="Stemming")
    def stem_text(text):
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    df['text'] = df['text'].progress_apply(stem_text)
    return df
def lemmatize_df(df):
    """
    Lemmatize text in a specified column of a DataFrame.

    :param df: DataFrame to process.
    :return: DataFrame with lemmatized text in the specified column.
    """
    tqdm.pandas(desc="Lemmatizing")
    df['text'] = df['text'].progress_apply(lambda text: ' '.join([token.lemma_ for token in nlp(text)]))
    return df

def remove_stopwords_df(df):
    """
    Remove stopwords from text in a specified column of a DataFrame.

    :param df: DataFrame to process.
    :return: DataFrame with text having stopwords removed in the specified column.
    """
    tqdm.pandas(desc="Removing Stopwords")
    df['text'] = df['text'].progress_apply(lambda text: ' '.join([token.text for token in nlp(text) if token.text.lower() not in STOP_WORDS]))
    return df


def preprocess_data_EN(file_path, lemmatize=False, remove_stopwords=False, stemming=False):
    """
    Preprocess the English dataset from the given file path.
    """
    if stemming and lemmatize:
        raise ValueError("Cannot stem and lemmatize at the same time.")
    # Load the data
    data = pd.read_csv(file_path, encoding='latin1')

    # Drop rows with any missing elements
    data.dropna(inplace=True)

    # Convert the labels to 0, 1, 2
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    data['label'] = data['label'].map(label_map)

    # Preprocess the text
    if remove_stopwords:
        data = remove_stopwords_df(data)
    if lemmatize:
        data = lemmatize_df(data)
    if stemming:
        data = stem_df(data)
    return data[["text","label"]]


if __name__=="__main__":
    data = preprocess_data_EN('SemEval2017-task4.csv',lemmatize=True)
    print(data)