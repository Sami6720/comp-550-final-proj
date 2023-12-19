import os
from ML_classifier import create_tensor_dataset, get_mbert_base_embedding, get_mbart_large_embedding, \
    create_case_conscious_tensor_dataset
from rusentitweet.preprocessing_RU_data import preprocess_data_RU
from SemEval2017.preprocessing_EN_data import preprocess_data_EN
from concurrent.futures import ProcessPoolExecutor

datasets_RU = {
    'RU Base': {'remove_emojis': False, 'replace_emojis': True, 'include_speech': False, 'lemmatize': False,
                'remove_stopwords': False},
    'RU Lemmatize': {'remove_emojis': False, 'replace_emojis': True,
                     'include_speech': False, 'lemmatize': True,
                     'remove_stopwords': False,
                     "stemming": False},
    'RU Stemming': {'remove_emojis': False,
                    'replace_emojis': True,
                    'include_speech': False,
                    'lemmatize': False,
                    'remove_stopwords': True, "stemming": True},
    'RU Stemming + StopWords': {'remove_emojis': False,
                                'replace_emojis': True,
                                'include_speech': False,
                                'lemmatize': False,
                                'remove_stopwords': True
                                , "stemming": True},
    'RU Lemmatize + StopWords': {'remove_emojis': False,
                                 'replace_emojis': True,
                                 'include_speech': False,
                                 'lemmatize': True,
                                 'remove_stopwords': True,
                                 "stemming": False},
}
datasets_EN = {
    'EN Base': {'lemmatize': False, 'remove_stopwords': False, "stemming": False},
    'EN Stemming': {'lemmatize': False, 'remove_stopwords': False, "stemming": True},
    'EN Lemmatize': {'lemmatize': True, 'remove_stopwords': False, "stemming": False},
    'EN Stemming + Stopwords': {'lemmatize': False, 'remove_stopwords': True, "stemming": True},
    'EN Lemmatize + Stopwords': {'lemmatize': True, 'remove_stopwords': True, "stemming": False}
}


def generate_filename(key, vectorizer, Russian_Tagger=False):
    """
    Generates a filename for a tensor file based on the given key.
    It includes 'Bert_' or 'Bart_' based on the vectorizer function used.
    It replaces spaces with underscores, except where '+' is present.

    :param key: The key from the datasets dictionary.
    :param vectorizer: The vectorizer function used for processing.
    :return: A string representing the filename.
    """
    # Check the name of the vectorizer function for 'bert' or 'bart'
    vectorizer_name = vectorizer.__name__.lower()
    if 'bert' in vectorizer_name:
        prefix = 'Bert_'
    elif 'bart' in vectorizer_name:
        prefix = 'Bart_'
    else:
        raise ValueError("Vectorizer function name must contain 'bert' or 'bart'")
    if Russian_Tagger:
        prefix += 'Case_Tagger_'
    # Replace spaces with underscores, preserve '+' as it is
    filename = key.replace(' ', '_').replace('_+_', '+') + '.pt'
    return prefix + filename


def process_dataset(db_filepath, key, params, preprocess_func, vectorizer_func, Russian_Tagger=False):
    # Preprocess data
    df = preprocess_func(db_filepath, **params)

    if Russian_Tagger:
        tensor_filename = generate_filename(key, vectorizer_func, Russian_Tagger)
        create_case_conscious_tensor_dataset(df, vectorizer_func, "./Tensor_Datasets/" + tensor_filename, verbose=True)
    else:
        # Create and save tensor dataset
        tensor_filename = generate_filename(key, vectorizer_func)
        create_tensor_dataset(df, vectorizer_func, "./Tensor_Datasets/" + tensor_filename, verbose=True)
    return f"{key} processing completed."


def process_and_save_datasets(db_filepath, datasets, preprocess_func, vectorizer_func, Russian_Tagger=False):
    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Using ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=num_cores - 2) as executor:
        futures = []
        for key, params in datasets.items():
            # Submit each dataset processing as a separate task
            # Note: Corrected the arguments to executor.submit
            future = executor.submit(process_dataset, db_filepath, key, params, preprocess_func, vectorizer_func,
                                     Russian_Tagger)
            futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            print(future.result())


if __name__ == "__main__":
    EN_Filepath = "./SemEval2017/SemEval2017-task4.csv"
    RU_Filepath = "rusentitweet/rusentitweet_full.csv"
    process_and_save_datasets(EN_Filepath, datasets_EN, preprocess_data_EN, get_mbert_base_embedding,
                              Russian_Tagger=False)
    process_and_save_datasets(EN_Filepath, datasets_EN, preprocess_data_EN, get_mbart_large_embedding,
                              Russian_Tagger=False)
    process_and_save_datasets(RU_Filepath, datasets_RU, preprocess_data_RU, get_mbert_base_embedding,
                              Russian_Tagger=False)
    process_and_save_datasets(RU_Filepath, datasets_RU, preprocess_data_RU, get_mbart_large_embedding,
                              Russian_Tagger=False)
    process_and_save_datasets(RU_Filepath, datasets_RU, preprocess_data_RU, get_mbert_base_embedding,
                              Russian_Tagger=True)
    process_and_save_datasets(RU_Filepath, datasets_RU, preprocess_data_RU, get_mbart_large_embedding,
                              Russian_Tagger=True)

