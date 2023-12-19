import csv

import pandas as pd


def convert_to_csv(input_file_path, output_file_path):
    """
    Converts a text file into a CSV file with columns 'id', 'text', and 'label'.

    Args:
    input_file_path (str): The file path to the input text file.
    output_file_path (str): The file path where the output CSV file will be saved.
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:

        writer = csv.writer(outfile)
        # Write the header
        writer.writerow(['id','label', 'text'])

        for line in infile:
            # Assuming each field in your text file is separated by a tab
            fields = line.strip().split('\t')
            if len(fields) == 3:
                writer.writerow(fields)

    print(f"Data has been written to {output_file_path}")


def filter_and_save(input_file_path, output_file_path):
    """
    Reads a CSV file, filters out rows with missing elements in any column, and saves the cleaned data to a new CSV file.

    Args:
    input_file_path (str): The file path to the input CSV file.
    output_file_path (str): The file path where the output CSV file will be saved.
    """
    # Load the dataset
    df = pd.read_csv(input_file_path)

    # Ensure the correct column names based on the provided data
    df.columns = ['id', 'label', 'text']

    # Check for rows with missing elements
    missing_data = df.isnull().any(axis=1)

    # Filter out rows with missing data
    df_clean = df[~missing_data]

    # Save the cleaned dataframe to a new CSV file
    df_clean.to_csv(output_file_path, index=False)

    return df_clean.shape

if __name__ == "__main__":
    convert_to_csv("SemEval2017-task4-dev.subtask-A.english.INPUT.txt", "SemEval2017-task4.csv")
    filter_and_save("SemEval2017-task4.csv", "SemEval2017-task4.csv")