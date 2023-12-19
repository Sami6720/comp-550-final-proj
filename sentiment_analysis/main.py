import nltk
from ML_classifier import SentimentClassifier, load_and_split_dataset


nltk.download('stopwords')
# Load stopwords
# stop_words = set(stopwords.words('english'))
"""
def remove_stopwords(words):
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words
"""

import torch

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model_performance(model, data_loader):
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            for i in range(inputs.size(0)):
                input_vector = inputs[i]
                true_label = labels.item() if labels.dim() == 0 else labels[i].item()
                predicted_label = model.predict(input_vector)
                all_predictions.append(predicted_label)
                all_true_labels.append(true_label)

    # Calculate metrics for each class
    precision_per_class = precision_score(all_true_labels, all_predictions, average=None, labels=[0, 1, 2])
    recall_per_class = recall_score(all_true_labels, all_predictions, average=None, labels=[0, 1, 2])
    f1_per_class = f1_score(all_true_labels, all_predictions, average=None, labels=[0, 1, 2])
    # Round the metrics to two decimal places
    precision_per_class = [round(p, 2) for p in precision_per_class]
    recall_per_class = [round(r, 2) for r in recall_per_class]
    f1_per_class = [round(f1, 2) for f1 in f1_per_class]
    # Return the metrics for each class
    return precision_per_class, recall_per_class, f1_per_class


def evaluate_accuracy(model, data_loader):
    """
    Evaluate the accuracy of the model on the given dataset using the model's predict function.

    :param model: The trained model.
    :param data_loader: DataLoader containing the dataset for evaluation.
    :return: Accuracy of the model on the dataset.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # If labels is a tensor of size 0, unsqueeze to make it 1D
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            for i in range(inputs.size(0)):  # Iterate through each instance in the batch
                input_vector = inputs[i]  # Get the input vector for the instance
                true_label = labels[i]  # Get the corresponding true label
                predicted_label = model.predict(input_vector)  # Use the model's predict function
                correct += (predicted_label == true_label.item())  # Compare predicted and true labels
                total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy
def convert_to_dict(scores):
    """
    Convert an array of scores to a dictionary with keys for negative, neutral, and positive.

    Args:
    scores (list or array): A list or array containing three numerical values.

    Returns:
    dict: A dictionary with the sentiment scores.
    """
    sentiment_labels = ['negative', 'neutral', 'positive']
    return {label: score for label, score in zip(sentiment_labels, scores)}

if __name__ == "__main__":
    optimal={'lr': 0.00012398654050400074, 'num_layers': 4, 'layer_reduction': 0.7501535615120707, 'batch_size': 128, 'epochs': 30}
    ENTrain, ENTest = load_and_split_dataset("./Tensor_Datasets/Bert_EN_No_Emojis.pt", 0.8)
    ENModel = SentimentClassifier(input_size=768, num_layers=optimal["num_layers"], layer_reduction=optimal['layer_reduction'], learning_rate=optimal['lr'])
    ENModel.fit(ENTrain, epochs=optimal["epochs"], batch_size=optimal['batch_size'], verbose=True)
    print(evaluate_accuracy(ENModel, ENTrain))
    print(evaluate_accuracy(ENModel, ENTest))
    print(evaluate_model_performance(ENModel, ENTrain))
    print(evaluate_model_performance(ENModel, ENTest))

    """
    russian_data = preprocess_data_RU(file_path="./rusentitweet/rusentitweet_full.csv")
    english_data = preprocess_data_EN(file_path="./Sentiment140/Sentiment140.csv", max_rows=10000)
    RUTrain, RUTest = prepare_data(english_data, get_mbert_base_embedding, 0.8,verbose=True)
    RuModel = SentimentClassifier(input_size=768, num_layers=4, layer_reduction=0.5, learning_rate=0.001)
    RuModel.fit(RUTrain, epochs=10, batch_size=64,verbose=True)
    print(evaluate_accuracy(RuModel, RUTest))
    """
