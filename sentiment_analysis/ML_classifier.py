import os

from tqdm import tqdm
from transformers import MBartTokenizer, MBartModel
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from SemEval2017.preprocessing_EN_data import preprocess_data_EN
from tagger import case_tagger, append_list_to_tensor

torch.manual_seed(0)
# Initialize tokenizer and model

MBartTokenizer_ = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
MBartModel_ = MBartModel.from_pretrained('facebook/mbart-large-cc25')
MBertTokenizer_ = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
MBertModel_ = BertModel.from_pretrained('bert-base-multilingual-cased')
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MBartModel_ = MBartModel_.to(device)
MBertModel_ = MBertModel_.to(device)

class SentimentClassifier(nn.Module):
    def __init__(self, input_size=768, num_layers=4, layer_reduction=0.5, learning_rate=0.001):
        super(SentimentClassifier, self).__init__()

        # Ensure layer_reduction is between 0 and 1
        if layer_reduction < 0 or layer_reduction > 1:
            raise ValueError('layer_reduction must be between 0 and 1')
        # Compute hidden layer sizes
        hidden_sizes = [input_size]
        for _ in range(1, num_layers):
            next_size = int(hidden_sizes[-1] * layer_reduction)
            hidden_sizes.append(max(next_size, 3))

        # Adjust the number of layers to match the num_layers parameter
        hidden_sizes = hidden_sizes[:num_layers] + [3]

        # Create the hidden layers
        layers = [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]

        # Add the output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], 3)
        self.current_loss = None
        # Register all layers
        self.layers = nn.ModuleList(layers)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Set the device attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the model to the specified device

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        x = x.squeeze(1)
        return x

    def update_learning_rate(self, new_learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def fit(self, train_data, epochs, batch_size, verbose=True):
        # Get the dataset from the Subset
        dataset = train_data.dataset

        # If the dataset is a TensorDataset, extract the labels tensor
        _, labels_tensor = dataset.tensors
        # Calculate class weights
        class_counts = torch.bincount(labels_tensor)
        class_weights = 1. / class_counts.float()
        class_weights[class_counts == 0] = 0  # Set weight for classes with 0 instances to 0
        class_weights = class_weights.to(self.device)

        # Update the criterion to include class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Create a DataLoader from the train_data Subset
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long()

                # Forward pass
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.current_loss = loss.item()
            if verbose:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, bert_vector):
        # Predict sentiment from BERT vector
        bert_vector = bert_vector.to(self.device)  # Move input vector to device
        with torch.no_grad():
            outputs = self(bert_vector.unsqueeze(0))  # Add batch dimension
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            sentiment_label = predicted.item()

            # Print the probabilities
            #print("Probabilities:", probabilities.cpu().numpy())  # Use .cpu() to move tensor back to CPU if necessary

            # Return the label and probabilities
            return sentiment_label


def get_mbert_base_embedding(text):
    # Tokenize and encode the text without truncation
    inputs = MBertTokenizer_(text, return_tensors='pt')
    inputs = inputs.to(device)
    # Generate embeddings
    with torch.no_grad():
        outputs = MBertModel_(**inputs)

    # Use the mean of the last layer embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def get_mbart_large_embedding(text):
    # Tokenize and encode the text without truncation
    inputs = MBartTokenizer_(text, return_tensors='pt')
    inputs = inputs.to(device)

    # Generate embeddings
    with torch.no_grad():
        outputs = MBartModel_(**inputs)

    # Use the mean of the last layer embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def prepare_data(df, vectorizer_func, train_proportion=0.8, verbose=False):
    # Split the dataset
    train_df, test_df = train_test_split(df, train_size=train_proportion)

    # Process training and testing data
    if verbose:
        print("\nProcessing training data...\n")
        train_data = [(vectorizer_func(text), torch.tensor([label], dtype=torch.long)) for text, label in tqdm(zip(train_df['text'], train_df['label']), total=len(train_df))]
        print("\nProcessing testing data...\n")
        test_data = [(vectorizer_func(text), torch.tensor([label], dtype=torch.long)) for text, label in tqdm(zip(test_df['text'], test_df['label']), total=len(test_df))]
    else:
        train_data = [(vectorizer_func(text), torch.tensor([label], dtype=torch.long)) for text, label in zip(train_df['text'], train_df['label'])]
        test_data = [(vectorizer_func(text), torch.tensor([label], dtype=torch.long)) for text, label in zip(test_df['text'], test_df['label'])]

    # Convert lists of tuples to TensorDataset
    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    # Stack all tensors to create a single tensor for features and labels
    train_features = torch.stack(train_features)
    train_labels = torch.cat(train_labels)
    test_features = torch.stack(test_features)
    test_labels = torch.cat(test_labels)

    # Create TensorDataset
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    return train_dataset, test_dataset



def create_tensor_dataset(df, vectorizer_func, tensor_filename, verbose=True):
    """
    Creates a tensor dataset from the given DataFrame without splitting into training and testing.

    :param df: DataFrame containing the data.
    :param vectorizer_func: Function to vectorize the text data.
    :param tensor_filename: The filename to save the dataset.
    :param verbose: If True, print out verbose messages.
    """
    # Check if the file already exists
    if os.path.exists(tensor_filename):
        print(f"Tensor file '{tensor_filename}' already exists.")
        return

    # Process data
    if verbose:
        print(f"Processing data {tensor_filename}...")
        data = [(vectorizer_func(text), torch.tensor([label], dtype=torch.long)) for text, label in tqdm(zip(df['text'], df['label']), total=len(df))]
    else:
        data = [(vectorizer_func(text), torch.tensor([label], dtype=torch.long)) for text, label in zip(df['text'], df['label'])]

    # Extract features and labels
    features, labels = zip(*data)

    # Stack all tensors to create a single tensor for features and labels
    features = torch.stack(features)
    labels = torch.cat(labels)

    # Create TensorDataset
    tensor_dataset = TensorDataset(features, labels)

    # Save to file
    torch.save(tensor_dataset, tensor_filename)
    if verbose:
        print(f"Tensor dataset saved as '{tensor_filename}'")

def create_case_conscious_tensor_dataset(df, vectorizer_func, tensor_filename, verbose=True):
    if os.path.exists(tensor_filename):
        print(f"Tensor file '{tensor_filename}' already exists.")
        return

    if verbose:
        print(f"Processing data {tensor_filename}...")
        data = []
        for text, label in tqdm(zip(df['text'], df['label']), total=len(df)):
            embedding = vectorizer_func(text)
            case_tags = case_tagger(text)
            combined_feature = append_list_to_tensor(embedding, case_tags)
            data.append((combined_feature, torch.tensor([label], dtype=torch.long)))
    else:
        data = []
        for text, label in zip(df['text'], df['label']):
            embedding = vectorizer_func(text)
            case_tags = case_tagger(text)
            combined_feature = append_list_to_tensor(embedding, case_tags)
            data.append((combined_feature, torch.tensor([label], dtype=torch.long)))

    # Extract features and labels
    features, labels = zip(*data)

    # Stack all tensors to create a single tensor for features and labels
    features = torch.stack(features)
    labels = torch.cat(labels)

    # Create TensorDataset
    tensor_dataset = TensorDataset(features, labels)

    # Save to file
    torch.save(tensor_dataset, tensor_filename)
    if verbose:
        print(f"Tensor dataset saved as '{tensor_filename}'")


def load_and_split_dataset(file_path, train_proportion=0.8):
    # Load the dataset
    full_dataset = torch.load(file_path)

    # Split the dataset
    train_size = int(train_proportion * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset


if __name__ == "__main__":
    create_tensor_dataset(preprocess_data_EN("./SemEval2017/SemEval2017-task4.csv"), get_mbart_large_embedding, "SemEval2017-task4-mbart-large-cc25.pt")
