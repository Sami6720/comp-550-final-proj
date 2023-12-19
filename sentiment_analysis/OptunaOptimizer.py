import optuna

from ML_classifier import SentimentClassifier, load_and_split_dataset
from main import evaluate_accuracy

def objective(trial, tensor_dataset_path,input_size):
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    #lr = round(lr, 6)  # Round lr to 6 decimal places
    num_layers = trial.suggest_int("num_layers", 3, 5)
    layer_reduction = trial.suggest_float("layer_reduction", 0.3, 0.8)
    #layer_reduction = round(layer_reduction, 2)  # Round to 2 decimal places
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    # Initialize model
    model = SentimentClassifier(input_size=input_size, num_layers=num_layers, layer_reduction=layer_reduction, learning_rate=lr)

    # Load data
    ENTrain, ENTest = load_and_split_dataset(tensor_dataset_path, 0.8)

    # Training logic
    model.fit(ENTrain, epochs=30, batch_size=batch_size)

    # Evaluation logic
    accuracy = evaluate_accuracy(model, ENTest)
    return accuracy*2 - model.current_loss

def format_trial_output(best_params):
    best_params['lr'] = round(best_params['lr'], 6)  # Round lr to 6 decimal places
    for param in ['num_layers', 'layer_reduction', 'batch_size']:
        if param in best_params:
            best_params[param] = round(best_params[param], 2)
    return best_params

if __name__=="__main__":
    tensor_dataset_path = "./Tensor_Datasets/Bert_RU_Replace_Emojis+Include_Speech_Label_as_Positive+Lemmatize+StopWords.pt"
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, tensor_dataset_path,768), n_trials=25)
    # Adjust the final output to round values accordingly
    best_params = format_trial_output(study.best_params)
    print("Best hyperparameters: ", best_params)
