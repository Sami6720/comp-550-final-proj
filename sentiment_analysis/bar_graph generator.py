import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.patches as mpatches

# Load the JSON file into a Python dictionary
with open('Eval_results.json', 'r') as file:
    eval_results = json.load(file)

# Initialize a dictionary to hold accuracy data for each model
accuracy_data_ru = {model: {'Train': [], 'Test': []} for model in
                    ['BART', 'BERT', 'BART CASE TAGGER', 'BERT CASE TAGGER']}
accuracy_data_en = {model: {'Train': [], 'Test': []} for model in ['BART', 'BERT']}
preprocessing_steps_ru = []
preprocessing_steps_en = []

hola=sorted(eval_results.items())
# Extract accuracy for Russian and English models
for key, value in sorted(eval_results.items()):
    if "RU" in key:
        step = key.split("RU")[1].strip()
        preprocessing_steps_ru.append(step)
        model_key = key.split("RU", 1)[0].strip()
        accuracy_data_ru[model_key]['Train'].append(value['Train Accuracy'])
        accuracy_data_ru[model_key]['Test'].append(value['Test Accuracy'])
    elif "EN" in key:
        step = key.split("EN")[1].strip()
        preprocessing_steps_en.append(step)
        model_key = key.split("EN", 1)[0].strip()
        accuracy_data_en[model_key]['Train'].append(value['Train Accuracy'])
        accuracy_data_en[model_key]['Test'].append(value['Test Accuracy'])


preprocessing_steps_en = sorted(set(preprocessing_steps_en))
preprocessing_steps_ru = sorted(set(preprocessing_steps_ru))

# Define a function for plotting
# Define colors for each model
model_colors = {
    'BART': ('#007acc', '#4da3ff'),  # Darker for Train, lighter for Test
    'BERT': ('#cc0000', '#ff6666'),
    'BART CASE TAGGER': ('#00cc66', '#66ff99'),
    'BERT CASE TAGGER': ('#9933ff', '#cc99ff'),
}

# Define a function for plotting accuracies
def plot_accuracies(accuracy_data, preprocessing_steps, title, filename):
    def autolabel(rects1, rects2):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)
    n_groups = len(preprocessing_steps)
    fig, ax = plt.subplots(figsize=(15, 10))
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    for i, (model, data) in enumerate(accuracy_data.items()):
        train_color, test_color = model_colors[model]
        rects1 = ax.bar(index + bar_width * i, data['Train'], bar_width,
                        alpha=opacity, color=train_color, label=f'{model} Train', edgecolor='black')
        rects2 = ax.bar(index + bar_width * i, data['Test'], bar_width,
                        alpha=opacity, color=test_color, label=f'{model} Test', edgecolor='black')

        autolabel(rects1, rects2)

    # Set the x-axis labels diagonally
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(preprocessing_steps, rotation=45, ha='right')

    # Set labels and title
    ax.set_xlabel('Preprocessing Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)

    # Create a custom legend
    legend_patches = [mpatches.Patch(color=model_colors[model][0], label=f'{model} Train') for model in accuracy_data]
    legend_patches.extend([mpatches.Patch(color=model_colors[model][1], label=f'{model} Test') for model in accuracy_data])
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(accuracy_data)*2, fancybox=True, shadow=True)

    # Save the plot as a .png file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



# Call the plotting function for Russian and English models
plot_accuracies(accuracy_data_ru, preprocessing_steps_ru, 'Russian Models Accuracy by Preprocessing Step', 'accuracy_ru_models.png')
plot_accuracies(accuracy_data_en, preprocessing_steps_en, 'English Models Accuracy by Preprocessing Step', 'accuracy_en_models.png')