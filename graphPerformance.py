# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# svdModels = ['SVD-CW = 1', 'SVD-CW = 2', 'SVD-CW = 3']
# svdAccuracy = [82.88, 85.24, 86.15]  # Accuracy scores for SVD models
# svdTime = [180, 195, 210]  # Training times for SVD models (in seconds)
#
# w2vModels = ['W2V-CW = 1', 'W2V-CW = 2', 'W2V-CW = 3']
# w2vAccuracy = [88.66, 90.22, 90.39]  # Accuracy scores for Word2Vec models 
# w2vTime = [2400, 2600, 2800]  # Training times for Word2Vec models (in seconds)
#
#
# markers = {'SVD-CW = 1': 'o', 'SVD-CW = 2': 'o', 'SVD-CW = 3': 'o', 'W2V-CW = 1': '^', 'W2V-CW = 2': 'v', 'W2V-CW = 3': 's'}
# # Extend lists and create DataFrame
# models = svdModels + w2vModels
# accuracy = svdAccuracy + w2vAccuracy
# training_time = svdTime + w2vTime
#
# print(models, accuracy, training_time)
# data = pd.DataFrame({'Model': models, 'Accuracy': accuracy, 'Training Time (s)': training_time})
#
# # Plotting
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x='Training Time (s)', y='Accuracy', hue='Model', s=100, markers=markers, legend='brief')
#
# # Add labels and title
# plt.xlabel('Training Time (seconds)')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Training Time for Different Models')
#
# # Add legend
# plt.legend(title='Model')
#
# # Show plot
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

svdModels = ['SVD-CW = 1', 'SVD-CW = 2', 'SVD-CW = 3']
svdAccuracy = [82.88, 85.24, 88.65]  # Accuracy scores for SVD models
svdTime = [180, 195, 210]  # Training times for SVD models (in seconds)

w2vModels = ['W2V-CW = 1', 'W2V-CW = 2', 'W2V-CW = 3']
w2vAccuracy = [88.66, 90.22, 90.39]  # Accuracy scores for Word2Vec models 
w2vTime = [2400, 2600, 2800]  # Training times for Word2Vec models (in seconds)

# Create DataFrame for SVD models
svd_data = pd.DataFrame({'Model': svdModels, 'Accuracy': svdAccuracy, 'Training Time (s)': svdTime})
svd_data['Group'] = 'SVD'

# Create DataFrame for Word2Vec models
w2v_data = pd.DataFrame({'Model': w2vModels, 'Accuracy': w2vAccuracy, 'Training Time (s)': w2vTime})
w2v_data['Group'] = 'Word2Vec'

# Combine data for both groups
data = pd.concat([svd_data, w2v_data])

# Plotting
plt.figure(figsize=(10, 6))

# Plot SVD models with 'o' markers
sns.scatterplot(data=svd_data, x='Training Time (s)', y='Accuracy', hue='Model', s=500, marker='P')

# Plot Word2Vec models with different markers
sns.scatterplot(data=w2v_data, x='Training Time (s)', y='Accuracy', hue='Model', s=500, marker='v')

# Add labels and title
plt.xlabel('Training Time (seconds)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Time for Different Models')

# Add legend
plt.legend(title='Model')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.savefig(f"graphPerformance.svg", format="svg")
plt.show()
