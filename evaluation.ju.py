# %% [markdown]
"""
## Evaluation of Models.
"""

# %%
from src.parser import Dataset
from src.Config import ELMOConfig
from src.classification import ClassifierDataset
from alive_progress import alive_bar
import torch
from torch.utils.data import DataLoader
# %%
dataset = Dataset()

# %%
dataset.getData()
mapping = dataset.getMapping()

# %%
dataset.getTestData()

# %%
testData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.testLabels, dataset.testData)]

# %%
def classificationCollate(batch):
    sentences, labels = zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(list(sentences), batch_first = True, padding_value = 0), torch.stack(labels)

# %%
classifierTestData = DataLoader(ClassifierDataset(testData, mapping),
                                batch_size = ELMOConfig.batchSize,
                                shuffle = True, collate_fn = classificationCollate)

# %%
def loadModel(model, modelName):
    savePathFile = f"./pretrained/{modelName}"
    model.load_state_dict(torch.load(savePathFile))
    print(f"Loaded model from {savePathFile}.")

# %%
def evaluation(model, dataset):
    correct = 0
    total = 0

    with torch.no_grad():
        with alive_bar(len(dataset), force_tty = True, length = 20) as bar:
            for data, labels in dataset:
                data = data.to(model.device)
                labels = labels.to(model.device)
                
                pred = model(data)
                pred = pred.view(-1, pred.shape[-1]).argmax(dim=1)
                
                mask = (labels != 0)
                matches = torch.sum(pred[mask] == labels[mask]).item()
                correct += matches
                total += len(labels[mask])
                bar()
        accuracy = (correct / total) * 100
        return accuracy

# %% [markdown]
"""
Loading the ELMO model.
"""


# %%
from src.elmo import ELMO
elmoModel = ELMO(mapping=mapping)
loadModel(elmoModel, "elmoModel.pt")


# %%
from src.classification import Classifier
classifier = Classifier(elmoModel)

# %% [markdown]
"""
### Frozen Parameters.
"""
loadModel(classifier, "Classifier_Frozen")
evaluation(classifier, classifierTestData)

# %% [markdown]
"""
### Learnable Parameters.
"""
loadModel(classifier, "Classifier_Learnable")
evaluation(classifier, classifierTestData)

# %% [markdown]
"""
### Learnable Function.
"""
loadModel(classifier, "Classifier_LearnableFunction")
evaluation(classifier, classifierTestData)
