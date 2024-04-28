# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
"""
ELMO Testing.
"""

# %% 
from src.parser import Dataset
from alive_progress import alive_bar
import torch
# %%
dataset = Dataset()

# %%
dataset.getData()
mapping = dataset.getMapping()

# %%
dataset.getTestData()

# %%
trainData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.labels, dataset.tokenizedData)]
testData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.testLabels, dataset.testData)]

# %%
from src.Config import Constants, ELMOConfig
from torch.utils.data import DataLoader
mapping[Constants.padToken]

# %%
def saveModel(model, modelName):
    savePathFile = f"./pretrained/{modelName}"
    torch.save(model.state_dict(), savePathFile)
    print(f"Saved model to {savePathFile}.")

# %%
def predictionCollate(batch : list[torch.Tensor]):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first = True, padding_value = 0)

# %%
from src.elmo import ELMO, PredictionDataset
elmoModel = ELMO(mapping=mapping)

trainDataset = DataLoader(PredictionDataset(trainData, mapping), 
                       batch_size = ELMOConfig.batchSize, 
                       shuffle = True, collate_fn = predictionCollate)

testDataset = DataLoader(PredictionDataset(testData, mapping),
                      batch_size = ELMOConfig.batchSize,
                      shuffle = True, collate_fn = predictionCollate)
saveModel(elmoModel, "ignore")

# %%
def evaluation():
    correct = 0
    total = 0
    with torch.no_grad():
        with alive_bar(len(testDataset), force_tty = True, length = 20) as bar:
            for data in testDataset:
                data = data.to(elmoModel.device)
                pred = elmoModel(data)
                pred = pred.view(-1, pred.shape[-1]).argmax(dim=1)
                labels = data.view(-1)
                
                mask = (labels != 0)
                matches = torch.sum(pred[mask] == labels[mask]).item()
                correct += matches
                total += len(labels[mask])
                bar()
    accuracy = (correct / total) * 100
    return accuracy

# %% 
def trainPrediction():
    Loss = []
    batchNum = 0
    
    # trainDataset = testDataset

    Optimizer = torch.optim.Adam(elmoModel.parameters(), lr = ELMOConfig.predictionLearningRate)
    LossFunction = torch.nn.CrossEntropyLoss(ignore_index = 0)

    for epoch in range(ELMOConfig.epochs):
        avgLoss = 0
        with alive_bar(len(trainDataset), force_tty = True, length = 20) as bar:
            for data in trainDataset:
                data = data.to(elmoModel.device)

                Optimizer.zero_grad()
                pred = elmoModel(data)

                pred = pred.view(-1, pred.shape[-1])
                labels = data.view(-1)
                loss = LossFunction(pred, labels)
                loss.backward()
                Optimizer.step()
                
                avgLoss /= (batchNum + 1)
                avgLoss *= (batchNum)
                avgLoss += loss.item()/(batchNum + 1)
                batchNum += 1
                bar()
        Loss.append(avgLoss)
        accuracy = evaluation()
        print(f"Epoch {epoch + 1}/{ELMOConfig.epochs} | Loss : {avgLoss:.4f} | Accuracy : {accuracy:.4f}")
    saveModel(elmoModel, "elmoModel")
# %%
trainPrediction()

# %%
evaluation()


# %%
from src.classification import Classifier
from src.classification import ClassifierDataset

# %%
classifier = Classifier(elmoModel)

# %%
def classificationCollate(batch : tuple[torch.Tensor, torch.Tensor]):
    sentences, labels = zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(list(sentences), batch_first = True, padding_value = 0), torch.stack(labels)

# %%
classifierTrainData = DataLoader(ClassifierDataset(trainData, mapping),
                                batch_size = ELMOConfig.batchSize,
                                shuffle = True, collate_fn = classificationCollate)
classifierTestData = DataLoader(ClassifierDataset(testData, mapping),
                                batch_size = ELMOConfig.batchSize,
                                shuffle = True, collate_fn = classificationCollate)

# %%
def classifierEvaluation():
    correct = 0
    total = 0

    with torch.no_grad():
        with alive_bar(len(classifierTestData), force_tty = True, length = 20) as bar:
            for data, labels in classifierTestData:
                data = data.to(elmoModel.device)
                labels = labels.to(elmoModel.device)
                pred = classifier(data)
                pred = pred.view(-1, pred.shape[-1]).argmax(dim=1)
                
                mask = (labels != 0)
                matches = torch.sum(pred[mask] == labels[mask]).item()
                correct += matches
                total += len(labels[mask])
                bar()
        accuracy = (correct / total) * 100
        return accuracy
# %%
classifierEvaluation()

# %%
classifierLoss = []
batchNum = 0
Optimizer = torch.optim.Adam(classifier.parameters(), lr = ELMOConfig.classifierLearningRate)
LossFunction = torch.nn.CrossEntropyLoss(ignore_index = 0)

for epoch in range(ELMOConfig.epochs):
    avgLoss = 0
    with alive_bar(len(classifierTrainData), force_tty = True, length = 20) as bar:
        for data, label in classifierTrainData:
            data = data.to(classifier.device)
            label = label.to(classifier.device)

            Optimizer.zero_grad()
            pred = classifier(data)

            pred = pred.view(-1, pred.shape[-1])
            # print(pred.shape, label.shape)
            loss = LossFunction(pred, label)
            loss.backward()
            Optimizer.step()
            
            avgLoss /= (batchNum + 1)
            avgLoss *= (batchNum)
            avgLoss += loss.item()/(batchNum + 1)
            batchNum += 1
            bar()
    classifierLoss.append(avgLoss)
    accuracy = classifierEvaluation()
    print(f"Epoch {epoch + 1}/{ELMOConfig.epochs} | Loss : {avgLoss:.4f} | Accuracy : {accuracy:.4f}")


# %%
classifier.saveModel("Classifier_Learnable")
