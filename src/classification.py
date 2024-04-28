import torch
from torch.nn import Module as BaseModule
import torch.optim as Optimizer
import torch.nn as NeuralNetwork
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix as ConfusionMatrix
from .Config import ELMOConfig, Structure, Constants
from bidict import bidict
from alive_progress import alive_bar, alive_it
import numpy as Numpy
import matplotlib.pyplot as Plot
import seaborn as Seaborn

from typing import Callable
import torch.nn as NeuralNetwork
import torch
import torch.optim as Optimizer
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset as TorchDataset
from alive_progress import alive_bar

from sklearn.metrics import confusion_matrix as ConfusionMatrix
import seaborn as Seaborn
import matplotlib.pyplot as Plot

from .Config import ClassifierConfig, Structure
from .elmo import ELMO
from .elmo import createDataPoints

class ClassifierDataset(TorchDataset):
    def __init__(self, data : list , mapping) -> None:
        self.data = data
        self.dataPoints = [(createDataPoints(sentence[1],mapping),torch.tensor(sentence[0]-1)) for sentence in alive_it(self.data, force_tty = True)]

    def __len__(self):
        return len(self.dataPoints)

    def __getitem__(self, idx):
        return self.dataPoints[idx]

class Classifier(BaseModule):
    def __init__(self, model) -> None:
        super().__init__()

        self.model = model
        self.device = self.model.device
        self.gamma = NeuralNetwork.Parameter(torch.rand(3, requires_grad = True))
        self.bias = NeuralNetwork.Parameter(torch.zeros(1, requires_grad = True))
        
        self.linear = NeuralNetwork.Linear(ELMOConfig.EmbeddingSize, 4)
        self.embeddingLayer = NeuralNetwork.Linear(ELMOConfig.EmbeddingSize, ELMOConfig.EmbeddingSize)

        self.LSTM = NeuralNetwork.LSTM(
            input_size = Constants.EmbeddingSize,
            hidden_size = ELMOConfig.hiddenStateSize,
            bidirectional = True,
            batch_first = True,
            num_layers = 1
        )

        self.to(self.device)
         
    def evaluation(self):
        if not hasattr(self, 'testDataset') :
            print("No test data found. Skipping evaluation.")
            return 0.0 
        
        correct = 0
        predicted = []
        actual = []
        with torch.no_grad():
            for y,x in self.testDataset:
                x = x.squeeze(2)
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.forward(x)
                y_hat = torch.argmax(y_hat)
                y = torch.argmax(y)
                predicted.append(y_hat.cpu().detach().numpy())
                actual.append(y.cpu().detach().numpy())
                correct += (y_hat == y)

        self.confusionMatrix = ConfusionMatrix(actual, predicted)
        return float(correct*100/ len(self.testDataset))

    def forward(self, sentence):
        e0, e1, e2  = self.model.getFrozenEmbeddings(sentence)
        embedding = self.embeddingLayer(e0 + e1 + e2)
        _, (out, _) = self.LSTM(embedding)
        out = out.permute(1,0,2).contiguous()
        out = out.view(out.shape[0], -1)
        out = self.linear(out)

        return out
    
    def saveModel(self, fileName):
        savePathFile = Structure.checkPointPath + fileName
        torch.save(self.state_dict(), savePathFile)
        print(f"Saved model to {savePathFile}.")

    def loadModel(self, fileName):
        savePathFile = Structure.checkPointPath + fileName
        self.load_state_dict(torch.load(savePathFile), strict=False)
        print(f"Loaded model from {savePathFile}.")

def getDataPointFromData(data , embeddingLookUp : Callable[[str], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    sentenceClass = torch.zeros(ClassifierConfig.numClasses, dtype=torch.float)
    sentenceClass[data[0]-1] = 1.0

    return sentenceClass, torch.stack([embeddingLookUp(word) for word in data[1]])
