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

from .Config import ClassificationConfig, Structure

class RNNClassfierDatset(TorchDataset):
    def __init__(self, data : list , mapping : Callable) -> None:
        self.data = data
        self.mapping = mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return getDataPointFromData(self.data[idx], embeddingLookUp=self.mapping)

class RNNClassifier(NeuralNetwork.Module):
    def __init__(self, 
                 inputSize     : int,
                 embeddingMap  : Callable[[str], torch.Tensor],
                 data          : list,
                 testData      : list,
                 fileName      : str  = "classifier.pt",
                 hiddenSize    : int  = ClassificationConfig.HiddenStateSize, 
                 outputSize    : int  = ClassificationConfig.numClasses, 
                 bidirectional : bool = ClassificationConfig.bidirectional,
                 stackSize     : int  = 1, 
                 loadIfAvail   : bool = True, 
                 learningRate  : float = ClassificationConfig.learningRate) -> None:
        super().__init__()
        
        if loadIfAvail :
            try :
                self.loadModel(fileName)
            except FileNotFoundError:
                print(f"No model found, building from scratch.")
            except :
                import traceback
                print(traceback.format_exc())

        self.hiddenSize = hiddenSize
        self.bidirectional = bidirectional
        self.outputSize = outputSize
        self.stackSize = stackSize
        self.lstm = NeuralNetwork.LSTM(input_size=inputSize, 
                                     hidden_size=hiddenSize, 
                                     bidirectional=bidirectional,
                                     num_layers=stackSize,
                                     batch_first=True)
        self.linear = NeuralNetwork.Linear(hiddenSize*(1 + self.bidirectional), outputSize)
        self.embeddingMap = embeddingMap
        self.data = RNNClassfierDatset(data = data, mapping=self.embeddingMap)
        self.testData = RNNClassfierDatset(data = testData, mapping=self.embeddingMap)
        self.lossFunction = NeuralNetwork.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = Optimizer.Adam(self.parameters(), lr = learningRate)
        self.relu = NeuralNetwork.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fileName = fileName
        self.to(self.device)

        print(f"Using {self.device} as device.")
    
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

    def forward(self, x):
        # print(x.shape)
        _, (x, _) = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        # x = x.reshape(1, -1)
        x = self.linear(x)
        x = self.relu(x)
        return x
    
    def __train(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        loss = self.lossFunction(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def trainModel(self):
        bestTestAccuracy = 0.0
        for epoch in range(ClassificationConfig.epochs):
            avgLoss = 0
            with alive_bar(len(self.trainDataset), force_tty=True) as bar:
                for x,y in self.trainDataset:
                    x = x.squeeze(2)
                    avgLoss += self.__train(x, y)
                    bar()
                avgLoss /= len(self.trainDataset)
                print(f"Epoch : {epoch+1}, Loss : {avgLoss}")
                testAcc = self.evaluation()
                print(f"Test Accuracy : {testAcc:.2f}%")
                if testAcc > bestTestAccuracy :
                    self.saveModel(fileName = self.fileName)
                    bestTestAccuracy = testAcc
        print("Training Complete.")

    def createDataset(self):
        self.trainDataset = DataLoader(self.data, batch_size=ClassificationConfig.batchSize, collate_fn=self.customCollate, shuffle=True)
        self.testDataset = DataLoader(self.testData, shuffle=False)
        # TODO : Batch the testset as well for faster evaluation.
    def customCollate(self, batch):
        # Extract sequences and labels from the batch
        labels, sequences = zip(*batch)
        # Pad sequences to the length of the longest sequence
        paddedSequences = NeuralNetwork.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1)
        # Convert labels to tensor
        paddedLabels = NeuralNetwork.utils.rnn.pad_sequence(labels, batch_first=True)
        return paddedSequences, paddedLabels
    
    def saveModel(self, fileName):
        savePathFile = Structure.modelPath + fileName
        torch.save(self.state_dict(), savePathFile)
        print(f"Saved model to {savePathFile}.")

    def loadModel(self, fileName):
        savePathFile = Structure.modelPath + fileName
        self.load_state_dict(torch.load(savePathFile), strict=False)
        print(f"Loaded model from {savePathFile}.")
    
    def getConfusionMatrix(self, predictedLabels, actualLabels):
        self.confusionMatrix = ConfusionMatrix(actualLabels, predictedLabels)
        return self.confusionMatrix
    
    def plotConfusionMatrix(self, name):
        Seaborn.heatmap(self.confusionMatrix, annot=True, cmap="Blues", fmt='d')
        Plot.xlabel("Predicted")
        Plot.ylabel("Actual")
        Plot.title("Confusion Matrix")
        Plot.tight_layout()
        Plot.savefig(Structure.resultsPath + f"confusionMatrix_{name}.svg", format="svg")
        Plot.show()

def getDataPointFromData(data , embeddingLookUp : Callable[[str], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    sentenceClass = torch.zeros(ClassificationConfig.numClasses, dtype=torch.float)
    sentenceClass[data[0]-1] = 1.0

    return sentenceClass, torch.stack([embeddingLookUp(word) for word in data[1]])
