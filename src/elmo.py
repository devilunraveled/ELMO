import torch
from torch.nn import Module as BaseModule
import torch.optim as Optimizer
import torch.nn as NeuralNetwork
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from bidict import bidict
from Config import ELMOConfig, Structure
from sklearn.metrics import confusion_matrix as ConfusionMatrix


def createDataPoints(sentence : list[str], mapping : bidict):
    dataPoints = [ mapping.get(word, 0) for word in sentence ]
    return dataPoints

class ELMODatasetClassification(TorchDataset):
    def __init__(self, sentences : list[list], mapping : bidict ) -> None:
        self.sentences = sentences
        self.mapping = mapping
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[index][1]
        dataPoint = createDataPoints(sentence, self.mapping)
        label = torch.zeros(ELMOConfig.numClasses, dtype=torch.float)
        label[sentence[0] - 1] = 1
        return torch.tensor(dataPoint), label

class ELMODatasetPrediction(TorchDataset):
    def __init__(self, sentences : list[list[int]] ) -> None:
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index : int) -> list:
        return self.sentences[index]

class ELMO(BaseModule):
    def __init__(self, 
                 mapping          : bidict, 
                 trainSentences   : list[list],
                 testSentences    : list[list],
                 classifierName   : str     = "ELMO_Classifier.pt",
                 predictionName   : str     = "ELMO_Predictor.pt",
                 numClasses       : int     = ELMOConfig.numClasses,
                 embeddingSize    : int     = ELMOConfig.EmbeddingSize,
                 hiddenSize       : int     = ELMOConfig.hiddenStateSize,
                 classifierLayers : tuple   = ELMOConfig.classifierLayers,
                 predictionLayers : tuple   = ELMOConfig.predictionLayers,
                 ) -> None:
        super().__init__()

        self.mapping = mapping
        self.trainSentences = trainSentences
        self.testSentences = testSentences

        self.deviceString = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.deviceString)

        self.EmbeddingSize = embeddingSize
        
        self.embeddingSize = embeddingSize
        self.hiddenStateSize = hiddenSize
        
        self.classifierLayers = classifierLayers
        self.numClasses = numClasses
        self.classifierName = classifierName

        self.predictionLayers = predictionLayers
        self.vocabLength = len(self.mapping)
        self.predictionName = predictionName

        self.trainSetClassification = ELMODatasetClassification(self.trainSentences, mapping=self.mapping)
        self.trainDatasetClassification = DataLoader(self.trainSetClassification, batch_size=ELMOConfig.batchSize, collate_fn=self.customCollate, shuffle=True)
        self.testSetClassification  = ELMODatasetClassification(self.testSentences, mapping=self.mapping)
        self.testDatasetClassification = DataLoader(self.testSetClassification, batch_size=ELMOConfig.batchSize, collate_fn=self.customCollate)
        
        self.trainSetPrediction = ELMODatasetPrediction(self.trainSentences)
        self.trainDatasetPrediction = DataLoader(self.trainSetPrediction, batch_size=ELMOConfig.batchSize, collate_fn=self.customCollate, shuffle=True)
        self.testSetPrediction  = ELMODatasetPrediction(self.testSentences)
        self.testDatasetPrediction = DataLoader(self.testSetPrediction, batch_size=ELMOConfig.batchSize, collate_fn=self.customCollate)


        ############################# Embedding ############################
        self.embedding = NeuralNetwork.Embedding( 
            num_embeddings = self.vocabLength, 
            embedding_dim = self.embeddingSize
        )

        ######################### BI-LSTM Structure #########################

        ### PreTrainedBiLSTM : Trained on the Next Word Prediction Task.
        self.PreTrainedBiLSTM = NeuralNetwork.LSTM(
            input_size      = self.embeddingSize,
            hidden_size     = self.hiddenStateSize,
            num_layers      = ELMOConfig.numLayers,
            batch_first     = True,
            bidirectional   = ELMOConfig.bidirectional,
            device          = self.device,
            dropout         = ELMOConfig.dropout
        )

        ### DownStreamBiLSTM : Trained on the Sentence Classification Task.
        self.DownStreamBiLSTM = NeuralNetwork.LSTM(
            input_size      = self.hiddenStateSize,
            hidden_size     = self.hiddenStateSize,
            num_layers      = ELMOConfig.numLayers,
            batch_first     = True,
            bidirectional   = ELMOConfig.bidirectional,
            device          = self.device,
            dropout         = ELMOConfig.dropout
        )
        
        ######################### Classification #########################
        self.classifier = NeuralNetwork.ModuleList()
        for i in self.classifierLayers:
            layerSize = self.classifierLayers[i]
            if i == 0 :
                self.classifier.append(NeuralNetwork.Linear(in_features = self.hiddenStateSize, out_features = layerSize))
            else :
                if i == (len(self.classifierLayers) - 1) :
                    self.classifier.append(NeuralNetwork.Linear(in_features = layerSize, out_features = self.numClasses))
                else :
                    previousLayerSize = self.classifierLayers[i - 1]
                    self.classifier.append(NeuralNetwork.Linear(in_features = previousLayerSize, out_features = layerSize))
        
        ########################### Prediction ###########################
        self.prediction = NeuralNetwork.ModuleList()
        for i in self.predictionLayers:
            layerSize = self.predictionLayers[i]
            if i == 0 :
                self.prediction.append(NeuralNetwork.Linear(in_features = self.hiddenStateSize, out_features = layerSize))
            else :
                if i == (len(self.predictionLayers) - 1) :
                    self.prediction.append(NeuralNetwork.Linear(in_features = layerSize, out_features = self.vocabLength))
                else :
                    previousLayerSize = self.predictionLayers[i - 1]
                    self.prediction.append(NeuralNetwork.Linear(in_features = previousLayerSize, out_features = layerSize))
        
        self.RELU = NeuralNetwork.ReLU()
        self.softmax = NeuralNetwork.Softmax(dim = 1)
        
        self.classificationLossFunction = NeuralNetwork.CrossEntropyLoss()
        self.predictionLossFunction = NeuralNetwork.CrossEntropyLoss()

        self.classificationOptimizer = Optimizer.Adam(self.parameters(), lr = ELMOConfig.classifierLearningRate)
        self.predictionOptimizer = Optimizer.Adam(self.parameters(), lr = ELMOConfig.predictionLearningRate)

        print(f"Using {self.deviceString} as device.")
        self.trained = False
    
    def saveModel(self, fileName):
        filePath = Structure.modelPath + fileName
        torch.save(self.state_dict(), filePath)
        print(f"Saved model to {filePath}.")

    def __classifierForward(self, sentence):
        _, (input, _) = self.PreTrainedBiLSTM(sentence)
        input = input.permute(1, 0, 2)
        x = input.reshape(input.shape[0], -1)
        for layer in self.classifier:
            x = layer(x)
            if layer != self.classifier[-1]:
                x = self.RELU(x)
            else :
                x = self.softmax(x)

        return x

    def trainClassifierStep(self, input, label):
        self.classificationOptimizer.zero_grad()
        y_hat = self.__classifierForward(input)
        loss = self.classificationLossFunction(y_hat, label)
        loss.backward()
        self.classificationOptimizer.step()
        return loss.item()

    def trainClassifier(self):
        bestTestAccuracy = 80
        for epoch in range(ELMOConfig.epochs):
            avgLoss = 0
            for x, y in self.trainDatasetClassification:
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.trainClassifierStep(x,y)
                avgLoss += loss
            avgLoss = avgLoss / len(self.trainDatasetClassification)
            testAccuracy = self.evaluation()
            print(f"Epoch : {epoch+1} | Loss: {avgLoss} | Test Accuracy : {testAccuracy}")
            if testAccuracy > bestTestAccuracy :
                bestTestAccuracy = testAccuracy
                self.saveModel(f"{self.classifierFileName}_{str(testAccuracy)}.model")
    

    def classifierEvaluation(self, data = None):
        if data is None :
            data = self.testData

        correct = 0
        predicted = []
        actual = []

        with torch.no_grad():
            for y,x in data:
                x = x.squeeze(2)
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.__classifierForward(x)
                y_hat = torch.argmax(y_hat)
                y = torch.argmax(y)
                predicted.append(y_hat.cpu().detach().numpy())
                actual.append(y.cpu().detach().numpy())
                correct += (y_hat == y)

        self.confusionMatrix = ConfusionMatrix(actual, predicted)
        return float(correct*100/ len(data))

    def __predictionForward(self, sentence):
        out, _ = self.PreTrainedBiLSTM(sentence)
        finalHiddenState = out[:, -1, :]
        
        x = finalHiddenState
        for layer in self.prediction:
            x = layer(x)
            x = self.RELU(x)

        return x

    def trainPredictionStep(self, input):
        self.predictionOptimizer.zero_grad()
        y_hat = self.__predictionForward(input)
        loss = self.predictionLossFunction(y_hat, input[:1:])
        loss.backward()
        self.predictionOptimizer.step()
        return loss.item()

    def trainPrediction(self):
        bestTestAccuracy = 80
        for epoch in range(ELMOConfig.epochs):
            avgLoss = 0
            for sentence in self.trainDatasetPrediction:
                sentence = sentence.to(self.device)
                loss = self.trainPredictionStep(sentence)
                avgLoss += loss
            avgLoss = avgLoss / len(self.trainDatasetPrediction)
            testAccuracy = self.evaluation()
            print(f"Epoch : {epoch+1} | Loss: {avgLoss} | Test Accuracy : {testAccuracy}")
            if testAccuracy > bestTestAccuracy :
                bestTestAccuracy = testAccuracy
                self.saveModel(f"{self.predictionFileName}_{str(testAccuracy)}.model")

    def predictionEvaluation(self, data = None):
        if data is None :
            data = self.testData

        correct = 0
        predicted = []
        actual = []

        with torch.no_grad():
            for y,x in data:
                x = x.squeeze(2)
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.__predictionForward(x)
                y_hat = torch.argmax(y_hat)
                y = torch.argmax(y)
                predicted.append(y_hat.cpu().detach().numpy())
                actual.append(y.cpu().detach().numpy())
                correct += (y_hat == y)

        return float(correct*100/ len(data))
