import torch
import torch.optim as Optimizer
import torch.nn as NeuralNetwork
from torch import sigmoid as Sigmoid
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import random
from alive_progress import alive_bar
from bidict import bidict
from .Config import Word2VecConfig


class CustomDataset(TorchDataset):
    def __init__(self, word2VecDataPoints : list, vocabLength : int) -> None:
        self.word2VecDataPoints = word2VecDataPoints
        self.vocabLength = vocabLength
    
    def __len__(self):
        return len(self.word2VecDataPoints)*(Word2VecConfig.negativeSamples + 1)

    def __getitem__(self, index):
        actualIndex = index // (Word2VecConfig.negativeSamples + 1)

        if index % (Word2VecConfig.negativeSamples + 1) == 0 :
            dataPoint = self.word2VecDataPoints[actualIndex]
            return dataPoint
        else :
            dataPoint = self.word2VecDataPoints[actualIndex]
            randomWord = random.randint(0, self.vocabLength - 1)
            return ( dataPoint[0], randomWord, False )
        
class Word2Vec(NeuralNetwork.Module):
    def __init__(self, mapping : bidict, embeddingSize : int = Word2VecConfig.EmbeddingSize, word2VecDataPoints : list = [] ) -> None:
        self.mapping = mapping
        super().__init__()
        
        self.deviceString = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.deviceString = 'cpu'
        self.EmbeddingSize = embeddingSize
        self.contextEmbedding = NeuralNetwork.Embedding(num_embeddings = len(self.mapping), embedding_dim = embeddingSize)
        self.wordEmbedding = NeuralNetwork.Embedding(num_embeddings = len(self.mapping), embedding_dim = embeddingSize)
        self.criterion = NeuralNetwork.BCELoss()
        self.optimizer = Optimizer.Adam(self.parameters(), lr = 0.002)
        self.dataset = CustomDataset(word2VecDataPoints, len(self.mapping))
        self.data = DataLoader(self.dataset, 
                               batch_size = Word2VecConfig.batchSize, 
                               shuffle = True, 
                               pin_memory=True,
                               pin_memory_device=self.deviceString,
                               num_workers=8)
        print(f"Using {self.deviceString} as device.")
        self.trained = False
        self.device = torch.device(self.deviceString)
        self.to(self.device)
        
    def forward(self, word, context):
        out1 = self.contextEmbedding(context)
        out2 = self.wordEmbedding(word)
        out = torch.bmm(out1.unsqueeze(1), out2.unsqueeze(2)).squeeze()
        return Sigmoid(out)

    def trainEmbeddings(self, numEpochs : int = Word2VecConfig.epochs, retrain : bool = False):
        try :
            if not retrain :
                if self.trained :
                    print("Word2Vec already trained. Skipping...")
                    return

            for epoch in range(numEpochs):
                avgLoss = 0
                with alive_bar(len(self.data), force_tty = True) as bar:
                    for word, context, label in self.data:
                        word = word.to(self.device)
                        context = context.to(self.device)
                        label = label.float().to(self.device)
                        
                        output = self(word, context)
                        loss = self.criterion(output, label)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        avgLoss += loss.item()
                        self.optimizer.step()
                        bar()
                    avgLoss /= len(self.data)
                    print(f"Epoch : {epoch+1}, Loss : {avgLoss:.4f}")
            self.trained = True 
        except:
            import traceback
            traceback.print_exc()

    def getWordEmbedding(self, word : str):
        try :
            if not self.trained :
                self.trainEmbeddings()
        except:
            import traceback
            traceback.print_exc()
        finally :
            wordIndex = torch.tensor([self.mapping[word]], dtype = torch.long).to(self.device)
            return self.wordEmbedding(wordIndex) + self.contextEmbedding(wordIndex)
