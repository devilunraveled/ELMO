from collections import Counter
from functools import cache
from typing import Optional
from nltk.tokenize import word_tokenize as Tokenizer
import re
from bidict import bidict
from ordered_set import OrderedSet
from scipy.sparse import lil_matrix as LilMatrix
from scipy.sparse.linalg import svds as SparseSVD
from numpy.linalg import svd as DenseSVD
from alive_progress import alive_bar
import pickle
import torch

from .Config import Constants, SVDConfig, Structure, Word2VecConfig
from .word2Vec import Word2Vec
class Dataset:
    def __init__(self, 
                 trainPath : str  = Structure.trainPath,
                 testPath  : str  = Structure.testPath,
                 fileName  : str  = "dataset.pkl",
                 svdLoad   : bool = False,
                 word2VecLoad : bool = False
                 ) -> None:
        """
            @param trainPath: Path to train.csv
            @param testPath: Path to test.csv
        """
        self.trainPath = trainPath
        self.testPath = testPath
        self.fileName = fileName

        self.vocab = OrderedSet(set())
        self.tokenizedData = []
        self.labels = []

        self.testData = []
        self.testLabels = []
        self.data = None

        if word2VecLoad:
            self.word2VecEmbedding = self.loadData(fileName = f"word2vecEmbedding_{Word2VecConfig.contextWindow}.pkl", dataName = "word2vecEmbedding")

        if svdLoad:
            self.SVDEmbedding = self.loadData(fileName = f"svdEmbeddings_{SVDConfig.contextWindow}.pkl", dataName = "SVDEmbedding")
            print(self.SVDEmbedding)

    def saveData(self, data, fileName) -> None:
        try :
            savePathFile = Structure.corpusPath + fileName
            with open(savePathFile, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved dataset to {savePathFile}.")
        except:
            print("Failed to save dataset.")
    
    def loadData(self, fileName, dataName : str = "data"):
        try :
            savePathFile = Structure.corpusPath + fileName
            with open(savePathFile, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {dataName} from {savePathFile}.")
            return data
        except :
            print(f"Could not find {dataName}. It will be built from scratch if needed.")

    def getVocabulary(self) -> OrderedSet | set:
        try :
            if not self.vocab :
                self.getData()
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return self.vocab
    
    def getTestData(self):
        try :
            if not self.testData :
                self.testData = []
                print(f"Loading test-data from {self.testPath}.")
                with open(self.testPath, "r") as f:
                    skipped = False
                    numLines = sum(1 for _ in f)
                    f.seek(0)
                    with alive_bar(numLines-1, force_tty = True) as bar:
                        for line in f:
                            # Skip the first line as it contains meta information.
                            if not skipped: 
                                skipped = True
                                continue
                            
                            label = int(line.split(",")[0])
                            description : str = ' '.join(line.split(",")[1:]).strip()
                            description = re.sub(SVDConfig.cleanser, " ", description)

                            self.testData.append( [word for word in Tokenizer(description)] )
                            self.testLabels.append(label)
                            bar()
        except:
            import traceback 
            traceback.print_exc()
    def getData(self) -> Optional[list]:
        try :
            if self.data is None:
                self.data = []
                print(f"Loading data from {self.trainPath}.")
                with open(self.trainPath, "r") as f:
                    self.vocab = {SVDConfig.startToken, SVDConfig.endToken, SVDConfig.unkToken}
                    skipped = False
                    numLines = sum(1 for _ in f)
                    f.seek(0)
                    with alive_bar(numLines-1, force_tty = True) as bar:
                        for line in f:
                            # Skip the first line as it contains meta information.
                            if not skipped: 
                                skipped = True
                                continue
                            
                            label = int(line.split(",")[0])
                            description : str = ' '.join(line.split(",")[1:]).strip()
                            description = re.sub(SVDConfig.cleanser, " ", description)

                            self.data.append( SVDConfig.startToken + ' ' + description + ' ' + SVDConfig.endToken )
                            self.tokenizedData.append([ word for word in Tokenizer(description) ])
                            self.vocab.update(set(self.tokenizedData[-1]))
                            self.labels.append(label)
                            bar()
        except:
            import traceback 
            traceback.print_exc()

    def getCoocurenceMatrix(self, sparse : bool = False) -> Optional[dict] | Optional[list[list]]:
        """ 
            This method returns a co-occurence matrix, 
            it can be of a sparse representation if the 
            sparse parameter is set True. Otherwise, it 
            creates the co-occurence based on the first k 
            words as found.
            @param sparse: Set to True if you want a sparse matrix representation.
        """
        try :
            if sparse:
                self.SparseCoOcMatrix = self.__getSparseCoocurenceMatrix__()
            else:
                self.StochasticCoOcMatrix = self.__getStochasticCoocurenceMatrix__()
        except:
            import traceback 
            traceback.print_exc()
    
    def __getSparseCoocurenceMatrix__(self):
        """
            Returns a sparse co-occurence matrix
            as a list of lists.
        """
        if not hasattr(self, "mapping") or self.mapping is None:
            self.__constructBidict__()
        
        print("Creating sparse co-occurence matrix...")
        matrixDimension = len(self.vocab)
        sparseMatrix = LilMatrix((matrixDimension, matrixDimension), dtype = "float64")
        
        with alive_bar(len(self.tokenizedData), force_tty = True) as bar:
            for sentence in self.tokenizedData:
                for index, token in enumerate(sentence):
                    windowStart = max(0, index - SVDConfig.contextWindow)
                    for windowToken in sentence[windowStart : index]:
                        sparseMatrix[self.mapping[token], self.mapping[windowToken]] += 1
                        sparseMatrix[self.mapping[windowToken], self.mapping[token]] += 1
                bar()
        return sparseMatrix

    def __getStochasticCoocurenceMatrix__(self):
        """
            Returns a stochastic co-occurence matrix
            based on the number of words in the config.
        """

        if not hasattr(self, "mapping") or self.mapping is None:
            self.__constructBidict__()

        def acceptable(word : str ) -> bool :
            if self.mapping[word] < SVDConfig.numWords:
                return True
            return False
        
        coOccMat = [ [ 0 for _ in range(SVDConfig.numWords) ] for _ in range(SVDConfig.numWords) ]

        print("Creating stochastic co-occurence matrix...")
        
        for i in range(min( len(self.vocab), SVDConfig.numWords)):
            word = self.mapping.inverse[i]
            context = self.__getWordContext__(word)
            for contextWord,freq in context.items():
                if acceptable(contextWord):
                    coOccMat[i][self.mapping[contextWord]] = freq
         
        return coOccMat

    def __getContextInLines__(self, word : str, line : list[str]) -> Optional[list]:
        """
            Returns the context of the word in a line.
            @param word: The word to search for.
            @param line: The line to search in.
            @return: The context of the word in the line.
        """
        context = []
        try :
            indices = [ i for i,token in enumerate(line) if token == word ]
            
            for index in indices:
                start = max(0,index - Word2VecConfig.contextWindow)
                end = min(len(line) - 1, index + Word2VecConfig.contextWindow)
                
                context.extend(line[start : end + 1])
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return context
    
    @cache
    def __getWordContext__(self, word : str) -> Counter:
        """
            Returns the context of the word in the corpus.
            @param word: The word to search for.
            @return: The context of the word in the corpus.
        """
        context = Counter()
        try :
            if not self.tokenizedData :
                self.getData()

            for line in self.tokenizedData:
                context.update(self.__getContextInLines__(word = word, line = line))
            
        except:
            import traceback 
            traceback.print_exc()
        finally:
            return context

    def __constructBidict__(self):
        """
            Constructs a bidirectional mapping from the vocabulary to 
            the index for ease of mapping.
        """
        try :
            if not hasattr(self, "vocab") or not self.vocab :
                self.getData()
            
            print("Constructing bidict mapping...")
            self.mapping = bidict({ word : index for index, word in enumerate(self.vocab)} )
            ## Replace the first three mappings to the custom tokens.
        except:
            import traceback 
            traceback.print_exc()
    
    def performSVD(self, sparse = True):
        """
            Performs SVD on the co-occurence matrix
            @param sparse: Set to True if you want a sparse matrix representation.
        """
        
        try :
            if sparse:
                if ( not hasattr(self, 'SparseCoOcMatrix') ):
                    self.getCoocurenceMatrix(sparse = True)
                
                print("Computing SVD on sparse co-occurence matrix...")
                U,S,Vt = SparseSVD(A = self.SparseCoOcMatrix, k = min(SVDConfig.EmbeddingSize, len(self.vocab) - 1) )

                self.U = U
                self.S = S
                self.Vt = Vt
            else:
                if ( not hasattr(self, 'StochasticCoOcMatrix') ):
                    self.getCoocurenceMatrix(sparse = False)
                
                U,S,Vt = DenseSVD(self.StochasticCoOcMatrix)
                
                self.U = U
                self.S = S
                self.Vt = Vt

        except:
            import traceback
            traceback.print_exc()

    def prepareData(self):
        try :
            self.getData()
            self.__constructBidict__()
            self.getWord2VecDataPoints()
        except:
            import traceback 
            traceback.print_exc()
    
    def getWord2VecDataPoints(self):
        self.word2VecDataPoints = []
        print(f"Preparing data for Word2Vec...")
        with alive_bar(len(self.tokenizedData), force_tty=True) as bar:
            for sentence in self.tokenizedData:
                for index, token in enumerate(sentence):
                    window = sentence[max(0,index - Word2VecConfig.contextWindow) : index] + sentence[index + 1 : index + 1 + Word2VecConfig.contextWindow]
                    for word in window:
                        if word in self.vocab:
                            self.word2VecDataPoints.append( (self.mapping[token], self.mapping[word], True) )
                bar()

    def getWordEmbedding(self, word : str) -> list:
        """
            Get the embedding for the word in the vocabulary.
        """
        try :
            if not hasattr(self,'U') :
                self.performSVD(sparse = True)
        except:
            import traceback 
            traceback.print_exc()
            return []
        finally:
            if self.U is None:
                return []
            return self.U[self.mapping[word]]
    
    def prepareWord2VecEmbedding(self) -> None:
        """
            Get the embedding for the word in the vocabulary.
        """
        
        if not hasattr(self, 'word2VecDataPoints'):
            self.prepareData()
        
        try :
            if not hasattr(self, 'word2Vec'):
                print("Initializing Word2Vec...")
                self.word2Vec = Word2Vec(mapping = self.mapping, word2VecDataPoints = self.word2VecDataPoints)
        except:
            import traceback 
            traceback.print_exc()
        finally:
            if self.word2Vec is not None:
                if self.word2Vec.trained is False:
                    print("Training Word2Vec...")
                    self.word2Vec.trainEmbeddings()
    
    def setSVDEmbeddings(self):
        if not hasattr(self, 'U') or self.U is None:
            self.performSVD(sparse = True)
        
        print("Initializing SVD Embeddings...")
        self.SVDEmbedding = {}
        try :
            if self.U is None :
                raise Exception("SVD could not be performed correctly.")
            for word in self.mapping :
                self.SVDEmbedding[word] = self.U[self.mapping[word]]
        except:
            import traceback 
            traceback.print_exc()
        finally:
            if self.SVDEmbedding is not None:
                print("Saving SVD Embeddings...")
                self.saveData(data = self.SVDEmbedding, fileName = f"svdEmbeddings_{SVDConfig.contextWindow}.pkl")

    def getSVDEmbedding(self, word : str ):
        try :
            if not hasattr(self, 'SVDEmbedding') or self.SVDEmbedding is None or not self.SVDEmbedding:
                print(f"Could not get SVDEmbeddings, preparing the same...")
                self.setSVDEmbeddings()
            if self.SVDEmbedding is None :
                raise Exception("SVD could not be performed correctly.")
            return torch.tensor(self.SVDEmbedding.get(word, self.SVDEmbedding[Constants.unkToken]).copy()).float()
        except:
            import traceback 
            traceback.print_exc()
            return torch.tensor([])

    def setWord2VecEmbeddings(self):
        if not hasattr(self, 'word2Vec') or self.word2Vec is None:
            self.prepareWord2VecEmbedding()
        print("Initializing Word2Vec Embeddings...")
        self.word2VecEmbedding = {}

        try :
            if self.word2Vec is None :
                raise Exception("Word2Vec could not be performed correctly.")
            for word in self.mapping :
                self.word2VecEmbedding[word] = self.word2Vec.getWordEmbedding(word).cpu().detach()
        except:
            import traceback 
            traceback.print_exc()
        finally:
            if self.word2VecEmbedding is not None:
                self.saveData(data = self.word2VecEmbedding, fileName = f"word2vecEmbedding_{Word2VecConfig.contextWindow}.pkl")
    
    def getWord2VecEmbedding(self, word : str):
        try :
            if not hasattr(self, 'word2VecEmbedding') or self.word2VecEmbedding is None or not self.word2VecEmbedding:
                self.setWord2VecEmbeddings()
            if self.word2VecEmbedding is None :
                raise Exception("Word2Vec could not be performed correctly.")
            return self.word2VecEmbedding.get(word, self.word2VecEmbedding[Constants.unkToken])
        except:
            import traceback 
            traceback.print_exc()
            return torch.tensor([])
