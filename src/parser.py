from typing import Optional
from nltk.tokenize import word_tokenize as Tokenizer
import re
from bidict import bidict
from ordered_set import OrderedSet
from alive_progress import alive_bar

from .Config import Constants, SVDConfig, Structure

class Dataset:
    def __init__(self, 
                 trainPath : str  = Structure.trainPath,
                 testPath  : str  = Structure.testPath,
                 fileName  : str  = "dataset.pkl",
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
                            description = description.lower()

                            self.testData.append( [SVDConfig.startToken] + [word for word in Tokenizer(description)] + [SVDConfig.endToken] )
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
                    self.vocab = OrderedSet(Constants.customTokens)
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
                            self.tokenizedData.append( [SVDConfig.startToken] + [ word for word in Tokenizer(description) ] + [SVDConfig.endToken] )
                            self.vocab.update(set(self.tokenizedData[-1]))
                            self.labels.append(label)
                            bar()
        except:
            import traceback 
            traceback.print_exc()
    
    def getMapping(self):
        try :
            if not hasattr(self, "mapping") or self.mapping is None:
                self.__constructBidict__()
            return self.mapping
        except:
            import traceback 
            traceback.print_exc()

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
