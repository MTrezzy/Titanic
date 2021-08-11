import sklearn
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier


class Titanic_Classifier:
    '''Create a object to implement the Titanic classifier.
    '''
    def __init__(self, model_path:str):
        self.model = self.get_model(model_path)
        self.survived = {
            0:'False',
            1:'True'
        }
    
    def get_model(self, model_path:str) -> RandomForestClassifier:
        '''Open the pkl file which store the model.
        Arguments: 
            model_path: Path model with pkl extension
        
        Returns:
            model: Model object
        '''

        with open(model_path,"rb") as f:
            model = pickle.load(f)
        
        return model

    def make_prediction(self, features:dict)->str:
        '''Predicts the survivors.
        Argument:
            features: list
        
        return:
            Survived: str
        '''
        features = np.array(list(features.values()))
        pred = self.model.predict(features.reshape(1,-1))[0]
        survived_pred = self.survived[pred]
        return survived_pred