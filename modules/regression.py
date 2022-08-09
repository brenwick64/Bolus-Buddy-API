from cgi import test
from imp import load_module
from json import load
import pickle
import tensorflow as tf
import numpy as np
import random


class BolusRegression:

    # TODO add shape constants in constructor
    def __init__(self):
        self.model = self.load_model()
        self.scaler = self.load_scaler()

    def load_model(self):
        with open('./Model/model.json', 'r') as f:
            json_model = f.read()
        model = tf.keras.models.model_from_json(json_model)
        model.load_weights('./Model/weights.h5')
        model.compile()
        return model

    # TODO configuration for different types of scalers
    def load_scaler(self):
        standard_scaler = pickle.load(
            open('./Model/standard_scaler.pkl', 'rb'))
        return standard_scaler

    # TODO: Convert array of numerical values into scaled values
    def get_prediction(self, guess):
        arr = np.array([0.0, 0.0, 0.6, 0.0, 25.0, guess, 0.608, 287])
        print(arr)
        # Per error log - used to fit standard scaler
        arr = arr.reshape(-1, 1)

        arr = self.scaler.transform(arr, copy=None)
        arr = np.expand_dims(arr,  axis=0)
        pred = self.model.predict(arr)
        converted_pred = self.scaler.inverse_transform(pred)
        return converted_pred[0][0]

    def test_get_bolus(self, dataDict):
        print(dataDict)
        testDict = {'aggressiveBolus': 5.5,
                    'reccomendedBolus': 4.57, 'passiveBolus': 3.7}
        return testDict

    def predict_bolus(self, payload, bolus_guess):
        injectedArr = np.array(
            [payload['carbs'], bolus_guess, payload['basal'], payload['bg']])

        # Per error log - used to fit standard scaler
        injectedArr = injectedArr.reshape(-1, 1)
        injectedArr = self.scaler.transform(injectedArr, copy=None)
        injectedArr = np.expand_dims(injectedArr,  axis=0)
        prediction = self.model.predict(injectedArr)
        converted_pred = self.scaler.inverse_transform(prediction)

        return converted_pred[0][0]

    def compute_optimal_bolus(self, payload):
        guess = payload['bolus']
        best_bolus = guess
        for i in range(35):
            guess = guess + random.uniform(-0.5, 0.5)
            guess_pred = self.predict_bolus(payload, guess)
            best_pred = self.predict_bolus(payload, best_bolus)
            if((abs(guess_pred) - 120.0) < (abs(best_pred) - 120.0)):
                best_bolus = guess
            else:
                guess = best_bolus
        print('best prediction: ' + str(best_pred))

        return str(best_bolus)
