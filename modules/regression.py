from imp import load_module
from json import load
import pickle
import tensorflow as tf
import numpy as np


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
