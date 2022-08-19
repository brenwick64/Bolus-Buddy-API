import json
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

    def compute_bolus(self, payload, bias):
        guess = payload['bolus']
        best_bolus = guess
        # Reccomended Bolus
        for i in range(35):
            guess = guess + random.uniform((bias * -1), (bias))
            guess_pred = self.predict_bolus(payload, guess)
            best_pred = self.predict_bolus(payload, best_bolus)
            # Drifts optimal bolus over each iteration
            if((abs(guess_pred) - 120.0) < (abs(best_pred) - 120.0)):
                best_bolus = guess
            else:
                guess = best_bolus

        return best_bolus

    def get_predictions(self, payload):
        recommended_bolus = self.compute_bolus(payload, 0.5)
        passive_bolus = self.compute_bolus(payload, 0.25)
        aggressive_bolus = self.compute_bolus(payload, 0.75)

        return json.dumps({"recommended_bolus": recommended_bolus,
                           "passive_bolus": passive_bolus,
                           "aggressive_bolus": aggressive_bolus
                           })
