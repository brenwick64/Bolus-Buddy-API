import warnings
import json
import pickle
import tensorflow as tf
import numpy as np
import random


def warn(*args, **kwargs):  # TODO - Bandaid solution to prevent flooding with SKlearn warking logs
    pass


warnings.warn = warn


"""
    Class for accepting a JSON payload of diabetic bolus event data and
    converting it into a predictive response.
"""


class BolusRegression:

    """
        Constructor for BolusRegression: 
    """
    # TODO add shape constants in constructor

    def __init__(self):
        self.model = self.load_model()
        self.scaler = self.load_scaler()
    """
        Loads and compiles a keras tensorflow model using a JSON export 
        of the model, and a serialized binary file of the weights and biases.
    """

    def load_model(self):
        with open('./Model/model.json', 'r') as f:
            json_model = f.read()
        model = tf.keras.models.model_from_json(json_model)
        model.load_weights('./Model/weights.h5')
        model.compile()
        return model

    # TODO configuration for different types of scalers

    """
        Loads the saved scaler, normalized to the training data, for converting
        numerical values into normalized vectors.
    """

    def load_scaler(self):
        standard_scaler = pickle.load(
            open('./Model/standard_scaler.pkl', 'rb'))
        return standard_scaler

    """
        Given an array of conditions, uses the trained model to compute the blood
        glucose one hour after the bolus would be delivered.
        
        Params:
            payload : str               - JSON payload containing user's bolus event data
            bolus_guess : number        - A generated guess to input into the compiled model
            
        Returns:
            converted_pred : number     - The model's estimate of bg value given the input data

    """

    def predict_bg(self, payload, bolus_guess):
        injectedArr = np.array(
            [payload['carbs'], bolus_guess, payload['basal'], payload['bg']])

        # Per error log - used to fit standard scaler
        injectedArr = injectedArr.reshape(-1, 1)
        injectedArr = self.scaler.transform(injectedArr, copy=None)
        injectedArr = np.expand_dims(injectedArr,  axis=0)
        prediction = self.model.predict(injectedArr)
        converted_pred = self.scaler.inverse_transform(prediction)
        return converted_pred[0][0]

    """
        Iterates a preset amount of guesses accompanied with a numerical bias in order to 
        "drift" towards a more optimal bolus value.
        
        Params:
            payload : str               - JSON payload containing user's bolus event data
            bias : number               - A numeric value which will raise or lower each iteration's guess by that amount
            
        Returns: 
            best_bolus : number         - The final result of the fixed iterations guessing and drift

    """

    def compute_bolus(self, payload, bias):
        guess = payload['bolus']
        best_bolus = guess
        # Reccomended Bolus
        for i in range(35):
            guess = guess + random.uniform((bias * -1), (bias))
            guess_pred = self.predict_bg(payload, guess)
            best_pred = self.predict_bg(payload, best_bolus)
            print('bias:', bias, ' - ', 'guess_pred:',
                  guess_pred, ' best_pred:', best_pred)
            # Drifts optimal bolus over each iteration
            if((abs(guess_pred) - 120.0) < (abs(best_pred) - 120.0)):
                best_bolus = guess
            else:
                guess = best_bolus

        return best_bolus

    """
        Generates three different bolus calculations of varying bias implementations
        
        Params:
            payload : str               - JSON payload containing user's bolus event data
                        
        Returns: 
            json.dumps : str            - JSON object containing 3 categories of model evaluation
    """

    def get_predictions(self, payload):
        recommended_bolus = self.compute_bolus(payload, 0.5)
        passive_bolus = self.compute_bolus(payload, 0.25)
        aggressive_bolus = self.compute_bolus(payload, 0.75)

        return json.dumps({"recommended_bolus": recommended_bolus,
                           "passive_bolus": passive_bolus,
                           "aggressive_bolus": aggressive_bolus
                           })
