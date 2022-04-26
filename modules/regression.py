import pickle
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


def load_model():
    with open('../Model/model.json', 'r') as f:
        json_model = f.read()

    model = tf.keras.models.model_from_json(json_model)
    model.load_weights('../Model/weights.h5')
    model.compile()
    return model


# TODO: Convert array of numerical values into scaled values
def convert_guess():
    pass


# TODO Pass scaled array into model to obtain prediction
def get_prediction():
    pass


saved_model = load_model()
standard_scaler = pickle.load(open('../Model/standard_scaler.pkl', 'rb'))
arr = np.array([0.0, 0.0, 0.6, 0.0, 25.0, 7.21, 0.608, 287])
arr = arr.reshape(-1, 1)
arr = standard_scaler.transform(arr, copy=None)


arr = np.expand_dims(arr,  axis=0)
pred = saved_model.predict(arr)
converted_pred = standard_scaler.inverse_transform(pred)
print(converted_pred[0][0])
