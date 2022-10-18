import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
from modules.regression import BolusRegression


app = Flask(__name__)
CORS(app, resources='*')
app.config['CORS_HEADERS'] = 'Content-Type'

"""
    ROUTE - '/'
    ALLOWED METHOD(S) - GET
    DESCRIPTION - Provides info to users about the API and accessable routes
"""
# TODO: Finish API guide


@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hello, World!</p>"


"""
    ROUTE - '/prediction'
    ALLOWED METHOD(S) - GET, POST
    DESCRIPTION - Accepts a JSON payload of valid diabetic user data and
                  returns a prediction JSON payload. 
"""


@app.route("/prediction", methods=['GET', 'POST'])
@cross_origin()
def get_prediction():
    regression = BolusRegression()
    if request.method == 'GET':
        return 'Error - No POST request found'

    if request.method == 'POST':
        validData = validate_data(request.get_json())
        if not validData:
            return {"error": "invalid json data"}
        else:
            prediction_payload = regression.get_predictions(
                validData)
            return prediction_payload


"""
    Helper function to validate the correct JSON data going into 
    the route.
"""


def validate_data(data):
    validData = {}
    # Checks if required fields exist
    if not {'carbs', 'bg', 'basal', 'bolus'} <= data.keys():
        return False
    # Checks if data is of correct type
    for key in data.keys():
        try:
            validData[key] = float(data[key])
        except:
            return False

    return validData


"""
    Entrypoint for Bolus_Buddy_API
"""
if __name__ == "__main__":
    app.run(threaded=True, port=5000)
