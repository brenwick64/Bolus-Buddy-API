import json
from audioop import cross
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from modules.regression import BolusRegression


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/bolus/")
@cross_origin()
def json_test():
    bolus = request.args.get('bolus', '', type=str)
    regression = BolusRegression()
    string_prediction = str(round(regression.get_prediction(bolus), 2))
    return {
        "label": "BG_1_hour",
        "prediction": str(string_prediction)
    }


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
            prediction_payload = json.dumps(
                regression.compute_optimal_bolus(validData))
            return prediction_payload


# Helper Functions


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


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
