from flask import Flask
from flask import request
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


if __name__ == "__main__":
    app.run(debug=True)
