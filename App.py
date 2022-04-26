import re
from tokenize import Number
from flask import Flask
from flask import request


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/test/")
def json_test():
    bolus = request.args.get('bolus', '', type=int)
    return {
        "title": "sample_itle",
        "number": bolus
    }


if __name__ == "__main__":
    app.run(debug=True)
