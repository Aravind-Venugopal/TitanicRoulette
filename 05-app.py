import pickle, joblib
from flask import Flask, request, render_template
import pandas as pd
from utils import HexTransformer

app = Flask(__name__, template_folder="templates")


@app.before_first_request
def load_model():
    global pipe
    pipe = joblib.load("model.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def predict():
    form = request.form
    print(form)
    new = [int(form['pclass']), int(form['sex']),
           int(form['age']), int(form['sib_sp']), int(form['par_ch']), float(form['fare']), int(form['embarked']), int(form['deck'])]
    print(new)
    survival = pipe.predict([new])[0]
    survival = survival == 1
    print(survival)

    return render_template("result.html", survival=survival)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
