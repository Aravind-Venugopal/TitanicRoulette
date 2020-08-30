import pickle, joblib
from flask import Flask, request, render_template
import pandas as pd
from utils import HexTransformer

app = Flask(__name__, template_folder="templates")


@app.before_first_request
def load_model():
    global pipe
    pipe = joblib.load("model.pkl")
    # with open("model.pkl", "rb") as f:
    #     pipe = joblib.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def predict():
    form = request.form
    ## Required params
    # pclass:    Ticket class
    # sex:    Sex
    # age:    Age in years
    # sib_sp:    # of siblings / spouses aboard the Titanic
    # par_ch:    # of parents / children aboard the Titanic
    # fare:    Passenger fare
    # embarked:    Port of Embarkation
    # deck:    Deck of cabin(0-8)

    # print(form)
    # new = pd.DataFrame(
    #     {
    #         "diameter": [float(form["diameter"])],
    #         "weight": [float(form["weight"])],
    #         "hexcode": [form["color"]],
    #     }
    # )
    # fruit = pipe.predict(new)[0]
    # orange = fruit == 'orange'
    # return render_template("result.html", orange=orange)

    new = [int(form['pclass']), int(form['sex']),
           int(form['age']), int(form['sib_sp']), int(form['par_ch']), float(form['fare']), int(form['embarked']), int(form['deck'])]
    print(new)
    survival = pipe.predict([new])[0]
    survival = survival == 1
    print(survival)

    return render_template("result.html", survival = survival)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
