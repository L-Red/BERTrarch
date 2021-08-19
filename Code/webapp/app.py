import sys
import os
sys.path.append("../")
from flask import Flask, render_template, request
from training.neural_nets import MulticlassClassification
TEMPLATES_AUTO_RELOAD = True
DEBUG = True

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
  if request.method == "POST":
    framework = request.form["framework"]
    data = request.form["data"]
  else:
    return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)
