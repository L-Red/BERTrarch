import sys
import os
sys.path.append("../")
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from training.neural_nets import MulticlassClassification


TEMPLATES_AUTO_RELOAD = True
DEBUG = True
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["POST", "GET"])
def index():
  if request.method == "POST":
    framework = request.form["framework"]
    data = request.files["data"]
    if data.filename == '':
      flash('No selected file')
      return redirect(request.url)
    if data and allowed_file(data.filename):
      filename = data.filename
      data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return redirect(url_for('download_file', name=filename))
    if framework == "UCDP":
      
      return None
    elif framework == "ACLED":
      return render_template("index.html")
    else:
      return render_template("index.html")
  else:
    return render_template("index.html")

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == "__main__":
  app.run(debug=True)
