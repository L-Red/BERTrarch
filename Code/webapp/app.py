import sys
import os
sys.path.append("../")
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from training.neural_nets import MulticlassClassification
from classify_upload import anno_ucdp


TEMPLATES_AUTO_RELOAD = True
DEBUG = True
UPLOAD_FOLDER = "./uploads"
DOWNLOAD_FOLDER = "./downloads"
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

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
      if framework == "UCDP":
        calc_dataframe = anno_ucdp(filename=os.path.join(app.config['UPLOAD_FOLDER'], filename))
        calc_dataframe.to_csv(filename=os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
        return redirect(url_for('download_file', name=filename))
        return None
      elif framework == "ACLED":
        return render_template("index.html")
      else:
        return render_template("index.html")
  else:
    return render_template("index.html")

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["DOWNLOAD_FOLDER"], name)

if __name__ == "__main__":
  app.run(debug=True)
