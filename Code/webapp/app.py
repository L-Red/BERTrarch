import sys
import os
sys.path.append("../")
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory, jsonify
import pandas as pd
from training.neural_nets import MulticlassClassification
import classify_upload
import json


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

def filetype(filename):
  return filename.rsplit('.', 1)[1].lower()

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
      if filetype(filename) == 'csv':
        df = pd.read_csv(filename)[:10]
      elif filetype(filename) == 'xlsx':
        df = pd.read_excel(filename)
      data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      if framework == "UCDP":
        calc_dataframe = classify_upload.anno_ucdp(filename=os.path.join(app.config['UPLOAD_FOLDER'], filename))
        calc_dataframe.to_csv(os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
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

@app.route('/sentences', methods=["POST", "GET"])
def sentences ():
  response_object = {'keywords': ['birds','scatt']}
  post_data = request.get_json()
  df = pd.DataFrame(columns=['header'])
  if post_data['inputs'] != None:
    for inp in post_data['inputs']:
      df = df.append(
        pd.Series([inp['sentence']], 
        index=['header']), 
        ignore_index=True
        )
    print(df)
    keywords = classify_upload.anno_ucdp(df)
    # if len(keywords) > 0:
    #     response_object = {'keywords' : keywords['ranked phrases']} 
    output_list = []
    kw_list = keywords.values.tolist()  
    for i in range(len(keywords.index)):     
      output_list.append({'sentence': kw_list[i]})
    response_object = {
      'labels': keywords.columns.tolist(),
      'outputs' : output_list
      }  
  return json.dumps(response_object)


if __name__ == "__main__":
  app.run(debug=True)
