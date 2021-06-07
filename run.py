# Flask
from flask import Flask, render_template, request, flash
from flask.helpers import url_for
# from werkzeug import secure_filename
from werkzeug.utils import redirect, secure_filename
import json
import os

# CV
from scripts import test

with open('config.json', 'r') as c:
    params  = json.load(c)["params"]
app = Flask(__name__)
app.config['SECRET_KEY'] = "dcc0e3e40867ccaa0d9afc35"
app.config['UPLOAD_FOLDER'] = params['upload_location']

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route("/predict" , methods=['GET', 'POST'])
def uploader():
    if request.method=='POST':
        f = request.files['file1']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        try:
            f.save(file_path)
            predictions = test.predict(file_path)
            # print(file_path)
            os.remove(file_path)
            return render_template("predict.html", predictions=predictions)
        except:
            flash (f"Please upload an Image!", category="danger")
            return redirect(url_for('home_page'))
            

if __name__ == '__main__':
    app.run(debug=True)

