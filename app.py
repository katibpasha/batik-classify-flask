from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import pickle
from skimage.feature import local_binary_pattern
import numpy as np
import pandas as pd

app = Flask(__name__)

# model loading
model_knn = pickle.load(open('./model/knn.pkl', 'rb'))
model_svc = pickle.load(open('./model/svc.pkl', 'rb'))
model_nb = pickle.load(open('./model/nb.pkl', 'rb'))

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


radius = 2
n_points = 8 * radius
METHOD = 'uniform'

# make function of LBP


def local_binary(img):
    img_lbp = local_binary_pattern(img, n_points, radius, METHOD)
    img_lbp_hist, bins = np.histogram(img_lbp.ravel(), 256, [0, 256])
    img_lbp_hist = np.transpose(img_lbp_hist[0:256, np.newaxis])
    return img_lbp_hist


def predict_label(img):
    knn = model_knn.predict(img)
    return knn


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Error apa ini ya?')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada gambar yang diunggah')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        return render_template('index.html', filename=filename)
    else:
        flash('Ekstensi file yang diterima adalah JPG/JPEG')
        return redirect(request.url)


def predict():
    file = request.files['file']
    x = local_binary(file)
    label_output = predict_label(x)
    return render_template('index.html', prediction=label_output)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
