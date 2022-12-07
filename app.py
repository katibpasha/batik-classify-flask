from flask import Flask, flash, request, redirect, url_for, render_template
from importlib.resources import path
import os
from werkzeug.utils import secure_filename
import pickle
from skimage.feature import local_binary_pattern
import numpy as np
import cv2 as cv

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


def predict_knn(img):
    img_lbp = local_binary_pattern(img, n_points, radius, METHOD)
    img_lbp_hist, bins = np.histogram(img_lbp.ravel(), 256, [0, 256])
    img_lbp_hist = np.transpose(img_lbp_hist[0:256, np.newaxis])

    knn = model_knn.predict(img_lbp_hist)
    return knn


def predict_svc(img):
    img_lbp = local_binary_pattern(img, n_points, radius, METHOD)
    img_lbp_hist, bins = np.histogram(img_lbp.ravel(), 256, [0, 256])
    img_lbp_hist = np.transpose(img_lbp_hist[0:256, np.newaxis])

    svc = model_svc.predict(img_lbp_hist)
    return svc


def predict_nb(img):
    img_lbp = local_binary_pattern(img, n_points, radius, METHOD)
    img_lbp_hist, bins = np.histogram(img_lbp.ravel(), 256, [0, 256])
    img_lbp_hist = np.transpose(img_lbp_hist[0:256, np.newaxis])

    nb = model_nb.predict(img_lbp_hist)
    return nb


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
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

        path = './static/uploads/' + file.filename
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        output_knn = predict_knn(img)
        output_svc = predict_svc(img)
        output_nb = predict_nb(img)
        return render_template('index.html', filename=filename, pred_knn=output_knn, pred_svc=output_svc, pred_nb=output_nb)
    else:
        flash('Ekstensi file yang diterima adalah JPG/JPEG')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
