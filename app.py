from flask import Flask,abort,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
data_base_dir = r'uploads/'
labels = ['covid','not_covid']
model = load_model('covid-cough-model-100acc.h5')
def extract_features_from_audio(audio_path):
    # setting path
    file_name = os.path.join(data_base_dir,audio_path)
    # check if the file is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        # X-> audio_time_series_data; sample_rate-> sampling rate
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
        # extraccting Mel-Frequeny Cepstral Coeficients feature from data
        # y -> accepts time-series audio data; sr -> accepts sampling rate
        # n_mfccs -> no. of MFCCs to return
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis = 0)
    except Exception as e:
        print("Error encountered while parsing file: ", e)
        return None, None
    # store mfccs features
    feature = mfccs
    
    return feature
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return str("Health!")

@app.route('/upload',methods = ['GET','POST'])
def upload_file():
    if request.method =='POST':
        file = request.files['file[]']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            features = extract_features_from_audio(filename)
            B = np.reshape(features, (-1, 40))
            preds = model.predict(B)
            preds = labels[np.argmax(preds[0])]
    return render_template('index.html',result = preds)


if __name__ == '__main__':
    app.run(debug = True)