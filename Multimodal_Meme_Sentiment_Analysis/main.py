from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from PIL import Image
from skimage.io import imread
from skimage import filters
import numpy as np
import pickle
import sklearn

app = Flask(__name__)

@app.route('/')
def upload_files():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        image_features = get_uploaded_image_features(f.filename)
        image_model = pickle.load(open('./final_result.pkl', 'rb'))
        outcome_label = image_model.predict(image_features)
        outcome_label = remap(outcome_label)

        return render_template('show_results.html', data = outcome_label)

def remap(outcome_label):
    if outcome_label == -1:
        return 'negative'
    outcome_label = str(outcome_label)[1] # because it is in format [3]
    order = { '1' : 'Positive', '0' : 'Neutral'}
    return order.get(outcome_label)

def get_sobel_features(image):
    ed_sobel = filters.sobel(image)
    return ed_sobel

def resize_image(imgpath):
        img = Image.open(imgpath)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img2 = img.resize((30,30))
        save_path = imgpath.split('.')[0] + '_resized.' + imgpath.split('.')[1]
        # print("Save path : ", save_path)
        new_name = save_path    
        img2.save(new_name)
        return new_name

def get_uploaded_image_features(filename):
    path = r'D:\Pppp\Project'
    imgpath = imgpath = path + "\\" + filename
    print('uploaded image path : ', imgpath)
    new_name = resize_image(imgpath)
    image1 = imread(new_name, as_gray=True)   # first resize the uploaded image
    features = np.array(get_sobel_features(image1)).reshape(1,900)
    return features

		
if __name__ == '__main__':
   app.run(debug = True)