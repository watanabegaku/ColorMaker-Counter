from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

#upload_folder =app.config['UPLOAD_FOLDER']
#if not os.path.exists(upload_folder):
#    os.makedirs(upload_folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return redirect(request.url)
        image_file = request.files['image_file']
        if image_file and allowed_file(image_file.filename):
            filename=image_file.filename
            image_path =os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
            output_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
            totals, results = count_color_markers(image_path, output_csv)
            return render_template('result.html', totals=totals, results=results, csv_file=output_csv)
    return render_template('index.html')
    
#@app.route('/', methods=['GET', 'POST'])
#def save():
    csv_file_path = request.form['csv_file_path']
    image_path = request.form['image_path']
    results, totals =count_color_markers(image_path)
    if results:
        save_to_csv(results, totals,csv_file_path)
        return f'結果を{csv_file_path}に保存しました'
    else:
        return 'CSVファイルの保存に失敗しました'
    
def count_color_markers(image_path, output_csv):
    image = cv2.imread(image_path)
    cropped_image =image[44:-340,22:-15]
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'red':((0,120,70), (10,255,255)),
        'blue':((100,150,50),(140,255,255)),
        'light_gray':((0,0,200),(180,30,255))
    }

    totals = {color:0 for color in color_ranges}
    results = []
    for x in range(0,hsv.shape[1] ,176):
        for y in range(0,hsv.shape[0] ,128):
            window = hsv[y:y +128, x:x +176]
            color_counts = {color:0 for color in color_ranges}
            for i in window:
                for pixel in i:
                    for color, (lower, upper) in color_ranges.items():
                        mask =cv2.inRange(window,lower,upper)
                        countours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        if np.all(pixel >= np.array(lower)) and np.all(pixel <= np.array(upper)):
                            color_counts[color] = len(countours)
                            totals[color] += len(countours)
            results.append({'x': x, 'y': y, **color_counts})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return totals, results

#def save_to_csv(results, totals, csv_file_path):
    df = pd.DataFrame(results,totals)
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    app.run(debug=True)








