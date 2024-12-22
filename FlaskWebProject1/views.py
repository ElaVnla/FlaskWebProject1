from datetime import datetime
from flask import render_template, request, jsonify
from FlaskWebProject1 import app
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

model_path = r'C:\Users\erika\source\repos\FlaskWebProject1\FlaskWebProject1\models\my_model.h5'
model = load_model(model_path)


# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image (the function you provided)
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 256, 256, 3)
    return img_array

@app.route('/')
@app.route('/home', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files["file"]

        # If no file is selected
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join("static", filename)
            file.save(file_path)

            # Preprocess the image and predict
            processed_image = preprocess_image(file_path)
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            # Replace with your actual class names
            class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]

            result = {
                "predicted_class": class_names[predicted_class],
                "confidence": round(confidence, 2),
                "image_url": file_path  # Provide image URL to be displayed
            }

            return render_template(
                'index.html',  # Make sure the HTML page renders with the result
                result=result,
                title="Home Page",
                year=datetime.now().year
            )

    # For GET request or if no file is uploaded
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year
    )