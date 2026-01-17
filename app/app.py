from flask import Flask, request, render_template

from torch_utils import RSNAModelService

app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    #xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# A service instance creation
model_service =RSNAModelService()

CLASS_NAMES = ["Normal", "Pneumonia"]
    
@app.route("/", methods =["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    file = request.files.get("file")

    if file is None or file.filename == "":
        return render_template("result.html", error = "No file uploaded")
    
    try:
        prediction_tensor = model_service.predict_from_bytes(file.read())
        prediction = CLASS_NAMES[prediction_tensor.item()]

        return render_template(
            "result.html",
            prediction=prediction,
        )
    except Exception as e:
        return render_template("result.html", error=str(e))

# @app.route('/predict', methods = ['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file is None or file.filename == "":
#             return jsonify({'error': 'no file'})
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'format not supported'})
        
#         try:
#             img_bytes = file.read()
#             tensor = transform_image(img_bytes)
#             prediction = get_prediction(tensor)
#             data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
#             return jsonify(data)
#         except:
#             return jsonify({'error': 'error during prediction'})
#     # 1. load image
#     # 2. image --> tensor
#     # 3. prediction
#     # 4. return json
#     return jsonify({'result': 1})

 
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True) 