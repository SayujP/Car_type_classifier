import os
import torch
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
import joblib

clf=joblib.load('logistic_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Model setup (load pre-trained model)
model = models.resnet18(weights=None)
model.load_state_dict(torch.load('resnet_full.pth'))
model.eval()

model.fc = torch.nn.Identity()


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set up the upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Helper function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')

    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature = model(img)
        feature = feature.squeeze().numpy().reshape(1, -1)  # convert to (1, 512) NumPy array
        output = clf.predict(feature)  # returns class index directly
    return int(output[0])


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        prediction = predict_image(filepath)
        class_name = 'Sedan' if prediction == 1 else 'Non-Sedan'

        return render_template('index.html', filename=filename, class_name=class_name)

    return 'Invalid file format!'

if __name__ == '__main__':
    app.run(debug=True)
