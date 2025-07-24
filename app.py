from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Lazy-load YOLO model
def get_model():
    return YOLO('best.pt')

@app.route('/public/<path:filename>')
def public_files(filename):
    return send_from_directory('public', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None

    # โหลด HTML template จาก index.html
    with open('index.html', 'r', encoding='utf-8') as f:
        html_template = f.read()

    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            new_filename = f"{uuid.uuid4().hex}.{ext}"
            img_path = os.path.join(UPLOAD_FOLDER, new_filename)
            file.save(img_path)

            model = get_model()
            results = model(img_path, conf=0.25)  # ปรับ confidence ได้ตามต้องการ

            if results:
                img_array = results[0].plot()
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                output_path = os.path.join(OUTPUT_FOLDER, new_filename)
                img_pil.save(output_path)
                result_image = new_filename

    return render_template_string(html_template, result_image=result_image)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com;"
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)