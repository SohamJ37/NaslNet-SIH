from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import UserMixin, login_user, LoginManager, current_user, logout_user
from forms import PostForm, RegisterForm, LoginForm
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename
from pathlib import Path
import torch
import os
from data import breed_data

from main import SimpleClassifier, predict_image, device

class_names = sorted(list(breed_data["breeds"].keys()))
model = SimpleClassifier(num_classes=41)
model.load_state_dict(torch.load("breed.pth", map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)
app.secret_key = "change-me"

bootstrap = Bootstrap5(app)

UPLOAD_FOLDER = Path("Uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "gif", "pdf"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024

def allowed_file(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower in ALLOWED


@app.route("/test", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist("files")   # <- grabs all 3
        files = [f for f in files if f and f.filename]  # drop empties

        if len(files) != 3:
            return "Please upload exactly 3 images", 400

        # optional: check extensions
        def allowed(fn): 
            return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED
        for f in files:
            if not allowed(f.filename):
                return f"File type not allowed: {f.filename}", 400

        # accumulate probabilities across the three images
        total_probs = torch.zeros(len(class_names))  # CPU tensor
        used_names = []

        for f in files:
            fname = secure_filename(f.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            f.save(path)
            # if your predict_image returns (label, conf, probs):
            # _, _, probs = predict_image(path, model, device)
            probs = predict_image(path, model, device)  # if it returns just probs
            total_probs += probs.cpu()
            used_names.append(fname)

        top_idx = int(torch.argmax(total_probs))
        prediction = class_names[top_idx]
        confidence = float(total_probs[top_idx].item())

        return redirect(url_for("information_page", breed_name=prediction))
    return render_template("upload.html")



@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact_page():
    if request.method == "POST":
        data = request.form
        return redirect(url_for("home_page"))
    return render_template("contact.html")

@app.route("/information/<string:breed_name>")
def information_page(breed_name):
    return render_template("information.html", breed_name=breed_name, data_dict=breed_data["breeds"][breed_name])
    

if __name__ == "__main__":
    app.run(debug=True)



