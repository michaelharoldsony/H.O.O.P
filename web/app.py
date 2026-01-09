from flask import Flask, render_template
import subprocess
import os

app = Flask(__name__)

# -------- PATHS --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# -------- ROUTES --------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run/<video_name>")
def run_video(video_name):
    video_path = os.path.join(DATA_DIR, video_name)

    # Run OpenCV pipeline EXACTLY like CLI
    subprocess.Popen(
        [PYTHON, "-m", "src.main", video_path],
        cwd=PROJECT_ROOT
    )

    return render_template("index.html", launched=video_name)

if __name__ == "__main__":
    app.run(debug=True)