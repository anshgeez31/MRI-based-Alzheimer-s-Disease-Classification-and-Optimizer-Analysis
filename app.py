from flask import Flask, request, render_template
import torch
import torchaudio
import pandas as pd
import os
from flask import Flask, request, render_template, jsonify

from utils.audio_processing import preprocess_audio
from model.model import CoughDetectionModel

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CoughDetectionModel()
model.load_state_dict(torch.load("model/my_model.pth", map_location=device))
model.to(device)
model.eval()

# Load metadata CSV
metadata_df = pd.read_csv("data/challenge1_metadata.csv")
metadata_df["file"] = metadata_df["uuid"].astype(str) + "-recording-1.wav"
metadata_df.set_index("file", inplace=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    filename = file.filename

    waveform, sample_rate = torchaudio.load(file)
    input_tensor = preprocess_audio(waveform, sample_rate).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).item()
        predicted_label = "TB Positive" if pred > 0.5 else "TB Negative"
        confidence = pred if pred > 0.5 else 1 - pred

    # Metadata lookup
    if filename in metadata_df.index:
        actual_tb = "TB Positive" if metadata_df.loc[filename, "TB"] == 1 else "TB Negative"
        age = str(metadata_df.loc[filename].get("age", "N/A"))
        gender = metadata_df.loc[filename].get("gender", "N/A")
    else:
        actual_tb = None
        age = None
        gender = None

    return jsonify({
        'label': predicted_label,
        'confidence': f"{confidence:.2f}",
        'actual_tb': actual_tb,
        'age': age,
        'gender': gender
    })

if __name__ == '__main__':
    app.run(debug=True)
