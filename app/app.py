from flask import Flask, render_template, request, jsonify
from app.prediction import load_all_models, predict_audio_file, predict_from_audio_file_with_clinical
import base64
import os
import uuid
import time
from pathlib import Path
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Load models once at startup
print("Loading models...")
load_all_models()
print("Models ready!")

# Configuration
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_FOLDER = BASE_DIR / "temp_audio"
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Input page - collects clinical data and records speech"""
    return render_template("index.html")

@app.route("/result")
def result():
    """Output page - displays prediction results"""
    return render_template("result.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict endpoint for live microphone input
    Receives: audio (base64), clinical (json), duration
    Returns: prediction, confidence, transcript, factors
    """
    temp_path = None
    try:
        data = request.get_json()
        audio_b64 = data["audio"]
        audio_bytes = base64.b64decode(audio_b64)
        
        # Get clinical data from frontend
        clinical_data = data.get("clinical", {})
        
        # Create unique temp file
        temp_filename = f"audio_{uuid.uuid4().hex}.wav"
        temp_path = TEMP_FOLDER / temp_filename
        
        # Save the audio file
        with temp_path.open("wb") as f:
            f.write(audio_bytes)
        
        print(f"✅ Saved: {temp_path}")
        print(f"   File size: {temp_path.stat().st_size} bytes")
        print(f"   Clinical data: {clinical_data}")
        
        # Predict using the new clinical model
        result = predict_from_audio_file_with_clinical(
            str(temp_path), 
            clinical_data=clinical_data
        )
        
        return jsonify({"success": True, **result})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
        
    finally:
        # Clean up
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                print(f"🗑️ Deleted: {temp_path}")
            except Exception as e:
                print(f"Could not delete: {e}")

@app.route('/analyze_upload', methods=['POST'])
def analyze_upload():
    audio_path = None
    try:
        # Get uploaded audio file
        audio_file = request.files.get('audio')
        if not audio_file or not allowed_file(audio_file.filename):
            return jsonify({"success": False, "error": "Invalid audio file"})
        
        # Get clinical data from form (it comes as JSON string)
        clinical_data_str = request.form.get('clinical', '{}')
        
        print(f"Raw clinical data string: {clinical_data_str}")  # DEBUG
        
        # Parse the JSON string
        try:
            clinical_data = json.loads(clinical_data_str) if clinical_data_str else {}
        except:
            clinical_data = {}
        
        print(f"Parsed clinical data: {clinical_data}")  # DEBUG
        
        # Validate required fields
        if not clinical_data.get('age_at_visit'):
            return jsonify({"success": False, "error": "Age is required. Please enter patient age."})
        
        if not clinical_data.get('mmse_score'):
            return jsonify({"success": False, "error": "MMSE score is required. Please enter MMSE (0-30)."})
        
        # Validate ranges
        age = int(clinical_data.get('age_at_visit'))
        mmse = int(clinical_data.get('mmse_score'))
        
        if age < 40 or age > 100:
            return jsonify({"success": False, "error": f"Age {age} is outside valid range (40-100)."})
        
        if mmse < 0 or mmse > 30:
            return jsonify({"success": False, "error": f"MMSE score {mmse} is outside valid range (0-30)."})
        
        # Save uploaded audio as WAV
        unique_id = uuid.uuid4().hex
        audio_filename = f"upload_{unique_id}.wav"
        audio_path = TEMP_FOLDER / audio_filename
        
        # Convert to WAV using pydub
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(audio_file)
        audio_segment.export(str(audio_path), format="wav")
        print(f"✅ Saved uploaded audio: {audio_path}")
        
        # Check for optional transcript upload
        transcript_file = request.files.get('transcript')
        transcript_text = None
        if transcript_file and transcript_file.filename:
            transcript_text = transcript_file.read().decode('utf-8')
            print(f"📝 Using provided transcript ({len(transcript_text)} chars)")
        
        # Predict using the uploaded audio
        result = predict_from_audio_file_with_clinical(
            audio_path, 
            clinical_data=clinical_data,
            provided_transcript=transcript_text
        )
        
        print(f"Prediction result: {result}")  # DEBUG
        
        return jsonify({"success": True, **result})
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})
        
    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except:
                pass
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting Flask server at http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
