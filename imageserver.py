from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import os
import easyocr
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
from PIL import Image
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load models and tokenizers
models = {}
tokenizers = {}

# Initialize EasyOCR reader
try:
    ocr_reader = easyocr.Reader(['en', 'es', 'fr'])
except Exception as e:
    logger.error(f"Failed to initialize OCR reader: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({'status': 'Server is running'}), 200

# Load models from the `models/` directory
try:
    model_pairs = ["es-en-model", "en-es-model", "fr-en-model", "en-fr-model", "es-fr-model", "fr-es-model"]
    for pair in model_pairs:
        model_path = os.path.join("models", pair)
        if os.path.exists(model_path):
            models[pair] = MarianMTModel.from_pretrained(model_path)
            tokenizers[pair] = MarianTokenizer.from_pretrained(model_path)
        else:
            logger.warning(f"Model path not found: {model_path}")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        logger.debug(f"Received translation request: {data}")

        # Validate input
        source_lang = data.get("source_lang")
        target_lang = data.get("target_lang")
        text = data.get("text")

        if not all([source_lang, target_lang, text]):
            return jsonify({"error": "source_lang, target_lang, and text are required"}), 400

        # Generate model key
        model_key = f"{source_lang}-{target_lang}-model"
        if model_key not in models:
            return jsonify({"error": f"Unsupported language pair: {model_key}"}), 400

        # Get the model and tokenizer
        model = models[model_key]
        tokenizer = tokenizers[model_key]

        # Tokenize input and perform translation
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(inputs["input_ids"])
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        return jsonify({"translated_text": translated_text})

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/ocr_translate", methods=["POST"])
def ocr_translate():
    try:
        logger.debug("Received OCR translate request")
        logger.debug(f"Files in request: {request.files}")
        logger.debug(f"Form data: {request.form}")

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Create a bytes IO object from the file
        img_bytes = io.BytesIO(file.read())
        
        # Open and verify the image
        try:
            img = Image.open(img_bytes)
            img.verify()  # Verify it's actually an image
            img_bytes.seek(0)  # Reset buffer position
            img = Image.open(img_bytes)  # Reopen after verify
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return jsonify({"error": "Invalid image file"}), 400

        # Save the image file temporarily
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(image_path)
        
        logger.debug(f"Image saved to: {image_path}")

        # Use EasyOCR to extract text from the image
        extracted_text = extract_text_from_image(image_path)
        logger.debug(f"Extracted text: {extracted_text}")

        if not extracted_text:
            return jsonify({"error": "No text found in the image"}), 400

        # Get translation parameters
        source_lang = request.form.get('source_lang', 'en')
        target_lang = request.form.get('target_lang', 'es')

        # Perform translation
        translated_text = perform_translation(source_lang, target_lang, extracted_text)
        logger.debug(f"Translated text: {translated_text}")

        # Clean up - remove temporary file
        try:
            os.remove(image_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")

        return jsonify({
            "extracted_text": extracted_text,
            "translated_text": translated_text
        })

    except Exception as e:
        logger.error(f"OCR translation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_text_from_image(image_path):
    try:
        results = ocr_reader.readtext(image_path)
        text = " ".join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise

def perform_translation(source_lang, target_lang, text):
    try:
        model_key = f"{source_lang}-{target_lang}-model"
        if model_key not in models:
            raise ValueError(f"Unsupported language pair: {model_key}")

        model = models[model_key]
        tokenizer = tokenizers[model_key]

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(inputs["input_ids"])
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)