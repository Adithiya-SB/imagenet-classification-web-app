import os
import time
from flask import Flask, request, render_template
import google.generativeai as genai
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__, static_folder='resources/user_images')
UPLOAD_FOLDER = 'resources/user_images/'

# ============== CONFIGURE GEMINI ==============
GEMINI_API_KEY = "AIzaSyC_0QAcM8btpd1YnfincPAE1_cQaJktvEw"   # ← Put your Gemini API key here
genai.configure(api_key=GEMINI_API_KEY)

# Triton Configuration
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "http://localhost:8000")
triton_client = None

def get_triton_client():
    global triton_client
    if triton_client is None:
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL.replace("http://", ""))
    return triton_client

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).numpy()

def predict_with_triton(image_path):
    try:
        client = get_triton_client()
        input_data = preprocess_image(image_path)

        inputs = []
        inputs.append(httpclient.InferInput("input", input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput("output"))

        results = client.infer(model_name="intel_image_classifier", inputs=inputs, outputs=outputs)
        output_data = results.as_numpy("output")[0]

        probs = np.exp(output_data) / np.sum(np.exp(output_data))   # softmax
        top_indices = probs.argsort()[-3:][::-1]
        
        # Load labels
        import json
        with open('resources/labels.json', 'r') as f:
            labels = json.load(f)
        
        names = [labels.get(str(i)) for i in top_indices]
        probs_top = probs[top_indices].tolist()

        return probs_top, names

    except Exception as e:
        print("Triton Error:", e)
        # Fallback to direct PyTorch if Triton fails
        from model_utils import predict_class
        return predict_class(image_path)

def get_llm_explanation(top_class, confidence):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an expert image analyst.
        The AI model classified this image as '{top_class}' with {confidence:.1f}% confidence.
        Write a short, natural, and informative explanation (2-3 sentences).
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return f"This image is classified as {top_class} with {confidence:.1f}% confidence."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_name = f'{int(time.time())}.jpg'
        image_file = request.files['image']
        image_loc = os.path.join(UPLOAD_FOLDER, image_name)
        image_file.save(image_loc)

        probs, names = predict_with_triton(image_loc)
        
        top_class = names[0]
        top_confidence = probs[0] * 100
        explanation = get_llm_explanation(top_class, top_confidence)

        return render_template('index.html', 
                             res=zip(probs, names), 
                             image_name=image_name,
                             top_class=top_class,
                             top_confidence=round(top_confidence, 1),
                             explanation=explanation)

    return render_template('index.html', res=None, image_name=None, 
                         top_class=None, top_confidence=None, explanation=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)