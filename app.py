from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==================== U-Net Model Definition ====================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = CBR(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ==================== Flask Setup ====================
app = Flask(__name__)
CORS(app)

MODEL_DIR = 'models'
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize models to None
clinical_model = None
scaler = None
pca = None
unet_classifier = None
unet = None

# --- Enhanced Error Logging for Model Loading ---
model_load_errors = [] 

try:
    # Model filenames from the user's provided app.py
    clinical_model = joblib.load(os.path.join(MODEL_DIR, 'final_voting_clinical_model1.pkl'))
    print("Loaded final_voting_clinical_model1.pkl")
except Exception as e:
    model_load_errors.append(f"Error loading final_voting_clinical_model1.pkl: {e}")
    print(f"Error loading final_voting_clinical_model1.pkl: {e}")
    traceback.print_exc()

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler1.pkl'))
    print("Loaded scaler1.pkl")
except Exception as e:
    model_load_errors.append(f"Error loading scaler1.pkl: {e}")
    print(f"Error loading scaler1.pkl: {e}")
    traceback.print_exc()

try:
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca1.pkl'))
    print("Loaded pca1.pkl")
except Exception as e:
    model_load_errors.append(f"Error loading pca1.pkl: {e}")
    print(f"Error loading pca1.pkl: {e}")
    traceback.print_exc()

try:
    # Load U-Net (NO CHANGES to this block as per user's request, but added weights_only=True for security)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet()
    unet.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'unet_pcos.pth'), map_location=device, weights_only=True))
    unet.to(device)
    unet.eval()
    print("Loaded unet_pcos.pth")

    # Load image classifier (NO CHANGES to this block as per user's request)
    unet_classifier = joblib.load(os.path.join(MODEL_DIR, 'unet_classifier.pkl'))
    print("Loaded unet_classifier.pkl")
except Exception as e:
    model_load_errors.append(f"Error loading image models (unet_pcos.pth or unet_classifier.pkl): {e}")
    print(f"Error loading image models (unet_pcos.pth or unet_classifier.pkl): {e}")
    traceback.print_exc()


# ==================== Routes ====================

@app.route('/')
def welcome():
    """Renders the welcome page with a 'Start' button."""
    return render_template('welcome.html')

@app.route('/home')
def home():
    """Renders the main input form page."""
    return render_template('index.html')

@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    """
    Handles clinical data prediction.
    Features and logic are updated to match the 10 features from 'Untitled.ipynb'.
    Input parsing uses .get() for robustness.
    Includes more specific error logging for ValueError.
    If PCOS is detected, it redirects to the recommendations page.
    """
    clinical_result = None
    # Check if clinical_model or scaler are None due to loading errors
    if clinical_model is None or scaler is None: # Removed pca from this check as it's being bypassed
        clinical_result = 'Error: Clinical prediction models not loaded or initialized.'
        if model_load_errors:
            clinical_result += " Details: " + "; ".join([err for err in model_load_errors if "clinical" in err.lower() or "scaler" in err.lower()])
        return render_template("index.html", clinical_result=clinical_result)

    print(f"Received form data: {request.form}") # Print raw form data for debugging

    try:
        # Extract all 10 clinical features as per Untitled.ipynb (cell 38's X definition)
        # Using .get() with a default empty string for robustness against missing fields.
        
        # Age
        try:
            age = float(request.form.get('age', ''))
        except ValueError:
            raise ValueError("Invalid input for 'Age'. Please enter a numerical value.")

        # BMI
        try:
            bmi = float(request.form.get('bmi', ''))
        except ValueError:
            raise ValueError("Invalid input for 'BMI'. Please enter a numerical value.")

        # Cycle Type (Regular/Irregular)
        cycle_r_i = 1 if request.form.get('cycle_type', '') == 'Irregular' else 0

        # Cycle Length (days)
        try:
            cycle_lengthdays = float(request.form.get('cycle_length', ''))
        except ValueError:
            raise ValueError("Invalid input for 'Cycle Length (days)'. Please enter a numerical value.")

        # Number of Abortions
        try:
            no_of_abortions = float(request.form.get('abortions', ''))
        except ValueError:
            raise ValueError("Invalid input for 'Number of Abortions'. Please enter a numerical value.")

        # AMH (ng/mL)
        try:
            amhngml = float(request.form.get('amh', ''))
        except ValueError:
            raise ValueError("Invalid input for 'AMH (ng/mL)'. Please enter a numerical value.")

        # FSH/LH Ratio
        try:
            fsh_lh = float(request.form.get('fsh_lh_ratio', ''))
        except ValueError:
            raise ValueError("Invalid input for 'FSH/LH Ratio'. Please enter a numerical value.")
        
        # Hair Growth (0=No, 1=Yes)
        hair_growth_str = request.form.get('hair_growth', '')
        try:
            if hair_growth_str == '':
                hair_growth = 0.0 # Default to 0.0 if empty string
            else:
                hair_growth = float(int(hair_growth_str)) # Convert to int then float
        except ValueError:
            raise ValueError("Invalid input for 'Hair Growth'. Please enter 0 or 1.")

        # Follicle No. (Left Ovary)
        try:
            follicle_no_l = float(request.form.get('follicle_l', ''))
        except ValueError:
            raise ValueError("Invalid input for 'Follicle No. (Left Ovary)'. Please enter a numerical value.")

        # Follicle No. (Right Ovary)
        try:
            follicle_no_r = float(request.form.get('follicle_r', ''))
        except ValueError:
            raise ValueError("Invalid input for 'Follicle No. (Right Ovary)'. Please enter a numerical value.")

        # Create a NumPy array with the features in the correct order
        # This order must match the order used during training in Untitled.ipynb (cell 38's X definition)
        # X = df[['age_yrs', 'BMI', 'cycle_r_i', 'cycle_lengthdays', 'no_of_abortions', 'amhngml', 'fsh_lh', 'hair_growth', 'follicle_no_l', 'follicle_no_r']]
        raw_features = np.array([[
            age, bmi, cycle_r_i, cycle_lengthdays, no_of_abortions,
            amhngml, fsh_lh, hair_growth, follicle_no_l, follicle_no_r
        ]])

        scaled = scaler.transform(raw_features)
        
        # FIX: Directly pass scaled features to the clinical model, bypassing PCA.
        # This assumes your clinical model was trained on the 10 scaled features.
        pred = clinical_model.predict(scaled)[0] 
        
        if pred == 1: # PCOS Detected
            # Clinical prediction redirects to general recommendations for PCOS
            return redirect(url_for("recommendations_page", diagnosis_type="Clinical", pcos_status="PCOS Detected"))
        else:
            clinical_result = 'No PCOS Detected'

    except ValueError as ve: # Catch the specific ValueError from our checks
        clinical_result = f'Error: {ve}'
        traceback.print_exc()
    except Exception as e:
        # Catch any other unexpected errors during prediction
        clinical_result = f'An unexpected error occurred during clinical prediction: {e}'
        traceback.print_exc()

    return render_template("index.html", clinical_result=clinical_result)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """
    Handles ultrasound image prediction.
    If PCOS is detected (early or advanced stage), it redirects to stages_recommendation.html.
    Otherwise, it displays a modal with "No PCOS Detected. Hence, no recommendation."
    """
    image_result = None
    if unet is None or unet_classifier is None:
        image_result = "Error: Image models not loaded. Cannot perform prediction."
        if model_load_errors:
            image_result += " Details: " + "; ".join([err for err in model_load_errors if "image" in err.lower() or "unet" in err.lower()])
        return render_template("index.html", image_result=image_result)

    try:
        file = request.files['image']
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert('L')
        transform = transforms.Compose([
            transforms.Resize((256, 256)), # User's original resize
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = torch.sigmoid(unet(img_tensor)).squeeze().cpu().numpy() # User's original sigmoid application

        mask = (output > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cyst_count = len(contours)
        total_area = sum(cv2.contourArea(c) for c in contours)
        avg_area = total_area / cyst_count if cyst_count > 0 else 0 

        circularities = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                circularities.append(circularity)
        avg_circularity = np.mean(circularities) if circularities else 0

        features = np.array([[cyst_count, total_area, avg_area, avg_circularity]])
        stage_raw = unet_classifier.predict(features)[0] # Get the raw output from the classifier

        print(f"Raw image prediction stage: {stage_raw}") # Debug print
        print(f"Type of raw image prediction stage: {type(stage_raw)}") # Debug print

        # Determine stage for recommendations based on the classifier output (now handling string outputs)
        pcos_stage_status = "N/A"
        if isinstance(stage_raw, (int, np.integer)): # If the output is numerical (0 or 1)
            if stage_raw == 0:
                pcos_stage_status = "Early Stage PCOS"
            elif stage_raw == 1:
                pcos_stage_status = "Advanced Stage PCOS"
            else:
                # This case means PCOS is not detected in a classified stage (e.g., stage is not 0 or 1)
                image_result = f"Image Stage: {stage_raw}" 
        elif isinstance(stage_raw, str): # If the output is a string (e.g., "early_stage", "advanced_stage")
            if stage_raw.lower() == "early_stage":
                pcos_stage_status = "Early Stage PCOS"
            elif stage_raw.lower() == "advanced_stage":
                pcos_stage_status = "Advanced Stage PCOS"
            else:
                # This case means PCOS is not detected in a classified stage
                image_result = f"Image Stage: {stage_raw}" 
        else:
            # Fallback for unexpected output type, assuming no PCOS detected for recommendations
            image_result = f"Image Stage: {stage_raw}" 

        print(f"Processed PCOS stage status for redirect: {pcos_stage_status}") # Debug print

        # Redirect if a recognized PCOS stage is detected
        if pcos_stage_status in ["Early Stage PCOS", "Advanced Stage PCOS"]:
            return redirect(url_for("stages_recommendations_page", diagnosis_type="Image", pcos_status=pcos_stage_status))
        else:
            # If not redirected, it means no specific PCOS stage was detected for recommendations.
            # We will now trigger the modal on index.html.
            return render_template("index.html", image_result=image_result, show_no_pcos_modal=True)
        
    except Exception as e:
        traceback.print_exc()
        image_result = f"Error in image prediction: {str(e)}"
        # If an error occurs, also show the modal
        return render_template("index.html", image_result=image_result, show_no_pcos_modal=True)

    return render_template("index.html", image_result=image_result) # Fallback if somehow not redirected or modal triggered

@app.route('/recommendations')
def recommendations_page():
    """Renders the general recommendations page (for clinical predictions)."""
    diagnosis_type = request.args.get('diagnosis_type', 'N/A')
    pcos_status = request.args.get('pcos_status', 'N/A')
    return render_template('recommendations.html', diagnosis_type=diagnosis_type, pcos_status=pcos_status)

@app.route('/stages_recommendations')
def stages_recommendations_page():
    """Renders the stage-specific recommendations page (for image predictions)."""
    diagnosis_type = request.args.get('diagnosis_type', 'N/A')
    pcos_status = request.args.get('pcos_status', 'N/A')
    return render_template('stages_recommendations.html', diagnosis_type=diagnosis_type, pcos_status=pcos_status)


# ==================== Main ====================
if __name__ == '__main__':
    app.run(debug=True)

