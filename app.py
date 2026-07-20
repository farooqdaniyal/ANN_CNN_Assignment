import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import shap
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import base64
import os
import gdown

# === PAGE CONFIG ===
st.set_page_config(
    page_title="MediCoPilot - AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS (Professional Theme) ===
def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

        :root {
            /* 🔵 Blue — Trust & Professional */
            --brand-900: #0d47a1;
            --brand-800: #1257b0;
            --brand-700: #1565c0;
            --brand-600: #1976d2;
            --brand-500: #2196f3;
            --brand-400: #64b5f6;

            /* 🟢 Green — Health & Success */
            --accent: #2e7d32;
            --success: #2e7d32;
            --success-bg: #e8f5e9;

            /* 🔴 Red — Emergency / Alerts */
            --danger: #c62828;
            --danger-bg: #fdecea;

            /* Warning (secondary, amber) */
            --warning: #ef6c00;
            --warning-bg: #fff3e0;

            /* ⚪ White — Clean & Medical */
            --info-bg: #e3f2fd;
            --surface: #ffffff;
            --surface-muted: #f4f8fc;
            --border: #d7e3f0;
            --text-main: #10233d;
            --text-muted: #5c7189;
            --radius: 14px;
            --shadow-sm: 0 1px 3px rgba(13, 71, 161, 0.07);
            --shadow-md: 0 8px 22px rgba(13, 71, 161, 0.10);
            --shadow-lg: 0 16px 38px rgba(13, 71, 161, 0.15);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-main);
        }

        .main {
            padding: 0.5rem 1.5rem 4rem 1.5rem;
            background: var(--surface-muted);
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(190deg, var(--brand-900) 0%, var(--brand-700) 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #eaf3fb;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.15);
            background: none;
            height: 1px;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            font-weight: 600;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 6px;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.14);
            border-color: rgba(255,255,255,0.25);
        }

        /* ===== HEADINGS ===== */
        h1, h2, h3 {
            font-family: 'Poppins', 'Inter', sans-serif;
            color: var(--brand-900);
            letter-spacing: -0.02em;
        }
        h1 { font-weight: 800; }
        h3 { font-weight: 700; }

        /* ===== GENERIC CARD ===== */
        .stCard, .info-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 22px;
            box-shadow: var(--shadow-sm);
            margin: 10px 0;
            transition: box-shadow 0.25s ease, transform 0.25s ease;
        }
        .stCard:hover, .info-box:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }
        .info-box {
            border-left: 4px solid var(--brand-500);
        }

        /* ===== METRIC / RESULT CARDS ===== */
        .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow-sm);
            border-left: 4px solid var(--brand-500);
            margin: 5px 0;
        }
        .metric-card.success { border-left-color: var(--success); }
        .metric-card.warning { border-left-color: var(--warning); }
        .metric-card.danger  { border-left-color: var(--danger); }

        .result-panel {
            border-radius: var(--radius);
            padding: 22px 20px;
            box-shadow: var(--shadow-sm);
            border: 1px solid transparent;
        }
        .result-panel.danger {
            background: var(--danger-bg);
            border-color: rgba(220,38,38,0.18);
        }
        .result-panel.success {
            background: var(--success-bg);
            border-color: rgba(22,163,74,0.18);
        }
        .result-panel h2 {
            margin: 0;
            font-size: 1.5rem;
        }
        .result-panel.danger h2 { color: var(--danger); }
        .result-panel.success h2 { color: var(--success); }

        /* ===== BUTTONS ===== */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.25s ease;
            border: 1px solid var(--border);
            padding: 0.55rem 1.5rem;
            letter-spacing: 0.01em;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--brand-500), var(--brand-800));
            color: white;
            border: none;
        }

        .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent), #1b5e20);
            color: white;
            border: none;
        }
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        /* ===== FILE UPLOAD ===== */
        .upload-container,
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed var(--brand-400) !important;
            border-radius: var(--radius) !important;
            background: var(--info-bg) !important;
        }

        /* ===== DIVIDER ===== */
        hr {
            margin: 1.75rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--brand-400), transparent);
        }

        /* ===== BADGES ===== */
        .badge {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }
        .badge-success { background: var(--success-bg); color: var(--success); }
        .badge-danger  { background: var(--danger-bg); color: var(--danger); }
        .badge-warning { background: var(--warning-bg); color: var(--warning); }
        .badge-info    { background: var(--info-bg); color: var(--brand-600); }

        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--brand-500), var(--accent));
        }

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: var(--brand-500);
            color: white;
        }

        /* ===== IMAGES ===== */
        .stImage img {
            border-radius: 14px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border);
        }

        /* ===== SLIDERS & INPUTS ===== */
        .stSlider [data-baseweb="slider"] > div > div {
            background: var(--brand-500) !important;
        }
        .stTextInput input, .stTextArea textarea {
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--brand-500) !important;
            box-shadow: 0 0 0 3px rgba(26,127,196,0.15) !important;
        }

        /* ===== ALERTS ===== */
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        /* ===== ANIMATION ===== */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in { animation: fadeIn 0.45s ease-out; }

        /* ===== HERO HEADER ===== */
        .hero-header {
            background: linear-gradient(135deg, var(--brand-900), var(--brand-600));
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 8px;
            box-shadow: var(--shadow-lg);
        }
        .hero-header h1 {
            color: white;
            margin: 0 0 6px 0;
            font-size: 2.1rem;
        }
        .hero-header p {
            color: rgba(255,255,255,0.85);
            margin: 0;
            font-size: 1.05rem;
        }

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #eef1f5; }
        ::-webkit-scrollbar-thumb {
            background: var(--brand-500);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--brand-800); }

        /* ===== SPINNER ===== */
        .stSpinner > div { border-color: var(--brand-500) !important; }

        /* ===== FOOTER ===== */
        .footer {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            background: var(--surface);
            padding: 10px;
            text-align: center;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-muted);
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero-header fade-in">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# === HELPER FUNCTION ===
def extract_num(text):
    """Extracts just the number inside the brackets from selectbox strings"""
    try:
        return int(text.split('(')[1].replace(')', ''))
    except:
        return int(text)

# === LOAD MODELS ===
@st.cache_resource
def load_models():
    # CNN Model download
    if not os.path.exists('cnn_model.keras'):
        with st.spinner("CNN model download ho raha hai... (222 MB)"):
            gdown.download(
                'https://drive.google.com/uc?id=1zsjTocDkWSAzai7ItYhJmkoQEbxqRz3x',
                'cnn_model.keras', quiet=False
            )
    # ANN Model download
    if not os.path.exists('ann_model.keras'):
        with st.spinner("ANN model download ho raha hai..."):
            gdown.download(
                'https://drive.google.com/uc?id=18-XoiEAnQF1NY32W0_fyC_J2Z0RTnXQi',
                'ann_model.keras', quiet=False
            )
    # Scaler download
    if not os.path.exists('scaler.pkl'):
        with st.spinner("Scaler download ho raha hai..."):
            gdown.download(
                'https://drive.google.com/uc?id=1rt6dGcYtb5mdGLasjeoVQ5wf_VGqsgPS',
                'scaler.pkl', quiet=False
            )
    cnn = tf.keras.models.load_model('cnn_model.keras')
    ann = tf.keras.models.load_model('ann_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return cnn, ann, scaler

try:
    cnn_model, ann_model, scaler = load_models()
except:
    st.error("⚠️ Models not found. Please ensure all model files are in the correct directory.")
    st.stop()

# === GRAD-CAM ===
def make_gradcam(img_array, model, threshold=0.3):
    img_tensor = tf.cast(np.expand_dims(img_array, 0), tf.float32)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer('conv4').output, model.output]
    )
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor)
        score = preds[:, 0]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
    conv_np = conv_out[0].numpy()

    heatmap = np.zeros(conv_np.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled):
        heatmap += w * conv_np[:,:,i]

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap*255)).resize((224,224))
    ) / 255.0

    colormap = plt.colormaps['jet']
    heatmap_color = colormap(heatmap_resized)[:,:,:3]
    overlay = np.clip(0.5*img_array + 0.5*heatmap_color, 0, 1)

    pred_prob = float(preds[0][0].numpy())
    label = "PNEUMONIA" if pred_prob < threshold else "NORMAL"
    confidence = (1-pred_prob)*100 if pred_prob < threshold else pred_prob*100

    return overlay, label, confidence

# === SESSION STATE ===
if 'xray_result' not in st.session_state:
    st.session_state.xray_result = None
if 'heart_result' not in st.session_state:
    st.session_state.heart_result = None
if 'approved' not in st.session_state:
    st.session_state.approved = False
if 'doctor_notes' not in st.session_state:
    st.session_state.doctor_notes = ""
if 'decision_status' not in st.session_state:
    st.session_state.decision_status = None
if 'overlay_img' not in st.session_state:
    st.session_state.overlay_img = None

# === SIDEBAR ===
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 24px 0 16px 0;">
        <div style="font-size: 2.8rem; line-height:1;">🏥</div>
        <h2 style="color: white; margin: 8px 0 2px 0; font-family:'Poppins',sans-serif;">MediCoPilot</h2>
        <p style="color: #a9c9e3; margin: 0; font-size: 0.85rem;">AI-Powered Healthcare Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        [
            "🫁 X-Ray Analysis",
            "❤️ Heart Disease",
            "👨‍⚕️ Doctor Review",
            "📄 PDF Report"
        ],
        index=0
    )
    
    st.divider()
    
    st.markdown("### 📊 Progress Tracker")
    
    xray_status = "✅" if st.session_state.xray_result else "⏳"
    heart_status = "✅" if st.session_state.heart_result else "⏳"
    review_status = "✅" if st.session_state.approved else "⏳"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**🫁**\n{xray_status}")
    with col2:
        st.markdown(f"**❤️**\n{heart_status}")
    with col3:
        st.markdown(f"**👨‍⚕️**\n{review_status}")
    
    st.divider()
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.08); border-radius: 12px; padding: 15px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.12);">
        <p style="color: #a9c9e3; font-size: 0.8rem; margin: 0; line-height:1.6;">
            🔒 HIPAA Compliant<br>
            🏥 Clinical Decision Support<br>
            📊 AI-Powered Diagnostics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #7fa8c9; font-size: 0.7rem; margin-top: 20px;">
        © 2026 MediCoPilot v2.0<br>
        <span style="opacity: 0.7;">AI for Better Healthcare</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 1 — X-RAY ANALYSIS
# ==========================================
if page == "🫁 X-Ray Analysis":
    hero("🫁 Chest X-Ray Analysis", "Upload a chest X-ray image — AI will detect pneumonia with high accuracy")
    
    st.divider()
    
    uploaded = st.file_uploader(
        "📤 Upload X-Ray Image (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear chest X-ray image for pneumonia detection"
    )
    
    if uploaded:
        img = Image.open(uploaded).convert('RGB').resize((224,224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">📷 Original X-Ray</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(img, use_column_width=True)
        
        with st.spinner("🔍 AI is analyzing the image..."):
            overlay, label, confidence = make_gradcam(img_array, cnn_model)
            st.session_state.overlay_img = overlay
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">🌡️ Grad-CAM Heatmap</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(overlay, use_column_width=True, 
                    caption="🔴 Red areas indicate potential infection")
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">🤖 AI Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if label == "PNEUMONIA":
                st.markdown("""
                <div class="result-panel danger">
                    <h2>🔴 PNEUMONIA</h2>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.1f}%", delta="High Risk")
                st.warning("⚠️ Please consult a doctor immediately!")
            else:
                st.markdown("""
                <div class="result-panel success">
                    <h2>🟢 NORMAL</h2>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.1f}%", delta="Low Risk")
                st.info("✅ The lungs appear to be clear")
        
        st.session_state.xray_result = {
            'label': label,
            'confidence': confidence
        }
        
        st.success("✅ Analysis Complete! Please go to the 'Doctor Review' section for final confirmation.")
        
        # Display additional info
        with st.expander("📊 How to interpret these results"):
            st.markdown("""
            - **PNEUMONIA**: AI detected patterns consistent with pneumonia. Please consult a doctor.
            - **NORMAL**: No pneumonia detected. The lungs appear healthy.
            - **Confidence Score**: Higher percentage means more confident prediction.
            - **Grad-CAM Heatmap**: Shows which areas of the X-ray influenced the AI's decision.
            """)

# ==========================================
# PAGE 2 — HEART DISEASE
# ==========================================
elif page == "❤️ Heart Disease":
    hero("❤️ Heart Disease Risk Assessment", "Comprehensive AI-powered cardiovascular risk analysis")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stCard">
            <h3 style="margin-top:0;">👤 Patient Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        age = st.slider("🎂 Age", 20, 80, 50, help="Patient's age in years")
        sex = st.selectbox("👤 Gender", ["Female (0)", "Male (1)"])
        cp = st.selectbox("💔 Chest Pain Type", ["Typical Angina (1)", "Atypical Angina (2)", "Non-anginal Pain (3)", "Asymptomatic (4)"])
        trestbps = st.slider("🩺 Blood Pressure (mm Hg)", 90, 200, 120, help="Resting blood pressure")
        chol = st.slider("🧪 Cholesterol (mg/dl)", 100, 600, 200, help="Serum cholesterol")
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", ["No (0)", "Yes (1)"])
        restecg = st.selectbox("📈 ECG Result", ["Normal (0)", "Abnormal (1)", "LVH (2)"])
    
    with col2:
        st.markdown("""
        <div class="stCard">
            <h3 style="margin-top:0;">📊 Medical Data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        thalach = st.slider("💓 Max Heart Rate", 70, 210, 150, help="Maximum heart rate achieved")
        exang = st.selectbox("🏃 Exercise Angina", ["No (0)", "Yes (1)"])
        oldpeak = st.slider("📉 ST Depression", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")
        slope = st.selectbox("📊 ST Slope", ["Upsloping (1)", "Flat (2)", "Downsloping (3)"])
        ca = st.selectbox("🩻 Blocked Vessels", ["0", "1", "2", "3"], help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox("🧬 Thalassemia", ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"])
    
    st.divider()
    
    if st.button("🔍 Analyze Heart Disease Risk", type="primary", use_container_width=True):
        features = np.array([[
            age,
            extract_num(sex),
            extract_num(cp),
            trestbps,
            chol,
            extract_num(fbs),
            extract_num(restecg),
            thalach,
            extract_num(exang),
            oldpeak,
            extract_num(slope),
            int(ca),
            extract_num(thal)
        ]])
        
        features_scaled = scaler.transform(features)
        pred_prob = float(ann_model.predict(features_scaled, verbose=0)[0][0])
        risk = "HIGH RISK" if pred_prob > 0.5 else "LOW RISK"
        confidence = pred_prob*100 if pred_prob > 0.5 else (1-pred_prob)*100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stCard">
                <h3 style="margin-top:0;">🤖 AI Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if risk == "HIGH RISK":
                st.markdown("""
                <div class="result-panel danger">
                    <h2>🔴 HIGH RISK</h2>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Disease Probability", f"{pred_prob*100:.1f}%", delta="High Risk")
                st.warning("⚠️ Please consult a cardiologist immediately!")
            else:
                st.markdown("""
                <div class="result-panel success">
                    <h2>🟢 LOW RISK</h2>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Disease Probability", f"{pred_prob*100:.1f}%", delta="Low Risk")
                st.info("✅ Heart appears healthy")
        
        with col2:
            st.markdown("""
            <div class="stCard">
                <h3 style="margin-top:0;">📊 SHAP Feature Impact</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Calculating SHAP values..."):
                def model_pred(x):
                    return ann_model.predict(x, verbose=0).flatten()
                
                bg = scaler.transform(np.zeros((1,13)))
                explainer = shap.KernelExplainer(model_pred, bg)
                shap_vals = explainer.shap_values(features_scaled, nsamples=50)
                
                feature_names = ['Age','Sex','Chest Pain','Blood Pressure','Cholesterol',
                               'Fasting Blood Sugar','ECG','Max Heart Rate','Exercise Angina',
                               'ST Depression','ST Slope','Blocked Vessels','Thalassemia']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#dc2626' if v > 0 else '#16a34a' for v in shap_vals[0]]
                ax.barh(feature_names, shap_vals[0], color=colors, alpha=0.85)
                ax.set_title('Feature Impact on Prediction', fontsize=14, fontweight='bold', color='#0B2545')
                ax.set_xlabel('SHAP Value', fontsize=12)
                ax.axvline(x=0, color='#334155', linewidth=0.8, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.patch.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig)
        
        st.session_state.heart_result = {
            'risk': risk,
            'probability': pred_prob*100,
            'age': age,
            'bp': trestbps,
            'chol': chol,
            'heart_rate': thalach
        }
        
        st.success("✅ Assessment complete! Proceed to Doctor Review for final verification.")

# ==========================================
# PAGE 3 — DOCTOR REVIEW
# ==========================================
elif page == "👨‍⚕️ Doctor Review":
    hero("👨‍⚕️ Doctor Review Panel", "Human-in-the-loop — final clinical verification")
    
    st.divider()
    
    if st.session_state.xray_result or st.session_state.heart_result:
        st.markdown("""
        <div style="margin: 10px 0 4px 0;">
            <h3>📋 AI Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.xray_result:
                r = st.session_state.xray_result
                st.markdown(f"""
                <div class="stCard">
                    <h4 style="margin-top:0;">🫁 X-Ray Analysis</h4>
                    <p><strong>Diagnosis:</strong> {r['label']}</p>
                    <p style="margin-bottom:0;"><strong>Confidence:</strong> {r['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.heart_result:
                h = st.session_state.heart_result
                st.markdown(f"""
                <div class="stCard">
                    <h4 style="margin-top:0;">❤️ Heart Disease</h4>
                    <p><strong>Risk:</strong> {h['risk']}</p>
                    <p style="margin-bottom:0;"><strong>Probability:</strong> {h['probability']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please complete X-Ray or Heart analysis first before proceeding.")
        st.stop()
    
    # Decision Status
    if st.session_state.decision_status:
        status_colors = {
            "Approved": "success",
            "Modified": "warning",
            "Rejected": "danger"
        }
        status_color = status_colors.get(st.session_state.decision_status, "info")
        st.markdown(f"""
        <div class="info-box">
            <h4 style="margin-top:0;">📋 Current Status</h4>
            <span class="badge badge-{status_color}">{st.session_state.decision_status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div class="stCard">
        <h3 style="margin-top:0;">👨‍⚕️ Final Clinical Decision</h3>
    </div>
    """, unsafe_allow_html=True)
    
    doctor_diagnosis = st.text_input(
        "Final Diagnosis:",
        placeholder="e.g. Bacterial Pneumonia confirmed",
        help="Enter the final diagnosis based on AI results and clinical judgment"
    )
    
    notes = st.text_area(
        "Doctor Notes:",
        placeholder="Additional observations, treatment recommendations, or follow-up notes...",
        height=100,
        help="Include any additional clinical observations"
    )
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ APPROVE", type="primary", use_container_width=True):
            st.session_state.approved = True
            st.session_state.decision_status = "Approved"
            st.session_state.doctor_notes = f"Diagnosis: {doctor_diagnosis}\nNotes: {notes}"
            st.success("✅ Approved! Proceed to PDF Report.")
            st.balloons()
            st.rerun()
    
    with col2:
        if st.button("✏️ MODIFY", use_container_width=True):
            st.session_state.approved = True
            st.session_state.decision_status = "Modified"
            st.session_state.doctor_notes = f"MODIFIED — {doctor_diagnosis}\n{notes}"
            st.warning("✏️ Modified and saved!")
            st.rerun()
    
    with col3:
        if st.button("❌ REJECT", use_container_width=True):
            st.session_state.approved = False
            st.session_state.decision_status = "Rejected"
            st.session_state.doctor_notes = f"Rejected — {doctor_diagnosis}\n{notes}"
            st.error("❌ Rejected! Reanalysis required.")
            st.rerun()

# ==========================================
# PAGE 4 — PDF REPORT
# ==========================================
elif page == "📄 PDF Report":
    hero("📄 Medical Report Generator", "Generate comprehensive medical reports with AI findings")
    
    st.divider()
    
    if st.session_state.decision_status == "Rejected":
        st.error("❌ The case has been REJECTED. PDF cannot be generated.")
        st.info("Please go to 👨‍⚕️ Doctor Review for reanalysis.")
    elif not st.session_state.approved:
        st.warning("⚠️ Please complete Doctor Review first!")
        st.info("Go to 👨‍⚕️ Doctor Review in the sidebar")
    else:
        status_label = st.session_state.decision_status or "Approved"
        st.success(f"✅ Doctor Status: **{status_label}** — Report ready for download")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="stCard">
                <h3 style="margin-top:0;">📋 Report Preview</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                <p><strong>👨‍⚕️ Decision Status:</strong> <span class="badge badge-success">{status_label}</span></p>
            """, unsafe_allow_html=True)
            
            if st.session_state.xray_result:
                r = st.session_state.xray_result
                st.markdown(f"""
                <p><strong>🫁 X-Ray:</strong> {r['label']} <span style="color: var(--text-muted);">(Confidence: {r['confidence']:.1f}%)</span></p>
                """, unsafe_allow_html=True)
            
            if st.session_state.heart_result:
                h = st.session_state.heart_result
                st.markdown(f"""
                <p><strong>❤️ Heart Disease:</strong> {h['risk']} <span style="color: var(--text-muted);">(Probability: {h['probability']:.1f}%)</span></p>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p><strong>📝 Doctor Notes:</strong></p>
                <div style="background: white; border-radius: 10px; padding: 15px; border: 1px solid var(--border);">
                    {st.session_state.doctor_notes.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stCard">
                <h3 style="margin-top:0;">📥 Download Options</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                story.append(Paragraph("MediCoPilot — AI Medical Report", styles['Title']))
                story.append(Spacer(1, 20))
                
                # Decision Status
                story.append(Paragraph(f"Doctor Decision Status: {status_label}", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                # X-Ray Results
                if st.session_state.xray_result:
                    r = st.session_state.xray_result
                    story.append(Paragraph("Chest X-Ray Analysis", styles['Heading2']))
                    story.append(Paragraph(f"AI Diagnosis: {r['label']}", styles['Normal']))
                    story.append(Paragraph(f"Confidence: {r['confidence']:.1f}%", styles['Normal']))
                    story.append(Spacer(1, 15))
                
                # Heart Results
                if st.session_state.heart_result:
                    h = st.session_state.heart_result
                    story.append(Paragraph("Heart Disease Assessment", styles['Heading2']))
                    story.append(Paragraph(f"Risk Level: {h['risk']}", styles['Normal']))
                    story.append(Paragraph(f"Disease Probability: {h['probability']:.1f}%", styles['Normal']))
                    story.append(Paragraph(f"Age: {h['age']} | BP: {h['bp']} | Cholesterol: {h['chol']}", styles['Normal']))
                    story.append(Spacer(1, 15))
                
                # Doctor Notes
                story.append(Paragraph("Doctor Review", styles['Heading2']))
                story.append(Paragraph(st.session_state.doctor_notes or "No notes", styles['Normal']))
                story.append(Spacer(1, 15))
                
                # Footer
                story.append(Paragraph("Generated by MediCoPilot AI System", styles['Italic']))
                story.append(Paragraph("Disclaimer: This report is for clinical decision support only.", styles['Italic']))
                
                doc.build(story)
                buffer.seek(0)
                
                st.download_button(
                    label="💾 Download PDF Report",
                    data=buffer,
                    file_name="medical_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ PDF ready for download!")
            
            # Quick actions
            st.markdown("""
            <div class="info-box" style="margin-top: 15px;">
                <h4 style="margin-top:0;">⚡ Quick Actions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Start New Assessment", use_container_width=True):
                for key in ['xray_result', 'heart_result', 'approved', 'decision_status', 'doctor_notes']:
                    if key in st.session_state:
                        st.session_state[key] = None
                st.success("✅ Reset complete! Start a new assessment.")
                st.rerun()
