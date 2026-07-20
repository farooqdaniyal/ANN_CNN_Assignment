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

# === PAGE CONFIG ===
st.set_page_config(
    page_title="MediCoPilot",
    page_icon="🏥",
    layout="wide"
)

# === HELPER FUNCTION (BUG FIX) ===
def extract_num(text):
    """
    Extracts just the number inside the brackets from selectbox
    strings like 'Male (1)'.
    Example: 'Male (1)' -> 1 , 'No (0)' -> 0
    """
    return int(text.split('(')[1].replace(')', ''))

# === LOAD MODELS ===
@st.cache_resource
def load_models():
    cnn    = tf.keras.models.load_model('cnn_model.keras')
    ann    = tf.keras.models.load_model('ann_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return cnn, ann, scaler

cnn_model, ann_model, scaler = load_models()

# === GRAD-CAM ===
def make_gradcam(img_array, model, threshold=0.3):
    img_tensor = tf.cast(np.expand_dims(img_array, 0), tf.float32)
    grad_model = tf.keras.Model(
        inputs  = model.input,
        outputs = [model.get_layer('conv4').output, model.output]
    )
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor)
        score = preds[:, 0]

    grads   = tape.gradient(score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
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

    colormap      = plt.colormaps['jet']
    heatmap_color = colormap(heatmap_resized)[:,:,:3]
    overlay       = np.clip(0.5*img_array + 0.5*heatmap_color, 0, 1)

    pred_prob  = float(preds[0][0].numpy())
    label      = "PNEUMONIA" if pred_prob < threshold else "NORMAL"
    confidence = (1-pred_prob)*100 if pred_prob < threshold else pred_prob*100

    return overlay, label, confidence

# === SESSION STATE ===
if 'xray_result'  not in st.session_state:
    st.session_state.xray_result  = None
if 'heart_result' not in st.session_state:
    st.session_state.heart_result = None
if 'approved'     not in st.session_state:
    st.session_state.approved     = False
if 'doctor_notes' not in st.session_state:
    st.session_state.doctor_notes = ""
if 'decision_status' not in st.session_state:
    st.session_state.decision_status = None   # "Approved" / "Modified" / "Rejected"
if 'overlay_img'  not in st.session_state:
    st.session_state.overlay_img  = None

# === SIDEBAR ===
st.sidebar.title("🏥 MediCoPilot")
st.sidebar.markdown("AI-Powered Healthcare Assistant")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "🫁 X-Ray Analysis",
    "❤️ Heart Disease",
    "👨‍⚕️ Doctor Review",
    "📄 PDF Report"
])

st.sidebar.divider()
st.sidebar.markdown("### Progress")
st.sidebar.markdown(f"X-Ray : {'✅' if st.session_state.xray_result  else '⏳'}")
st.sidebar.markdown(f"Heart  : {'✅' if st.session_state.heart_result else '⏳'}")
st.sidebar.markdown(f"Review : {'✅' if st.session_state.approved     else '⏳'}")

# ==========================================
# PAGE 1 — X-RAY ANALYSIS
# ==========================================
if page == "🫁 X-Ray Analysis":
    st.title("🫁 Chest X-Ray Analysis")
    st.markdown("Upload a Chest X-ray Image — AI Will Detect Pneumonia")
    st.divider()

    uploaded = st.file_uploader(
        "Upload X-Ray Image (JPG/PNG)",
        type=['jpg','jpeg','png']
    )

    if uploaded:
        img       = Image.open(uploaded).convert('RGB').resize((224,224))
        img_array = np.array(img, dtype=np.float32) / 255.0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📷 Original X-Ray")
            st.image(img, use_column_width=True)

        with st.spinner("🔍 AI is analyzing..."):
            overlay, label, confidence = make_gradcam(img_array, cnn_model)
            st.session_state.overlay_img = overlay

        with col2:
            st.subheader("🌡️ Grad-CAM Heatmap")
            st.image(overlay, use_column_width=True,
                     caption="Red = Infection area")

        with col3:
            st.subheader("🤖 AI Prediction")
            if label == "PNEUMONIA":
                st.error(f"🔴 {label}")
                st.metric("Confidence", f"{confidence:.1f}%")
                st.warning("⚠️ Please consult a doctor!")
            else:
                st.success(f"🟢 {label}")
                st.metric("Confidence", f"{confidence:.1f}%")
                st.info("✅ The lungs appear to be clear")

        st.session_state.xray_result = {
            'label'     : label,
            'confidence': confidence
        }

        st.success("✅ Analysis Complete! Please go to the 'Doctor Review' section in the sidebar.")

# ==========================================
# PAGE 2 — HEART DISEASE
# ==========================================
elif page == "❤️ Heart Disease":
    st.title("❤️ Heart Disease Risk Assessment")
    st.markdown("Patient enter details — AI will predict heart disease risk")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Info")
        age      = st.slider("🎂 Age", 20, 80, 50)
        sex      = st.selectbox("👤 Gender", ["Female (0)", "Male (1)"])
        cp       = st.selectbox("💔 Chest Pain Type (1-4)", ["1","2","3","4"])
        trestbps = st.slider("🩺 Blood Pressure (mm Hg)", 90, 200, 120)
        chol     = st.slider("🧪 Cholesterol (mg/dl)", 100, 600, 200)
        fbs      = st.selectbox("🍬 Fasting Blood Sugar > 120", ["No (0)","Yes (1)"])
        restecg  = st.selectbox("📈 ECG Result", ["Normal (0)","Abnormal (1)","LVH (2)"])

    with col2:
        st.subheader("Medical Data")
        thalach = st.slider("💓 Max Heart Rate", 70, 210, 150)
        exang   = st.selectbox("🏃 Exercise Angina", ["No (0)","Yes (1)"])
        oldpeak = st.slider("📉 ST Depression", 0.0, 6.0, 1.0)
        slope   = st.selectbox("📊 ST Slope", ["Up (1)","Flat (2)","Down (3)"])
        ca      = st.selectbox("🩻 Blocked Vessels (0-3)", ["0","1","2","3"])
        thal    = st.selectbox("🧬 Thalassemia", ["Normal (3)","Fixed (6)","Reversible (7)"])

    st.divider()

    if st.button("🔍 Heart Disease Risk Predict", type="primary"):
        # === FIX APPLIED HERE: extract_num() is used ===
        # sex, fbs, restecg, exang, slope, thal -> these are all in
        # "Text (number)" format, so extract_num() needs to be applied.
        # cp and ca are already plain number strings, so converting
        # them with int() alone is enough.
        features = np.array([[
            age,
            extract_num(sex),
            int(cp),
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
        pred_prob       = float(ann_model.predict(
                            features_scaled, verbose=0)[0][0])
        risk            = "HIGH RISK" if pred_prob > 0.5 else "LOW RISK"
        confidence      = pred_prob*100 if pred_prob > 0.5 else (1-pred_prob)*100

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 AI Prediction")
            if risk == "HIGH RISK":
                st.error(f"🔴 {risk}")
                st.metric("Disease Probability", f"{pred_prob*100:.1f}%")
                st.warning("⚠️ See a cardiologist!")
            else:
                st.success(f"🟢 {risk}")
                st.metric("Disease Probability", f"{pred_prob*100:.1f}%")
                st.info("✅ Heart appears healthy")

        with col2:
            st.subheader("📊 SHAP Feature Impact")
            with st.spinner("Calculating SHAP..."):
                def model_pred(x):
                    return ann_model.predict(x, verbose=0).flatten()

                bg          = scaler.transform(np.zeros((1,13)))
                explainer   = shap.KernelExplainer(model_pred, bg)
                shap_vals   = explainer.shap_values(
                                features_scaled, nsamples=50)

                feature_names = ['age','sex','cp','trestbps','chol',
                                 'fbs','restecg','thalach','exang',
                                 'oldpeak','slope','ca','thal']

                fig, ax = plt.subplots(figsize=(7, 5))
                colors  = ['#e74c3c' if v > 0 else '#2ecc71'
                           for v in shap_vals[0]]
                ax.barh(feature_names, shap_vals[0], color=colors)
                ax.set_title('Feature Impact on Prediction')
                ax.set_xlabel('SHAP Value')
                ax.axvline(x=0, color='black', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)

        st.session_state.heart_result = {
            'risk'       : risk,
            'probability': pred_prob*100,
            'age'        : age,
            'bp'         : trestbps,
            'chol'       : chol,
            'heart_rate' : thalach
        }
        st.success("✅ Assessment complete! Go to Doctor Review.")

# ==========================================
# PAGE 3 — DOCTOR REVIEW
# ==========================================
elif page == "👨‍⚕️ Doctor Review":
    st.title("👨‍⚕️ Doctor Review Panel")
    st.markdown("Human-in-the-Loop — Doctor verifies AI results")
    st.divider()

    # Show results
    if st.session_state.xray_result:
        r = st.session_state.xray_result
        st.info(f"🫁 X-Ray AI Result: **{r['label']}** "
                f"(Confidence: {r['confidence']:.1f}%)")

    if st.session_state.heart_result:
        h = st.session_state.heart_result
        st.info(f"❤️ Heart AI Result: **{h['risk']}** "
                f"(Probability: {h['probability']:.1f}%)")

    if not st.session_state.xray_result and not st.session_state.heart_result:
        st.warning("⚠️ Please do the X-Ray or Heart analysis first!")

    else:
        # === Badge for current decision status (if a decision has already been made) ===
        if st.session_state.decision_status == "Approved":
            st.success("✅ Current Status: **APPROVED**")
        elif st.session_state.decision_status == "Modified":
            st.warning("✏️ Current Status: **MODIFIED**")
        elif st.session_state.decision_status == "Rejected":
            st.error("❌ Current Status: **REJECTED**")

        st.divider()
        st.subheader("👨‍⚕️ Doctor Decision")

        doctor_diagnosis = st.text_input(
            "Final Diagnosis:",
            placeholder="e.g. Bacterial Pneumonia confirmed"
        )
        notes = st.text_area(
            "Doctor Notes:",
            placeholder="Additional observations..."
        )

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ APPROVE",
                         type="primary",
                         use_container_width=True):
                st.session_state.approved        = True
                st.session_state.decision_status = "Approved"
                st.session_state.doctor_notes = (
                    f"Diagnosis: {doctor_diagnosis}\n"
                    f"Notes: {notes}"
                )
                st.success("✅ Approved! Now generate the PDF.")
                st.balloons()
                st.rerun()

        with col2:
            if st.button("❌ REJECT",
                         use_container_width=True):
                st.session_state.approved        = False
                st.session_state.decision_status = "Rejected"
                st.session_state.doctor_notes = (
                    f"Diagnosis: {doctor_diagnosis}\n"
                    f"Notes: {notes}"
                )
                st.error("❌ Rejected! Reanalysis required.")
                st.rerun()

        with col3:
            if st.button("✏️ MODIFY",
                         use_container_width=True):
                st.session_state.approved        = True
                st.session_state.decision_status = "Modified"
                st.session_state.doctor_notes = (
                    f"MODIFIED — {doctor_diagnosis}\n{notes}"
                )
                st.warning("✏️ Modified and saved!")
                st.rerun()

# ==========================================
# PAGE 4 — PDF REPORT
# ==========================================
elif page == "📄 PDF Report":
    st.title("📄 Medical Report Generator")
    st.divider()

    if st.session_state.decision_status == "Rejected":
        st.error("❌ The doctor has REJECTED the case — PDF cannot be generated.")
        st.info("Go to 👨‍⚕️ Doctor Review for reanalysis or another review.")
    elif not st.session_state.approved:
        st.warning("⚠️ Please go to Doctor Review first and Approve/Modify!")
        st.info("Click 👨‍⚕️ Doctor Review in the sidebar")
    else:
        status_label = st.session_state.decision_status or "Approved"
        st.success(f"✅ Doctor Status: **{status_label}** — Report ready!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Report Preview")
            st.markdown(f"**👨‍⚕️ Decision Status:** {status_label}")
            if st.session_state.xray_result:
                r = st.session_state.xray_result
                st.markdown(f"**🫁 X-Ray:** {r['label']} ({r['confidence']:.1f}%)")
            if st.session_state.heart_result:
                h = st.session_state.heart_result
                st.markdown(f"**❤️ Heart:** {h['risk']} ({h['probability']:.1f}%)")
            st.markdown(f"**👨‍⚕️ Doctor Notes:** {st.session_state.doctor_notes}")

        with col2:
            if st.button("📥 Generate PDF", type="primary"):
                buffer = io.BytesIO()
                doc    = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story  = []

                # Title
                story.append(Paragraph(
                    "MediCoPilot — AI Medical Report",
                    styles['Title']
                ))
                story.append(Spacer(1, 20))

                # Decision Status
                story.append(Paragraph(
                    f"Doctor Decision Status: {status_label}",
                    styles['Heading2']
                ))
                story.append(Spacer(1, 10))

                # X-Ray Results
                if st.session_state.xray_result:
                    r = st.session_state.xray_result
                    story.append(Paragraph(
                        "Chest X-Ray Analysis",
                        styles['Heading2']
                    ))
                    story.append(Paragraph(
                        f"AI Diagnosis: {r['label']}",
                        styles['Normal']
                    ))
                    story.append(Paragraph(
                        f"Confidence: {r['confidence']:.1f}%",
                        styles['Normal']
                    ))
                    story.append(Spacer(1, 15))

                # Heart Results
                if st.session_state.heart_result:
                    h = st.session_state.heart_result
                    story.append(Paragraph(
                        "Heart Disease Assessment",
                        styles['Heading2']
                    ))
                    story.append(Paragraph(
                        f"Risk Level: {h['risk']}",
                        styles['Normal']
                    ))
                    story.append(Paragraph(
                        f"Disease Probability: {h['probability']:.1f}%",
                        styles['Normal']
                    ))
                    story.append(Paragraph(
                        f"Age: {h['age']} | BP: {h['bp']} | "
                        f"Cholesterol: {h['chol']}",
                        styles['Normal']
                    ))
                    story.append(Spacer(1, 15))

                # Doctor Notes
                story.append(Paragraph(
                    "Doctor Review",
                    styles['Heading2']
                ))
                story.append(Paragraph(
                    st.session_state.doctor_notes or "No notes",
                    styles['Normal']
                ))
                story.append(Spacer(1, 15))
                story.append(Paragraph(
                    "Generated by MediCoPilot AI System",
                    styles['Italic']
                ))

                doc.build(story)
                buffer.seek(0)

                st.download_button(
                    label     = "📥 Download PDF",
                    data      = buffer,
                    file_name = "medical_report.pdf",
                    mime      = "application/pdf"
                )
                st.success("✅ PDF ready — download it!")