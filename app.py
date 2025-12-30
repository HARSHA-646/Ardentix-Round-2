import streamlit as st
import joblib
import numpy as np
import time
import re

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Luminex | Skill-to-Role Mapping",
    layout="wide"
)

# ======================================================
# CSS – CLEAN WHITE PREMIUM UI
# ======================================================
st.markdown("""
<style>

/* Page background */
html, body, [class*="stApp"] {
    background-color: #ffffff;
    color: #111827;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    padding: 48px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 12px 30px rgba(79,70,229,0.35);
    color: white;
}

/* Input card */
.input-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Textarea */
textarea {
    font-size: 20px !important;
    padding: 14px !important;
    border-radius: 14px !important;
    border: 2px solid #6366F1 !important;
}
textarea:focus {
    box-shadow: 0 0 12px rgba(99,102,241,0.5) !important;
}

/* Predict button */
.stButton > button {
    font-size: 20px;
    padding: 14px 40px;
    border-radius: 14px;
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    color: white;
    border: none;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: scale(1.06);
    box-shadow: 0 10px 25px rgba(79,70,229,0.5);
}

/* Results section */
.results-box {
    background: #F9FAFB;
    padding: 30px;
    border-radius: 20px;
    margin-top: 20px;
    animation: fadeUp 0.6s ease forwards;
}

/* Role card */
.role-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 18px;
    border-left: 6px solid #4F46E5;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* Progress bar */
.progress {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #4F46E5, #22C55E);
}

/* Tip box */
.tip-box {
    background: #ECFDF5;
    padding: 26px;
    border-radius: 18px;
    border-left: 6px solid #10B981;
    box-shadow: 0 6px 18px rgba(16,185,129,0.25);
}

/* Animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(25px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_logistic_model.joblib")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

roles = model.classes_
feature_names = np.array(tfidf.get_feature_names_out())

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div class="hero">
    <h1>🚀 Luminex Hackathon Project</h1>
    <h3>Skill-to-Role Mapping with Actionable Career Tips using Machine Learning</h3>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INPUT
# ======================================================
st.markdown("<div class='input-card'>", unsafe_allow_html=True)
st.markdown("### 🧠 Enter Your Skills / Experience")

user_input = st.text_area(
    "Skills Input",
    placeholder="Python, SQL, Machine Learning, UI/UX, Power BI, Git",
    height=150,
    label_visibility="collapsed"
)

predict = st.button("✨ Predict Job Roles")
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# INPUT VALIDATION
# ======================================================
valid_pattern = re.compile(r"^[A-Za-z.,\s]+$")

if predict:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter at least one skill.")

    elif not valid_pattern.match(user_input):
        st.error(
            "❌ Invalid input.\n\n"
            "Only **letters**, **comma (,)** and **dot (.)** are allowed.\n\n"
            "Example: `Python, Machine Learning, SQL.`"
        )

    else:
        # ======================================================
        # PREDICTION
        # ======================================================
        with st.spinner("🔍 Analyzing skills and predicting roles..."):
            time.sleep(1)

            X = tfidf.transform([user_input])
            probs = model.predict_proba(X)[0]
            top_idx = np.argsort(probs)[::-1][:5]

        st.success("✅ Prediction completed successfully!")

        st.markdown("<div class='results-box'>", unsafe_allow_html=True)
        left, right = st.columns([2.3, 1])

        # LEFT RESULTS
        with left:
            st.markdown("<div class='section-title'>🔍 Top Job Role Recommendations</div>", unsafe_allow_html=True)

            for i, idx in enumerate(top_idx, 1):
                conf = int(probs[idx] * 100)
                st.markdown(f"""
                <div class="role-card">
                    <h4>{i}. {roles[idx]}</h4>
                    <p><b>Match Strength:</b> {conf}%</p>
                    <div class="progress" style="width:{conf}%;"></div>
                </div>
                """, unsafe_allow_html=True)

        # RIGHT TIP
        with right:
            st.markdown("<div class='section-title'>💡 Skill Expansion Tip</div>", unsafe_allow_html=True)

            role_index = np.where(roles == roles[top_idx[0]])[0][0]
            weights = model.coef_[role_index]
            top_features = np.argsort(weights)[::-1][:30]

            skills = [feature_names[i] for i in top_features if len(feature_names[i]) > 2]
            user_tokens = set(user_input.lower().replace(",", " ").split())
            missing = [s for s in skills if s.lower() not in user_tokens]

            tip = (
                f"Learning <b>{', '.join(missing[:3])}</b> can significantly strengthen your profile."
                if missing else
                "Your skills already align strongly with this role 🚀"
            )

            st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center; color:#6B7280;">
Built for <b>Luminex Hackathon</b> • Clean UI • Real ML
</p>
""", unsafe_allow_html=True)
