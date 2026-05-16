import streamlit as st
import joblib
import numpy as np
import time
import re
import os
import requests

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Luminex | Skill-to-Role Mapping",
    layout="wide"
)

# ======================================================
# API KEYS
# ======================================================
APP_ID = st.secrets["APP_ID"]
APP_KEY = st.secrets["APP_KEY"]

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>

html, body, [class*="stApp"] {
    background-color: #ffffff;
    color: #111827;
}

.hero {
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    padding: 48px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 12px 30px rgba(79,70,229,0.35);
    color: white;
}

.input-card {
    background: #ffffff;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

textarea {
    font-size: 20px !important;
    padding: 14px !important;
    border-radius: 14px !important;
    border: 2px solid #6366F1 !important;
}

.stButton > button {
    font-size: 18px;
    padding: 12px 30px;
    border-radius: 14px;
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    color: white;
    border: none;
}

.role-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 18px;
    border-left: 6px solid #4F46E5;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

.tip-box {
    background: #ECFDF5;
    padding: 24px;
    border-radius: 18px;
    border-left: 6px solid #10B981;
}

.job-card {
    background: #F9FAFB;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 18px;
    border: 1px solid #E5E7EB;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_logistic_model.joblib"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.joblib"
)

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

roles = model.classes_
feature_names = np.array(tfidf.get_feature_names_out())

# ======================================================
# API FUNCTION
# ======================================================
def fetch_jobs(role_query):

    url = (
        f"https://api.adzuna.com/v1/api/jobs/in/search/1"
        f"?app_id={APP_ID}"
        f"&app_key={APP_KEY}"
        f"&what={role_query.replace(' ', '+')}"
    )

    response = requests.get(url)

    if response.status_code == 200:
        return response.json().get("results", [])

    return []

# ======================================================
# HERO
# ======================================================
st.markdown("""
<div class="hero">
    <h1>🚀 Skill2Role AI</h1>
    <h3>Skill-to-Role Mapping with Live Career Opportunities</h3>
</div>
""", unsafe_allow_html=True)

# ======================================================
# INPUT SECTION
# ======================================================
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

# ------------------------------------------------------
# JOB TYPE FILTER
# ------------------------------------------------------
job_type = st.selectbox(
    "🎯 Select Opportunity Type",
    [
        "Full-time",
        "Internship"
    ]
)

st.markdown("### 🧠 Enter Your Skills")

user_input = st.text_area(
    "Skills",
    placeholder="""
Examples:
Python, SQL, Power BI, Data Analysis
Machine Learning, TensorFlow, Scikit-learn
AWS, Docker, Kubernetes
""",
    height=180,
    label_visibility="collapsed"
)

predict = st.button("✨ Predict Roles")

st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VALIDATION
# ======================================================
valid_pattern = re.compile(r"^[A-Za-z0-9.,+\-\s]+$")

if predict:

    # --------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------
    if user_input.strip() == "":
        st.warning("Please enter at least one skill.")

    # --------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------
    elif not valid_pattern.match(user_input):
        st.error("Invalid input format.")

    else:

        with st.spinner("Analyzing skills..."):
            time.sleep(1)

            # ------------------------------------------
            # VECTORIZE
            # ------------------------------------------
            X = tfidf.transform([user_input])

            # ------------------------------------------
            # PREDICT
            # ------------------------------------------
            probs = model.predict_proba(X)[0]

            top_idx = np.argsort(probs)[::-1][:5]

            max_conf = probs[top_idx[0]]

        # ======================================================
        # LOW CONFIDENCE
        # ======================================================
        if max_conf < 0.08:

            st.error("""
Your profile currently shows a very basic skill match.

Please improve foundational skills like:
Python, SQL, Excel, Data Analysis, Communication.
""")

        # ======================================================
        # GOOD PROFILE
        # ======================================================
        else:

            left, right = st.columns([2, 1])

            # ==================================================
            # LEFT SIDE
            # ==================================================
            with left:

                st.subheader("🎯 Top Role Recommendations")

                rank = 1

                for idx in top_idx:

                    conf = int(probs[idx] * 100)

                    # Skip weak predictions
                    if conf < 2:
                        continue

                    st.markdown(f"""
                    <div class="role-card">
                        <h4>{rank}. {roles[idx]}</h4>
                        <p>Confidence Score: <b>{conf}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    rank += 1

            # ==================================================
            # RIGHT SIDE
            # ==================================================
            with right:

                st.subheader("💡 Skill Expansion Tip")

                role_index = np.where(
                    roles == roles[top_idx[0]]
                )[0][0]

                weights = model.coef_[role_index]

                top_features = np.argsort(weights)[::-1][:30]

                skills = [
                    feature_names[i]
                    for i in top_features
                    if len(feature_names[i]) > 2
                ]

                user_tokens = set(
                    user_input.lower().replace(",", " ").split()
                )

                missing = [
                    s for s in skills
                    if s.lower() not in user_tokens
                ]

                if missing:

                    st.markdown(f"""
                    <div class="tip-box">
                    Learn <b>{", ".join(missing[:3])}</b>
                    to unlock more advanced opportunities.
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.success(
                        "Your profile strongly aligns with this role."
                    )

            # ======================================================
            # LIVE OPENINGS
            # ======================================================
            st.subheader("🌍 Live Career Opportunities")

            for idx in top_idx:

                conf = int(probs[idx] * 100)

                # Skip weak predictions
                if conf < 2:
                    continue

                role_name = roles[idx]

                # --------------------------------------------------
                # INTERNSHIP QUERY
                # --------------------------------------------------
                if job_type == "Internship":

                    api_query = f"{role_name} Internship"

                else:

                    api_query = role_name

                # --------------------------------------------------
                # EXPANDER
                # --------------------------------------------------
                with st.expander(
                    f"{role_name} ({conf}% Match)"
                ):

                    st.write(
                        f"Showing {job_type} opportunities related to {role_name}"
                    )

                    jobs = fetch_jobs(api_query)

                    # ----------------------------------------------
                    # NO JOBS
                    # ----------------------------------------------
                    if len(jobs) == 0:

                        st.warning(
                            "No live openings found currently."
                        )

                    # ----------------------------------------------
                    # JOBS
                    # ----------------------------------------------
                    else:

                        for job in jobs[:5]:

                            title = job.get(
                                "title",
                                "N/A"
                            )

                            company = job.get(
                                "company", {}
                            ).get(
                                "display_name",
                                "N/A"
                            )

                            location = job.get(
                                "location", {}
                            ).get(
                                "display_name",
                                "N/A"
                            )

                            salary_min = job.get(
                                "salary_min",
                                "Not Available"
                            )

                            salary_max = job.get(
                                "salary_max",
                                "Not Available"
                            )

                            apply_link = job.get(
                                "redirect_url",
                                None
                            )

                            st.markdown(f"""
                            <div class="job-card">
                                <h4>{title}</h4>
                                <p><b>Company:</b> {company}</p>
                                <p><b>Location:</b> {location}</p>
                                <p><b>Salary:</b> {salary_min} - {salary_max}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            if apply_link:

                                st.link_button(
                                    "🚀 Apply Now",
                                    apply_link
                                    
                                )

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
Built with Machine Learning + Live Job Intelligence
</p>
""", unsafe_allow_html=True)