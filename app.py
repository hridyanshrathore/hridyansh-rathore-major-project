# ----------------------------------------------------------
# app.py  — v3  (Upgrades 2 + 3 + 4 + 5)
# ----------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import io
from pathlib import Path
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; font-weight: 400; }
.stApp { background: #F7F6F2; }
.block-container { padding-top: 2rem !important; }
.app-header { background: #1C2B3A; color: #F0EDE6; padding: 2.2rem 2.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.8rem; }
.app-header h1 { color: #F0EDE6; font-size: 2.2rem; margin:0; }
.app-header p  { color: #9BAEBB; margin: 0.4rem 0 0; font-size: 1rem; }
.step-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: #7A8A96; margin-bottom: 0.3rem; }
.result-card { background: #FFFFFF; border: 1px solid #E4E0D8; border-radius: 14px; padding: 1.6rem; margin-bottom: 1rem; }
.disease-name { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #1C2B3A; margin: 0; }
.confidence-pill { display: inline-block; background: #E8F5E9; color: #2E7D32; border-radius: 20px; padding: 3px 12px; font-size: 0.85rem; font-weight: 500; margin-top: 0.4rem; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.15rem; color: #1C2B3A; margin: 1.4rem 0 0.6rem; }
.precaution-item { background: #F0F4F8; border-left: 3px solid #1C2B3A; padding: 0.5rem 0.9rem; border-radius: 0 8px 8px 0; margin: 0.35rem 0; font-size: 0.93rem; color: #334155; }
.disclaimer { background: #FFF8E1; border: 1px solid #FFE082; border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.82rem; color: #795548; margin-top: 1rem; }
.stButton > button { background: #1C2B3A !important; color: #F0EDE6 !important; border: none !important; border-radius: 10px !important; padding: 0.55rem 2rem !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important; font-weight: 500 !important; }
.stMultiSelect [data-baseweb="tag"] { background: #1C2B3A !important; color: #F0EDE6 !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("disease_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    symptoms      = joblib.load("symptom_columns.pkl")
    return model, label_encoder, symptoms

model, label_encoder, symptoms = load_artifacts()

# ── Load optional CSVs ───────────────────────────────────────
@st.cache_data
def load_extra():
    desc_df = prec_df = None
    if Path("symptom_Description.csv").exists():
        try:
            desc_df = pd.read_csv("symptom_Description.csv", encoding="latin1")
            desc_df.columns = [c.strip() for c in desc_df.columns]
        except Exception:
            pass
    if Path("symptom_precaution.csv").exists():
        try:
            prec_df = pd.read_csv("symptom_precaution.csv", encoding="latin1")
            prec_df.columns = [c.strip() for c in prec_df.columns]
        except Exception:
            pass
    return desc_df, prec_df

desc_df, prec_df = load_extra()

@st.cache_data
def load_dataset():
    if not Path("dataset.csv").exists():
        return None
    df = pd.read_csv("dataset.csv", encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    return df

dataset_df = load_dataset()

# ── SHAP explainer ───────────────────────────────────────────
@st.cache_resource
def get_explainer(_model, _symptoms):
    background = np.zeros((1, len(_symptoms)), dtype=np.float32)
    return shap.TreeExplainer(_model, background)

# ── Body-system symptom grouping ─────────────────────────────
BODY_SYSTEMS = {
    "All symptoms": [],
    "🌡️ Fever & Infection": ["fever","high_fever","mild_fever","chills","sweating",
                              "shivering","cold_hands_and_feets","fatigue","malaise",
                              "lethargy","weakness_in_limbs","weight_loss"],
    "🫁 Respiratory":       ["cough","breathlessness","phlegm","mucoid_sputum",
                              "rusty_sputum","throat_irritation","runny_nose",
                              "congestion","chest_pain","fast_heart_rate"],
    "🍽️ Digestive":         ["nausea","vomiting","diarrhoea","constipation","acidity",
                              "stomach_pain","abdominal_pain","belly_pain","indigestion",
                              "loss_of_appetite","stomach_bleeding","distention_of_abdomen"],
    "🧠 Neurological":      ["headache","dizziness","loss_of_balance","lack_of_concentration",
                              "altered_sensorium","slurred_speech","spinning_movements",
                              "visual_disturbances","unsteadiness"],
    "🦴 Musculoskeletal":   ["joint_pain","muscle_pain","back_pain","neck_pain",
                              "knee_pain","hip_joint_pain","muscle_wasting",
                              "cramps","stiff_neck","swollen_joints",
                              "movement_stiffness","painful_walking"],
    "🩸 Skin & Allergy":    ["itching","skin_rash","nodal_skin_eruptions","dischromic_patches",
                              "skin_peeling","silver_like_dusting","small_dents_in_nails",
                              "inflammatory_nails","blister","red_sore_around_nose",
                              "yellowish_skin","yellowing_of_eyes","bruising","pus_filled_pimples"],
    "🚽 Urinary":           ["burning_micturition","spotting_urination","dark_urine",
                              "yellow_urine","continuous_feel_of_urine","bladder_discomfort",
                              "foul_smell_of_urine","polyuria"],
}

def symptoms_for_system(system_name):
    sym_set = set(symptoms)
    if system_name == "All symptoms":
        return symptoms
    return [s for s in BODY_SYSTEMS.get(system_name, []) if s in sym_set]

def build_input(selected_raw):
    x = np.zeros((1, len(symptoms)), dtype=np.float32)
    for s in selected_raw:
        if s in symptoms:
            x[0, symptoms.index(s)] = 1
    return x

# ── PDF report generator (Upgrade 5) ─────────────────────────
def generate_pdf_report(selected_raw, top3_diseases, top3_scores, precautions):
    # Margins MUST be set before add_page() — fpdf2 bakes them in at page creation
    L, T, R = 15, 15, 15          # left / top / right margins in mm
    PW      = 210                  # A4 page width mm
    USABLE  = PW - L - R          # 180 mm  — every cell/line must stay within this

    pdf = FPDF()
    pdf.set_margins(L, T, R)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header bar (drawn from edge, not affected by margins) ──
    pdf.set_fill_color(28, 43, 58)
    pdf.rect(0, 0, PW, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(240, 237, 230)
    pdf.set_xy(L, 7)
    pdf.cell(USABLE, 14, "Disease Prediction Report", align="C")

    pdf.set_xy(L, 36)
    pdf.set_text_color(30, 30, 30)

    # ── Disclaimer ─────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 100, 50)
    pdf.multi_cell(USABLE, 5,
        "DISCLAIMER: This report is generated by an AI/ML model for educational "
        "purposes only. It is NOT a medical diagnosis. Always consult a licensed "
        "medical professional.")
    pdf.ln(4)

    def section_heading(title):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(28, 43, 58)
        pdf.cell(USABLE, 8, title, ln=True)
        pdf.set_draw_color(28, 43, 58)
        pdf.set_line_width(0.4)
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.line(x, y, x + USABLE, y)
        pdf.ln(4)

    # ── Symptoms ───────────────────────────────────────────────
    section_heading("Symptoms Reported")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 65, 80)
    for s in selected_raw:
        pdf.cell(USABLE, 6, "-  " + s.replace("_", " ").title(), ln=True)
    pdf.ln(4)

    # ── Top prediction ─────────────────────────────────────────
    section_heading("Top Prediction")
    pdf.set_fill_color(232, 245, 233)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 80, 40)
    label = top3_diseases[0] + "   (" + f"{top3_scores[0]:.1f}%" + " confidence)"
    pdf.cell(USABLE, 12, label, fill=True, ln=True)
    pdf.ln(4)

    # ── Top 3 table ────────────────────────────────────────────
    section_heading("Top 3 Possible Diseases")
    # Column widths must sum exactly to USABLE (180)
    cw = [12, 128, 40]   # 12 + 128 + 40 = 180
    pdf.set_fill_color(28, 43, 58)
    pdf.set_text_color(240, 237, 230)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(cw[0], 8, "#",          border=0, fill=True)
    pdf.cell(cw[1], 8, "Disease",    border=0, fill=True)
    pdf.cell(cw[2], 8, "Confidence", border=0, fill=True, ln=True)

    row_fills = [(255, 255, 255), (240, 244, 248)]
    for i, (disease, score) in enumerate(zip(top3_diseases, top3_scores)):
        r, g, b = row_fills[i % 2]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(50, 65, 80)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(cw[0], 7, str(i + 1),        border=0, fill=True)
        pdf.cell(cw[1], 7, disease,            border=0, fill=True)
        pdf.cell(cw[2], 7, f"{score:.1f}%",   border=0, fill=True, ln=True)
    pdf.ln(5)

    # ── Precautions ────────────────────────────────────────────
    if precautions:
        section_heading("Recommended Precautions")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 65, 80)
        for p in precautions:
            pdf.multi_cell(USABLE, 6, "-  " + str(p))
        pdf.ln(3)

    # ── Footer ─────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(USABLE, 5, "Generated by Disease Predictor - AI/ML College Project", align="C")

    return bytes(pdf.output())


# ════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════
tab_predict, tab_dataset = st.tabs(["🩺 Predict", "📊 Dataset Overview"])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR (Upgrades 2 + 3 + 5)
# ════════════════════════════════════════════════════════════
with tab_predict:

    st.markdown("""
    <div class="app-header">
        <h1>🩺 Disease Predictor</h1>
        <p>Select your symptoms below for an AI-powered disease prediction with explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1
    st.markdown('<div class="step-label">Step 1 — Filter by body system</div>',
                unsafe_allow_html=True)
    system_choice = st.selectbox("body system", list(BODY_SYSTEMS.keys()),
                                 label_visibility="collapsed")

    # Step 2
    available_raw = symptoms_for_system(system_choice)
    nice_to_raw   = {s.replace("_", " ").title(): s for s in available_raw}
    choices_nice  = sorted(nice_to_raw.keys())

    st.markdown('<div class="step-label">Step 2 — Choose your symptoms</div>',
                unsafe_allow_html=True)
    selected_nice = st.multiselect("symptoms", choices_nice,
                                   placeholder="Type or scroll to pick symptoms…",
                                   label_visibility="collapsed")
    selected_raw = [nice_to_raw[n] for n in selected_nice]

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("Predict Disease", use_container_width=True)
    with col_info:
        if selected_raw:
            st.caption(f"✔ {len(selected_raw)} symptom(s) selected")
        else:
            st.caption("Select at least one symptom to predict.")

    # ── Prediction ───────────────────────────────────────────
    if predict_clicked:
        if not selected_raw:
            st.warning("Please select at least one symptom first.")
            st.stop()

        x = build_input(selected_raw)
        proba    = model.predict_proba(x)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_diseases = label_encoder.inverse_transform(top3_idx)
        top3_scores   = proba[top3_idx] * 100

        # Gather precautions for PDF
        precaution_items = []
        if prec_df is not None:
            dcol = next((c for c in prec_df.columns if c.lower().startswith("disease")), None)
            pres = [c for c in prec_df.columns if "precaution" in c.lower()]
            if dcol and pres:
                row = prec_df[prec_df[dcol].astype(str).str.strip().str.lower()
                              == top3_diseases[0].strip().lower()]
                if not row.empty:
                    precaution_items = [str(row.iloc[0][c]) for c in pres
                                        if pd.notna(row.iloc[0][c])]

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            # Primary result
            st.markdown(f"""
            <div class="result-card">
                <div class="step-label">Top prediction</div>
                <p class="disease-name">{top3_diseases[0]}</p>
                <span class="confidence-pill">✓ {top3_scores[0]:.1f}% confidence</span>
            </div>
            """, unsafe_allow_html=True)

            # Description
            if desc_df is not None:
                dcol = next((c for c in desc_df.columns if c.lower().startswith("disease")), None)
                ccol = next((c for c in desc_df.columns if "description" in c.lower()), None)
                if dcol and ccol:
                    row = desc_df[desc_df[dcol].astype(str).str.strip().str.lower()
                                  == top3_diseases[0].strip().lower()]
                    if not row.empty:
                        st.markdown('<div class="section-title">About this disease</div>',
                                    unsafe_allow_html=True)
                        st.write(row.iloc[0][ccol])

            # Precautions
            if precaution_items:
                st.markdown('<div class="section-title">Recommended precautions</div>',
                            unsafe_allow_html=True)
                for it in precaution_items:
                    st.markdown(f'<div class="precaution-item">• {it}</div>',
                                unsafe_allow_html=True)

            # ── Upgrade 5: PDF Download ──────────────────────
            st.markdown('<div class="section-title">Download report</div>',
                        unsafe_allow_html=True)
            try:
                pdf_bytes = generate_pdf_report(
                    selected_raw, top3_diseases, top3_scores, precaution_items
                )
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"disease_report_{top3_diseases[0].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.info(f"PDF generation unavailable: {e}. Run: pip install fpdf2")

            st.markdown("""
            <div class="disclaimer">
            ⚠️ This tool is for educational purposes only. Always consult a licensed
            medical professional for diagnosis and treatment.
            </div>
            """, unsafe_allow_html=True)

        with right_col:
            # Top-3 chart
            st.markdown('<div class="section-title">Top 3 possible diseases</div>',
                        unsafe_allow_html=True)
            colors = ["#1C2B3A", "#4A6FA5", "#9BAEBB"]
            fig_bar = go.Figure(go.Bar(
                x=top3_scores[::-1], y=top3_diseases[::-1],
                orientation="h", marker_color=colors[::-1],
                text=[f"{v:.1f}%" for v in top3_scores[::-1]],
                textposition="inside", insidetextanchor="start",
                textfont=dict(color="white", size=12),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=20, t=10, b=10),
                xaxis=dict(showgrid=False, showticklabels=False,
                           range=[0, max(top3_scores) * 1.15]),
                yaxis=dict(showgrid=False,
                           tickfont=dict(family="DM Sans", size=13, color="#1C2B3A")),
                height=180,
            )
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={"displayModeBar": False})

            # SHAP
            st.markdown('<div class="section-title">Why this prediction? (SHAP)</div>',
                        unsafe_allow_html=True)
            st.caption("How much each symptom contributed to the top prediction.")
            try:
                explainer      = get_explainer(model, symptoms)
                shap_values    = explainer.shap_values(x)
                pred_class_idx = int(top3_idx[0])

                # Handle all possible SHAP output shapes depending on version:
                # Old SHAP  -> list of arrays, one per class, each shape (1, n_features)
                # New SHAP  -> single ndarray of shape (n_samples, n_features, n_classes)
                #           or (n_samples, n_features) for binary/single-output
                sv_raw = np.array(shap_values)
                if isinstance(shap_values, list):
                    sv = np.array(shap_values[pred_class_idx]).flatten()
                elif sv_raw.ndim == 3:
                    sv = sv_raw[0, :, pred_class_idx]
                elif sv_raw.ndim == 2:
                    sv = sv_raw[0]
                else:
                    sv = sv_raw.flatten()

                selected_indices = [i for i, s in enumerate(symptoms) if s in selected_raw]
                if selected_indices:
                    shap_selected = [(symptoms[i].replace("_", " ").title(), float(sv[i]))
                                     for i in selected_indices]
                    shap_selected.sort(key=lambda t: abs(t[1]), reverse=True)
                    shap_selected = shap_selected[:8]
                    labels     = [t[0] for t in shap_selected]
                    vals       = [t[1] for t in shap_selected]
                    bar_colors = ["#1C8B3A" if v >= 0 else "#C62828" for v in vals]

                    fig_shap = go.Figure(go.Bar(
                        x=vals[::-1], y=labels[::-1], orientation="h",
                        marker_color=bar_colors[::-1],
                        text=[f"+{v:.3f}" if v >= 0 else f"{v:.3f}" for v in vals[::-1]],
                        textposition="outside",
                        textfont=dict(family="DM Sans", size=11, color="#334155"),
                    ))
                    fig_shap.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=60, t=10, b=10),
                        xaxis=dict(showgrid=True, gridcolor="#E4E0D8",
                                   zeroline=True, zerolinecolor="#1C2B3A",
                                   zerolinewidth=1.5, showticklabels=False),
                        yaxis=dict(tickfont=dict(family="DM Sans", size=12, color="#334155")),
                        height=max(180, len(shap_selected) * 38),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.caption("🟢 Green = pushes toward this disease  |  🔴 Red = pushes away")
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")


# ════════════════════════════════════════════════════════════
# TAB 2 — DATASET OVERVIEW (Upgrade 4)
# ════════════════════════════════════════════════════════════
with tab_dataset:

    st.markdown("""
    <div class="app-header">
        <h1>📊 Dataset Overview</h1>
        <p>Explore the training data used to build the prediction model.</p>
    </div>
    """, unsafe_allow_html=True)

    if dataset_df is None:
        st.warning("dataset.csv not found. Place it in the same folder as app.py.")
    else:
        df = dataset_df
        symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
        n_diseases   = df["Disease"].nunique() if "Disease" in df.columns else 0
        n_rows       = len(df)
        n_dupes      = df.duplicated().sum()

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total records",   f"{n_rows:,}")
        c2.metric("Unique diseases", f"{n_diseases}")
        c3.metric("Unique symptoms", f"{len(symptoms)}")
        c4.metric("Duplicate rows",  f"{n_dupes}",
                  delta=f"-{n_dupes} cleaned" if n_dupes else "None found",
                  delta_color="inverse")

        st.markdown("---")
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown('<div class="section-title">Disease class distribution</div>',
                        unsafe_allow_html=True)
            st.caption("Number of training records per disease.")
            counts = df["Disease"].value_counts().reset_index()
            counts.columns = ["Disease", "Count"]
            fig = px.bar(counts, x="Count", y="Disease", orientation="h",
                         color="Count",
                         color_continuous_scale=["#9BAEBB", "#4A6FA5", "#1C2B3A"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=20, t=10, b=10),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#E4E0D8", title=""),
                yaxis=dict(title="", tickfont=dict(size=11)),
                height=max(400, len(counts) * 22),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_right:
            st.markdown('<div class="section-title">Top 20 most frequent symptoms</div>',
                        unsafe_allow_html=True)
            st.caption("How often each symptom appears across all training records.")
            sym_counts = {}
            for col in symptom_cols:
                for val in df[col].dropna().astype(str).str.strip().str.lower():
                    if val:
                        sym_counts[val] = sym_counts.get(val, 0) + 1
            top20  = sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            sym_df = pd.DataFrame(top20, columns=["Symptom", "Frequency"])
            sym_df["Symptom"] = sym_df["Symptom"].str.replace("_", " ").str.title()
            fig2 = px.bar(sym_df[::-1], x="Frequency", y="Symptom", orientation="h",
                          color="Frequency",
                          color_continuous_scale=["#E1F5EE", "#1D9E75", "#085041"])
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=20, t=10, b=10),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#E4E0D8", title=""),
                yaxis=dict(title="", tickfont=dict(size=11)),
                height=500,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title">Raw data preview</div>',
                    unsafe_allow_html=True)
        st.caption("First 20 rows of the training dataset.")
        st.dataframe(df.head(20), use_container_width=True, height=280)