# app.py — Purchase Intent Dashboard
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Purchase Intent Analyzer",
    page_icon="📊",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e9ecef;
    }
    .metric-value { font-size: 32px; font-weight: 600; margin: 4px 0; }
    .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-sub { font-size: 12px; margin-top: 4px; }
    .high { color: #2d6a4f; } .med { color: #b5621a; } .low { color: #c0392b; }
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    .badge-high { background: #d8f3dc; color: #2d6a4f; }
    .badge-med  { background: #fef3cd; color: #b5621a; }
    .badge-low  { background: #fde8e8; color: #c0392b; }
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model('model/model.h5')
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('model/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, scaler, encoder, label_encoders

# ── Preprocessing ─────────────────────────────────────────────────────────────
DROP_COLS = ['Administrative', 'Informational', 'ProductRelated',
             'OperatingSystems', 'Region', 'TrafficType']
CAT_COLS  = ['Month', 'Browser', 'VisitorType']
LE_COLS   = ['Month', 'VisitorType', 'Weekend']

def preprocess(df, scaler, encoder, label_encoders):
    X = df.copy()

    # Drop columns that exist in the data
    existing_drop = [c for c in DROP_COLS if c in X.columns]
    X.drop(existing_drop, axis=1, inplace=True)

    # Drop Revenue and index if present
    for col in ['Revenue', 'index']:
        if col in X.columns:
            X.drop(col, axis=1, inplace=True)

    # Reset index
    X.reset_index(drop=True, inplace=True)

    # Label encode
    for col in LE_COLS:
        if col in X.columns:
            le = label_encoders[col]
            # Handle unseen labels gracefully
            X[col] = X[col].apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else 0
            )

    # One-hot encode
    existing_cat = [c for c in CAT_COLS if c in X.columns]
    if existing_cat:
        cat_encoded = encoder.transform(X[existing_cat])
        cat_df = pd.DataFrame(cat_encoded, index=X.index)
        X.drop(existing_cat, axis=1, inplace=True)
        X = X.join(cat_df)

    # Fix column names (one-hot encoded columns come in as integers)
    X.columns = X.columns.astype(str)

    # Scale
    X_scaled = scaler.transform(X)
    return X_scaled

def score_to_segment(score):
    if score >= 0.65:
        return "High", "badge-high", "🟢"
    elif score >= 0.35:
        return "Medium", "badge-med", "🟡"
    else:
        return "Low", "badge-low", "🔴"

def get_campaign(segment):
    return {
        "High":   "🎁 Offer loyalty points — they're likely buying anyway",
        "Medium": "👥 Show social proof or limited stock warning",
        "Low":    "🏷️ Trigger discount coupon or exit-intent popup"
    }[segment]

# ── Main App ──────────────────────────────────────────────────────────────────
st.title("📊 Purchase Intent Analyzer")
st.markdown("Upload a CSV of session data to score each visitor and get campaign recommendations.")

# Check model exists
if not os.path.exists('model/model.h5'):
    st.error("⚠️ Model files not found. Please run `python train_and_save_model.py` first.")
    st.stop()

model, scaler, encoder, label_encoders = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    high_threshold = st.slider("High intent threshold", 0.5, 0.9, 0.65, 0.05)
    low_threshold  = st.slider("Low intent threshold",  0.1, 0.5, 0.35, 0.05)
    st.markdown("---")
    st.markdown("### 📋 Expected CSV columns")
    st.markdown("""
    Your CSV should have these columns from the UCI Online Shoppers dataset:
    - `Administrative_Duration`
    - `Informational_Duration`  
    - `ProductRelated_Duration`
    - `BounceRates`
    - `ExitRates`
    - `PageValues`
    - `SpecialDay`
    - `Month`
    - `Browser`
    - `VisitorType`
    - `Weekend`
    
    Optional (will be ignored):
    - `Administrative`, `Informational`, `ProductRelated`
    - `OperatingSystems`, `Region`, `TrafficType`
    - `Revenue` (actual outcome, used for accuracy if present)
    """)
    st.markdown("---")
    st.download_button(
        "⬇️ Download sample CSV template",
        data=pd.DataFrame(columns=[
            'Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
            'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
            'VisitorType', 'Weekend', 'Revenue'
        ]).to_csv(index=False),
        file_name="session_template.csv",
        mime="text/csv"
    )

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload session CSV",
    type=["csv"],
    help="Export this from Google Analytics, Mixpanel, or any analytics tool"
)

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_raw):,} sessions")

        with st.spinner("Scoring sessions..."):
            has_revenue = 'Revenue' in df_raw.columns
            X_scaled = preprocess(df_raw, scaler, encoder, label_encoders)
            scores = model.predict(X_scaled, verbose=0).flatten()

        # Build results dataframe
        results = df_raw.copy()
        results['intent_score'] = np.round(scores * 100, 1)
        results['segment'] = results['intent_score'].apply(
            lambda s: "High" if s/100 >= high_threshold
                      else ("Medium" if s/100 >= low_threshold else "Low")
        )
        results['campaign'] = results['segment'].apply(get_campaign)
        if has_revenue:
            results['actual_purchased'] = df_raw['Revenue'].astype(bool)

        # ── KPI Row ───────────────────────────────────────────────────────────
        st.markdown("---")
        total   = len(results)
        n_high  = (results['segment'] == 'High').sum()
        n_med   = (results['segment'] == 'Medium').sum()
        n_low   = (results['segment'] == 'Low').sum()
        avg_score = results['intent_score'].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total sessions", f"{total:,}")
        with col2:
            st.metric("🟢 High intent", f"{n_high:,}", f"{n_high/total*100:.1f}%")
        with col3:
            st.metric("🟡 Medium intent", f"{n_med:,}", f"{n_med/total*100:.1f}%")
        with col4:
            st.metric("🔴 Low intent", f"{n_low:,}", f"{n_low/total*100:.1f}%")
        with col5:
            st.metric("Avg intent score", f"{avg_score:.1f}")

        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p class="section-title">Segment distribution</p>', unsafe_allow_html=True)
            seg_counts = results['segment'].value_counts().reindex(['High', 'Medium', 'Low'])
            seg_df = pd.DataFrame({
                'Segment': seg_counts.index,
                'Count': seg_counts.values
            })
            st.bar_chart(seg_df.set_index('Segment'), color=["#2d6a4f"])

        with col_b:
            st.markdown('<p class="section-title">Intent score distribution</p>', unsafe_allow_html=True)
            hist_df = pd.cut(results['intent_score'], bins=10).value_counts().sort_index()
            st.bar_chart(hist_df)

        # ── Accuracy (if Revenue column present) ─────────────────────────────
        if has_revenue:
            st.markdown("---")
            st.markdown('<p class="section-title">Model accuracy on this dataset</p>', unsafe_allow_html=True)
            predicted_binary = (results['intent_score'] >= 50).astype(int)
            actual_binary    = results['actual_purchased'].astype(int)
            accuracy = (predicted_binary == actual_binary).mean() * 100

            col_x, col_y, col_z = st.columns(3)
            tp = ((predicted_binary == 1) & (actual_binary == 1)).sum()
            fp = ((predicted_binary == 1) & (actual_binary == 0)).sum()
            fn = ((predicted_binary == 0) & (actual_binary == 1)).sum()
            tn = ((predicted_binary == 0) & (actual_binary == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

            with col_x:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            with col_y:
                st.metric("Precision", f"{precision*100:.1f}%")
            with col_z:
                st.metric("Recall", f"{recall*100:.1f}%")

        # ── Segment Breakdown ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">Campaign recommendations by segment</p>', unsafe_allow_html=True)

        for seg, color in [("High", "#d8f3dc"), ("Medium", "#fef3cd"), ("Low", "#fde8e8")]:
            seg_df = results[results['segment'] == seg]
            if len(seg_df) == 0:
                continue
            with st.expander(f"{seg} intent — {len(seg_df):,} visitors ({len(seg_df)/total*100:.1f}%)"):
                st.info(get_campaign(seg))
                st.markdown(f"**Avg intent score:** {seg_df['intent_score'].mean():.1f}")

                display_cols = ['intent_score', 'segment']
                if 'PageValues' in seg_df.columns:
                    display_cols = ['PageValues', 'BounceRates', 'ExitRates',
                                    'ProductRelated_Duration'] + display_cols
                if has_revenue:
                    display_cols.append('actual_purchased')

                st.dataframe(
                    seg_df[[c for c in display_cols if c in seg_df.columns]].head(50),
                    use_container_width=True
                )

        # ── Full results download ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">Export scored sessions</p>', unsafe_allow_html=True)
        st.markdown("Download the full results with intent scores and campaign tags — ready to upload to Mailchimp, Klaviyo, or any CRM.")

        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download scored sessions CSV",
            data=csv_out,
            file_name="scored_sessions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.markdown("Make sure your CSV columns match the expected format shown in the sidebar.")

else:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload CSV")
        st.markdown("Export session data from Google Analytics, Mixpanel, or any analytics platform.")
    with col2:
        st.markdown("#### 2. Model scores each session")
        st.markdown("Your trained neural network predicts purchase probability for every visitor.")
    with col3:
        st.markdown("#### 3. Get campaign actions")
        st.markdown("Download segmented lists ready to upload into your email or ad platform.")