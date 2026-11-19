
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# --------------------------------------------------
# 2. Custom Styling
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #101321;
    color: #F1F1F5;
}

.block-container {
    padding-top: 2.5rem;
    max-width: 1150px;
}

h1 {
    color: #FFFFFF;
    font-size: 38px !important;
    font-weight: 700 !important;
}

h2, h3 {
    color: #FFFFFF;
}

p, label {
    color: #B8BDD0;
}

div[data-testid="stSlider"] {
    padding-top: 10px;
    padding-bottom: 20px;
}

div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #B69CFF;
}

.result-card {
    background: linear-gradient(135deg, #292449, #382B60);
    border: 1px solid #62518E;
    border-radius: 16px;
    padding: 24px;
    margin-top: 12px;
}

.result-label {
    color: #BDA9FF;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-name {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.tip-card {
    background-color: #1B2032;
    border-left: 4px solid #A78BFA;
    border-radius: 10px;
    padding: 17px;
    margin-top: 15px;
    color: #D7D8E5;
}

div[data-testid="stPlotlyChart"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Load Model and Dataset
# --------------------------------------------------

@st.cache_resource
def load_everything():

    saved = joblib.load("segment_model.pkl")
    customers = pd.read_csv("Mall_Customers_clean.csv")

    return saved, customers


saved, customers = load_everything()

model = saved["model"]
scaler = saved["scaler"]
names = saved["names"]
tips = saved["tips"]


# --------------------------------------------------
# 4. Header
# --------------------------------------------------

st.markdown(
    "<h1>🛍️ Customer Segmentation Studio</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "Discover customer groups using <b>K-Means Clustering</b>.",
    unsafe_allow_html=True
)

st.write("")


# --------------------------------------------------
# 5. Customer Inputs and Prediction
# --------------------------------------------------

left, right = st.columns([1, 1], gap="large")

with left:

    st.subheader("Customer Profile")

    income = st.slider(
        "Annual Income (thousand $)",
        min_value=10,
        max_value=150,
        value=60
    )

    spending = st.slider(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

    new_customer = pd.DataFrame({
        "income": [income],
        "spending": [spending]
    })

    group = int(
        model.predict(scaler.transform(new_customer))[0]
    )


with right:

    st.subheader("Predicted Customer Segment")

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Customer Group {group}</div>
        <div class="result-name">{names[group]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="tip-card">
        <b>💡 Marketing Recommendation</b>
        <br><br>
        {tips[group]}
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 6. Customer Segmentation Chart
# --------------------------------------------------

st.write("")
st.divider()

st.subheader("Customer Segmentation Map")

st.caption(
    "Explore how customers are grouped by annual income and spending score."
)

all_groups = model.predict(
    scaler.transform(customers[["income", "spending"]])
)

fig, ax = plt.subplots(figsize=(11, 5))

fig.patch.set_facecolor("#101321")
ax.set_facecolor("#191E30")

scatter = ax.scatter(
    customers["income"],
    customers["spending"],
    c=all_groups,
    cmap="tab10",
    vmin=0,
    vmax=max(9, len(names) - 1),
    s=65,
    alpha=0.8,
    edgecolors="#252A3B",
    linewidths=0.5
)

# Highlight the selected customer

ax.scatter(
    income,
    spending,
    marker="*",
    s=450,
    color="#FFD166",
    edgecolors="white",
    linewidths=1.3,
    label="Selected Customer",
    zorder=5
)

ax.set_xlabel(
    "Annual Income (thousand $)",
    color="#D4D5E0",
    fontsize=11
)

ax.set_ylabel(
    "Spending Score",
    color="#D4D5E0",
    fontsize=11
)

ax.tick_params(colors="#B8BDD0")

ax.grid(
    True,
    color="#34394D",
    linestyle="--",
    alpha=0.4
)

for spine in ax.spines.values():
    spine.set_color("#34394D")

legend = ax.legend(
    facecolor="#252A3B",
    edgecolor="#454B62",
    labelcolor="white",
    loc="upper right"
)

st.pyplot(fig, use_container_width=True)

plt.close(fig)
