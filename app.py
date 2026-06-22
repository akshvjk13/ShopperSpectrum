import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# PAGE CONFIGURATION (must be the first Streamlit command executed)

st.set_page_config(
    page_title="ShopperSpectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PATH CONSTANTS

DATA_DIR = "data"
MODELS_DIR = "models"

CUSTOMER_SEGMENTS_PATH = os.path.join(DATA_DIR, "customer_segments.csv")
CLEANED_RETAIL_PATH = os.path.join(DATA_DIR, "cleaned_online_retail.csv")

KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans.pkl")
SCALER_MODEL_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
RECOMMENDATIONS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "recommendations.pkl"
)

# Cluster label mapping as specified by the project requirements.
CLUSTER_LABELS = {
    0: "Occasional",
    1: "High Value",
    2: "Regular",
    3: "At Risk",
}

# Visual styling metadata per segment: emoji, CSS class, and short blurb.
CLUSTER_STYLE = {
    "High Value": {
        "emoji": "🌟",
        "css_class": "segment-high-value",
        "message": "This customer shops often and spends generously. Prioritize retention and loyalty perks.",
    },
    "Regular": {
        "emoji": "🙂",
        "css_class": "segment-regular",
        "message": "A dependable, steady customer. Keep them engaged with consistent offers.",
    },
    "Occasional": {
        "emoji": "🛍",
        "css_class": "segment-occasional",
        "message": "This customer shops infrequently. Targeted promotions could increase engagement.",
    },
    "At Risk": {
        "emoji": "⚠️",
        "css_class": "segment-at-risk",
        "message": "This customer hasn't purchased recently and may be disengaging. Consider a win-back campaign.",
    },
}


# --------------------------------------------------------------------------
# CUSTOM CSS — rounded cards, colored segment boxes, modern look & feel
# --------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Inject custom CSS for a polished, modern UI."""
    st.markdown(
        """
        <style>
        /* ---------- General page polish ---------- */
        .main {
            background-color: #fafafa;
        }
 
        /* ---------- Headings ---------- */
        h1, h2, h3 {
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
 
        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: #1f2937;
        }
        section[data-testid="stSidebar"] * {
            color: #f9fafb !important;
        }
 
        /* ---------- Recommendation Cards ---------- */
        .rec-card {
            background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .rec-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.10);
        }
        .rec-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #6366f1;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        .rec-card-product {
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
        }
 
        /* ---------- Segment Result Boxes ---------- */
        .segment-box {
            border-radius: 18px;
            padding: 1.8rem 2rem;
            margin-top: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
        }
        .segment-title {
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .segment-message {
            font-size: 0.98rem;
            opacity: 0.9;
        }
        .segment-high-value {
            background: linear-gradient(135deg, #fef9c3 0%, #fde68a 100%);
            color: #78350f;
            border: 1px solid #fbbf24;
        }
        .segment-regular {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            color: #1e3a8a;
            border: 1px solid #60a5fa;
        }
        .segment-occasional {
            background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
            color: #4c1d95;
            border: 1px solid #a78bfa;
        }
        .segment-at-risk {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            color: #7f1d1d;
            border: 1px solid #f87171;
        }
 
        /* ---------- Generic info container ---------- */
        .info-pill {
            display: inline-block;
            background-color: #eef2ff;
            color: #4338ca;
            border-radius: 999px;
            padding: 0.25rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
 
        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            padding: 1.2rem 0 0.4rem 0;
            margin-top: 2.5rem;
            border-top: 1px solid #e5e7eb;
            color: #6b7280;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# MODEL / DATA LOADING (cached so files are read only once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    artifacts = {
        "kmeans": None,
        "scaler": None,
        "recommendations": None,
        "segments_df": None,
        "retail_df": None,
        "errors": [],
    }

    # ---- KMeans model ----
    try:
        with open(KMEANS_MODEL_PATH, "rb") as f:
            artifacts["kmeans"] = pickle.load(f)
    except FileNotFoundError:
        artifacts["errors"].append(f"KMeans model not found at `{KMEANS_MODEL_PATH}`.")
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"Failed to load KMeans model: {exc}")

    # ---- Scaler ----
    try:
        with open(SCALER_MODEL_PATH, "rb") as f:
            artifacts["scaler"] = pickle.load(f)
    except FileNotFoundError:
        artifacts["errors"].append(f"Scaler not found at `{SCALER_MODEL_PATH}`.")
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"Failed to load scaler: {exc}")

    # ---- Recommendations (for product recommendations) ----
    try:
        with open(RECOMMENDATIONS_MODEL_PATH, "rb") as f:
            artifacts["recommendations"] = pickle.load(f)
    except FileNotFoundError:
        artifacts["errors"].append(
            f"Recommendations model not found at `{RECOMMENDATIONS_MODEL_PATH}`."
    )
    except Exception as exc:
        artifacts["errors"].append(
            f"Failed to load recommendations model: {exc}"
    )

    # ---- Optional supporting datasets ----
    try:
        if os.path.exists(CUSTOMER_SEGMENTS_PATH):
            artifacts["segments_df"] = pd.read_csv(CUSTOMER_SEGMENTS_PATH)
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"Failed to load customer_segments.csv: {exc}")

    try:
        if os.path.exists(CLEANED_RETAIL_PATH):
            artifacts["retail_df"] = pd.read_csv(CLEANED_RETAIL_PATH)
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"Failed to load cleaned_online_retail.csv: {exc}")

    return artifacts


def get_product_catalog(recommendations_dict):

    if recommendations_dict is None:
        return []

    return sorted(list(recommendations_dict.keys()))


# --------------------------------------------------------------------------
# RECOMMENDATION LOGIC
# --------------------------------------------------------------------------
def recommend(product, recommendations_dict, n=5):

    product = product.upper().strip()

    if product not in recommendations_dict:
        raise ValueError(
            f"Product '{product}' was not found in the catalog."
        )

    products = recommendations_dict[product]["products"][:n]
    scores = recommendations_dict[product]["scores"][:n]

    return list(zip(products, scores)), product

# --------------------------------------------------------------------------
# MODULE 1: PRODUCT RECOMMENDATION PAGE
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# MODULE 1: PRODUCT RECOMMENDATION PAGE
# --------------------------------------------------------------------------
def product_recommendation_page(artifacts: dict) -> None:
    """Render the Product Recommendation module."""

    st.title("🎁 Product Recommendation")

    st.markdown(
        "Discover products frequently bought alongside an item using "
        "**item-based collaborative filtering**."
    )

    st.divider()

    recommendations_dict = artifacts["recommendations"]

    if recommendations_dict is None:
        st.error(
            "🚫 The recommendations model (`models/recommendations.pkl`) could not be loaded."
        )
        return

    catalog = get_product_catalog(recommendations_dict)

    col_input, col_info = st.columns([2.2, 1])

    with col_input:

        product_name = st.text_input(
            "Enter Product Name",
            placeholder="e.g. JUMBO BAG RED RETROSPOT"
        )

        n_recs = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5
        )

        get_recs_clicked = st.button(
            "🔍 Get Recommendations",
            type="primary",
            use_container_width=True
        )

    with col_info:

        st.markdown(
            f"""
            <div class="rec-card">
                <div class="rec-card-title">Catalog Size</div>
                <div class="rec-card-product">{len(catalog):,} products</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if catalog:

            with st.expander("📋 Browse sample products"):

                sample_size = min(15, len(catalog))

                for p in catalog[:sample_size]:
                    st.markdown(f"- {p}")

    st.markdown("")

    if get_recs_clicked:

        if not product_name.strip():

            st.warning(
                "⚠️ Please enter a product name before requesting recommendations."
            )

            return

        with st.spinner("Finding similar products..."):

            try:

                results, matched_product = recommend(
                    product_name,
                    recommendations_dict,
                    n=n_recs
                )

            except ValueError as e:

                st.error(str(e))

                return

            except Exception as exc:

                st.error(f"❌ An unexpected error occurred: {exc}")

                return

        if not results:

            st.warning(
                "⚠️ No similar products could be found for this item."
            )

            return

        st.success(
            f"✅ Top {len(results)} Recommended Products for **{matched_product}**"
        )

        st.markdown("")

        cards_per_row = 2

        for row_start in range(0, len(results), cards_per_row):

            row_items = results[row_start:row_start + cards_per_row]

            cols = st.columns(len(row_items))

            for col, (idx, (name, score)) in zip(
                    cols,
                    enumerate(row_items, start=row_start + 1)
            ):

                with col:

                    st.success(f"🎁 Recommendation {idx}")

                    st.markdown(
                        f"#### {name}"
                    )

                    st.caption(
                        f"Similarity Score: {score:.2f}"
                    )

# --------------------------------------------------------------------------
# MODULE 2: CUSTOMER SEGMENTATION PAGE
# --------------------------------------------------------------------------
def customer_segmentation_page(artifacts: dict) -> None:
    """Render the Customer Segmentation module."""
    st.title("📊 Customer Segmentation")
    st.markdown(
        "Predict a customer's segment using their **RFM (Recency, Frequency, "
        "Monetary)** profile and a pre-trained **KMeans** clustering model."
    )
    st.divider()

    kmeans = artifacts["kmeans"]
    scaler = artifacts["scaler"]

    if kmeans is None or scaler is None:
        st.error(
            "🚫 Required models could not be loaded. Please make sure "
            "`models/kmeans.pkl` and `models/scaler.pkl` both exist and are valid."
        )
        return

    col_form, col_help = st.columns([2, 1])

    with col_form:
        st.subheader("Enter Customer Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input(
                "Recency (days)", min_value=0, max_value=3650, value=30, step=1,
                help="Days since the customer's last purchase.",
            )
        with c2:
            frequency = st.number_input(
                "Frequency (purchases)", min_value=0, max_value=10000, value=5, step=1,
                help="Total number of purchases made by the customer.",
            )
        with c3:
            monetary = st.number_input(
                "Monetary (total spend)", min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0,
                help="Total amount spent by the customer.",
            )

        predict_clicked = st.button("🚀 Predict Cluster", type="primary", use_container_width=True)

    with col_help:
        st.info(
            "ℹ️ **RFM Quick Guide**\n\n"
            "- **Recency**: lower is better (recent activity)\n"
            "- **Frequency**: higher means more loyal\n"
            "- **Monetary**: higher means more revenue generated"
        )

    st.markdown("")

    if predict_clicked:
        try:
            with st.spinner("Scaling features and predicting cluster..."):
                input_df = pd.DataFrame(
                    {
                        "Recency": [recency],
                        "Frequency": [frequency],
                        "Monetary": [monetary],
                    }
                )

                scaled_features = scaler.transform(input_df)
                cluster_id = int(kmeans.predict(scaled_features)[0])

            segment_name = CLUSTER_LABELS.get(cluster_id, "Unknown")
            style = CLUSTER_STYLE.get(
                segment_name,
                {"emoji": "❓", "css_class": "segment-occasional", "message": "Segment information unavailable."},
            )

            st.markdown(
                f"""
                <div class="segment-box {style['css_class']}">
                    <div class="segment-title">{style['emoji']} {segment_name} Customer</div>
                    <div class="segment-message">{style['message']}</div>
                    <div class="info-pill">Cluster Number: {cluster_id}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success("✅ Prediction complete.")

            with st.expander("📐 View input summary"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Failed to predict cluster: {exc}")


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page key."""
    with st.sidebar:
        st.markdown("## 🛒 ShopperSpectrum")
        st.caption("Customer Segmentation & Product Recommendation System")
        st.markdown("---")

        page = st.radio(
            "Navigate to:",
            options=["🎁 Product Recommendation", "📊 Customer Segmentation"],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            **About this app**

            ShopperSpectrum combines unsupervised learning and
            collaborative filtering to help retailers:

            - 🎯 Recommend relevant products
            - 👥 Understand customer segments
            - 📈 Drive targeted marketing strategies
            """
        )
        st.markdown("---")
        st.caption("v1.0.0 · Built with Streamlit")

    return page


# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            🛒 <strong>ShopperSpectrum</strong> &nbsp;|&nbsp; Developed by <strong>Akshitha Chunchu</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------------------------------
def main() -> None:
    """Main application entry point."""
    inject_custom_css()

    with st.spinner("Loading models and data..."):
        artifacts = load_models()

    page = render_sidebar()

    # Surface any loading issues once, near the top of the main content area.
    if artifacts["errors"]:
        with st.expander("⚠️ Some resources failed to load — click to view details", expanded=False):
            for err in artifacts["errors"]:
                st.warning(err)

    if page == "🎁 Product Recommendation":
        product_recommendation_page(artifacts)
    else:
        customer_segmentation_page(artifacts)

    render_footer()


if __name__ == "__main__":
    main()