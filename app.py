import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import plotly.express as px
import plotly.graph_objects as go
from numpy.linalg import norm

EXCEL_FILE = "Enchanto_Products.xlsx"
PRODUCT_SHEET = "Products"
PRODUCT_ID_COL = "Product ID"
PRODUCT_NAME_COL = "Product Name"
IMAGE_URL_COL = "Image_URL"

st.set_page_config(page_title="Enchanto Recommendations", layout="wide")

# ----------------------------------------------------------------------
# PAGE CSS (Pastel Gradient + Glassmorphism + Compact)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background - Reverted to Default (White) */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Reduce Top Padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Reduce Header Margins */
    h1, h2, h3 {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar - Neutral Theme */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e2e8f0;
    }

    /* Standard Cards for Main Area (Clean White) */
    .glass-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Product Card Styling */
    .rec-img {
        border-radius: 12px;
        margin-bottom: 8px;
        width: 100%;
        height: 200px;
        object-fit: cover;
    }
    .rec-title {
        font-weight: 700;
        font-size: 1rem;
        color: #1e293b;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rec-price {
        font-weight: 800;
        font-size: 1.1rem;
        color: #0f766e;
    }
    .rec-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 99px;
        background: #ecfdf5;
        color: #047857;
        font-size: 0.7rem;
        margin-top: 4px;
        font-weight: 600;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    /* Justified Tabs */
    button[data-baseweb="tab"] {
        flex-grow: 1;
    }
    
    /* Sky Blue View Buttons */
    div[data-testid="stButton"] > button {
        background-color: #87CEEB !important; /* Sky Blue */
        color: #0f172a !important; /* Dark Slate for contrast */
        border: 1px solid #7dd3fc !important;
        font-weight: 500 !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #38bdf8 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# HEADER
# ======================================================================
import os
col_logo, col_title = st.columns([1, 6])

with col_logo:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=80)
    else:
        st.write("🧴")

with col_title:
    # Italic, smaller font to fit in one line
    st.markdown("""
    <h2 style='font-style: italic; margin-top: 10px; font-size: 1.8rem;'>
        Enchanto – AI-Powered Recommendation Dashboard
    </h2>
    """, unsafe_allow_html=True)
st.caption("Collaborative • Content-Based • Hybrid | CLV • Engagement • Segmentation")

# ======================================================================
# LOAD PRODUCTS
# ======================================================================
@st.cache_data
def load_products():
    try:
        products = pd.read_excel(EXCEL_FILE, sheet_name=PRODUCT_SHEET)
        if "Bottle Size" in products.columns:
            products["BottleML"] = (
                products["Bottle Size"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            )
        else:
            products["BottleML"] = 50.0
        return products
    except Exception as e:
        st.error(f"Error loading products: {e}")
        return pd.DataFrame()

products = load_products()

# ======================================================================
# GENERATE MEANINGFUL DATA (PERSONAS)
# ======================================================================
@st.cache_data
def generate_meaningful_data(n_users=200):
    random.seed(42)
    np.random.seed(42)
    
    if products.empty:
        return pd.DataFrame(), pd.DataFrame()

    regions = ["North", "South", "East", "West"]
    cats = products["Category"].dropna().unique().tolist()
    scents = products["Scent Notes"].dropna().unique().tolist()
    
    # Define Personas
    personas = [
        {"name": "The Loyalist", "weight": 0.2, "income_range": (80000, 150000), "activity_mult": 2.0, "buy_prob": 0.4},
        {"name": "The Window Shopper", "weight": 0.3, "income_range": (20000, 50000), "activity_mult": 1.5, "buy_prob": 0.05},
        {"name": "The Big Spender", "weight": 0.1, "income_range": (150000, 300000), "activity_mult": 1.0, "buy_prob": 0.6},
        {"name": "The Newbie", "weight": 0.2, "income_range": (30000, 70000), "activity_mult": 0.5, "buy_prob": 0.1},
        {"name": "The Deal Hunter", "weight": 0.2, "income_range": (25000, 60000), "activity_mult": 1.2, "buy_prob": 0.2}
    ]
    
    users = []
    activities = []
    
    prod_ids = products[PRODUCT_ID_COL].tolist()
    price_map = products.set_index(PRODUCT_ID_COL)["Price"].to_dict()
    
    for i in range(1, n_users + 1):
        uid = f"U{str(i).zfill(3)}"
        
        # Assign Persona
        persona = random.choices(personas, weights=[p["weight"] for p in personas], k=1)[0]
        
        age = random.randint(18, 60)
        income = random.randint(*persona["income_range"])
        gender = random.choice(["Male", "Female"])
        region = random.choice(regions)
        pref_cat = random.choice(cats)
        pref_scent = random.choice(scents)
        
        users.append({
            "user_id": uid,
            "persona": persona["name"],
            "gender": gender,
            "age": age,
            "income": income,
            "region": region,
            "marital_status": random.choice(["Single", "Married", "Other"]),
            "occupation": random.choice(["Student", "Professional", "Self-Employed", "Homemaker"]),
            "preferred_category": pref_cat,
            "preferred_scent": pref_scent
        })
        
        # Generate Activities based on Persona
        # Adjusted for small catalog (56 products): range 3-12 interactions
        n_interactions = int(random.randint(3, 12) * persona["activity_mult"])
        
        # Select products: Bias towards preferred category/scent
        candidates = products[
            (products["Category"] == pref_cat) | (products["Scent Notes"] == pref_scent)
        ][PRODUCT_ID_COL].tolist()
        
        # Fill with randoms if not enough candidates
        if len(candidates) < n_interactions:
            candidates.extend(random.sample(prod_ids, min(len(prod_ids), n_interactions - len(candidates))))
            
        selected_pids = random.sample(candidates, min(len(candidates), n_interactions))
        
        for pid in selected_pids:
            # Interaction Logic
            views = random.randint(1, 5)
            
            # Purchase logic based on persona buy_prob and price
            price = price_map.get(pid, 1000)
            price_factor = 1.0
            if persona["name"] == "The Window Shopper" and price > 5000:
                price_factor = 0.1
            if persona["name"] == "The Big Spender":
                price_factor = 1.5
                
            buy_chance = persona["buy_prob"] * price_factor
            
            wishlist = 1 if random.random() < 0.3 else 0
            add_to_cart = 1 if random.random() < (buy_chance + 0.1) else 0
            payment_page = 1 if add_to_cart and random.random() < 0.8 else 0
            purchase_count = 0
            if payment_page and random.random() < 0.9:
                purchase_count = random.choice([1, 1, 2])
            
            # Rating
            rating = np.nan
            if purchase_count > 0:
                rating = random.choice([4, 5, 5, 3]) # Bias towards positive
            elif wishlist:
                rating = random.choice([3, 4, 5])
                
            activities.append({
                "user_id": uid,
                "product_id": pid,
                "views": views,
                "wishlist": wishlist,
                "add_to_cart": add_to_cart,
                "payment_page": payment_page,
                "purchase_count": purchase_count,
                "rating": rating,
                "price": price # Track price for accurate spend calc
            })
            
    users_df = pd.DataFrame(users)
    activity_df = pd.DataFrame(activities)
    
    # Calculate Engagement Score
    def engagement(row):
        score = (
            0.1 * row["views"] +
            0.5 * row["wishlist"] +
            1.0 * row["add_to_cart"] +
            2.0 * row["purchase_count"]
        )
        if not pd.isna(row["rating"]):
            score += 0.5 * (row["rating"] - 3)
        return score

    activity_df["engagement_score"] = activity_df.apply(engagement, axis=1)
    activity_df["cf_rating"] = activity_df["engagement_score"].clip(1, 5) # Simple scaling
    activity_df["is_positive"] = (activity_df["purchase_count"] > 0) | (activity_df["rating"] >= 4)
    
    # --- ENRICH USER DATA ---
    # Calculate aggregate metrics per user for the dashboard
    user_metrics = activity_df.groupby("user_id").agg({
        "views": "sum",
        "purchase_count": "sum",
        "add_to_cart": "sum",
        "price": lambda x: (x * activity_df.loc[x.index, "purchase_count"]).sum() # Real Spend
    }).rename(columns={"price": "total_spend"})
    
    # Get Cart Items (Product Names)
    cart_items = activity_df[activity_df["add_to_cart"] > 0].groupby("user_id")["product_id"].apply(
        lambda x: ", ".join(products[products[PRODUCT_ID_COL].isin(x)][PRODUCT_NAME_COL].tolist()[:3]) + ("..." if len(x) > 3 else "")
    ).rename("cart_preview")

    users_df = users_df.merge(user_metrics, on="user_id", how="left").fillna(0)
    users_df = users_df.merge(cart_items, on="user_id", how="left").fillna("")
    
    users_df["avg_order_value"] = users_df.apply(
        lambda x: x["total_spend"] / x["purchase_count"] if x["purchase_count"] > 0 else 0, axis=1
    )

    return users_df, activity_df

users_df, activity_df = generate_meaningful_data()

# ======================================================================
# RECOMMENDER ENGINE
# ======================================================================
@st.cache_resource
def build_recommender(users_df, activity_df, products):
    if activity_df.empty:
        return None, None, None, {}

    ratings = activity_df.pivot_table(index="user_id", columns="product_id", values="cf_rating")
    user_mean = ratings.mean(axis=1)

    def pearson_sim(u, v):
        ru, rv = ratings.loc[u], ratings.loc[v]
        mask = (~ru.isna()) & (~rv.isna())
        if mask.sum() < 2: return 0.0
        ru_c = ru[mask] - user_mean[u]
        rv_c = rv[mask] - user_mean[v]
        num = (ru_c * rv_c).sum()
        den = (ru_c.pow(2).sum()**0.5) * (rv_c.pow(2).sum()**0.5)
        return num / den if den else 0.0

    def cf_predict(user_id, top_n=5):
        if user_id not in ratings.index: return []
        sims = {other: pearson_sim(user_id, other) for other in ratings.index if other != user_id}
        sims = {u: s for u, s in sims.items() if s > 0} # Only positive correlations
        if not sims: return []
        neighbors = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:10]
        
        preds = {}
        ru = ratings.loc[user_id]
        for item in ratings.columns:
            if not math.isnan(ru[item]): continue
            num, den = 0.0, 0.0
            for v, sim_val in neighbors:
                rv = ratings.loc[v, item]
                if math.isnan(rv): continue
                num += sim_val * (rv - user_mean[v])
                den += abs(sim_val)
            if den != 0: preds[item] = user_mean[user_id] + num / den
        return sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Content Based
    feat_base = products.copy()
    feat_base["Price"] = pd.to_numeric(feat_base["Price"], errors="coerce").fillna(0)
    feat_base["Category"] = feat_base["Category"].astype(str)
    feat_base["Scent Notes"] = feat_base["Scent Notes"].astype(str)
    
    feat = pd.concat([
        pd.get_dummies(feat_base["Category"], prefix="cat"),
        pd.get_dummies(feat_base["Scent Notes"], prefix="note"),
        ((feat_base["Price"] - feat_base["Price"].mean()) / feat_base["Price"].std()).rename("price_norm")
    ], axis=1).astype(float)
    feat.index = products[PRODUCT_ID_COL]

    def build_user_profile(user_id):
        liked = activity_df[(activity_df["user_id"] == user_id) & (activity_df["cf_rating"] >= 3)]
        liked = liked[liked["product_id"].isin(feat.index)]
        if liked.empty: return None
        weights = liked["cf_rating"].values.reshape(-1, 1)
        vectors = feat.loc[liked["product_id"]].values
        prof = (vectors * weights).sum(axis=0) / (weights.sum() + 1e-9)
        return prof

    def content_predict(user_id, top_n=5):
        prof = build_user_profile(user_id)
        if prof is None: return []
        denom = (norm(feat.values, axis=1) * norm(prof) + 1e-9)
        sims = (feat.values @ prof) / denom
        s = pd.Series(sims, index=feat.index)
        seen = activity_df[activity_df["user_id"] == user_id]["product_id"].unique().tolist()
        s = s.drop(index=seen, errors="ignore")
        return list(s.sort_values(ascending=False).head(top_n).items())

    def hybrid_predict(user_id, top_n=5):
        cf = dict(cf_predict(user_id, 20))
        cb = dict(content_predict(user_id, 20))
        all_keys = set(cf.keys()) | set(cb.keys())
        scores = {}
        for k in all_keys:
            s1 = cf.get(k, 0)
            s2 = cb.get(k, 0)
            scores[k] = 0.6 * s1 + 0.4 * s2
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return cf_predict, content_predict, hybrid_predict

cf_predict, content_predict, hybrid_predict = build_recommender(users_df, activity_df, products)

# ======================================================================
# SIDEBAR FILTERS
# ======================================================================
st.sidebar.subheader("🎯 Segment Filters")

age_range = st.sidebar.slider("Age Range", int(users_df.age.min()), int(users_df.age.max()), (18, 60))
income_range = st.sidebar.slider("Income Range (₹)", int(users_df.income.min()), int(users_df.income.max()), (15000, 300000), step=5000)

def filter_col(col):
    vals = sorted(users_df[col].unique())
    return st.sidebar.multiselect(col.replace("_", " ").title(), vals, default=vals)

f_gender = filter_col("gender")
f_region = filter_col("region")
f_marital = filter_col("marital_status")
f_occ = filter_col("occupation")
f_cat = filter_col("preferred_category")
f_scent = filter_col("preferred_scent")

mask = (
    users_df["age"].between(*age_range) &
    users_df["income"].between(*income_range) &
    users_df["gender"].isin(f_gender) &
    users_df["region"].isin(f_region) &
    users_df["marital_status"].isin(f_marital) &
    users_df["occupation"].isin(f_occ) &
    users_df["preferred_category"].isin(f_cat) &
    users_df["preferred_scent"].isin(f_scent)
)
seg_users = users_df[mask] if not users_df[mask].empty else users_df

st.sidebar.markdown("---")
st.sidebar.subheader("👤 User Selector")

if not seg_users.empty:
    selected_user = st.sidebar.selectbox("Select a User", sorted(seg_users["user_id"].unique()))
    u = users_df[users_df["user_id"] == selected_user].iloc[0]
    
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.5); padding: 15px; border-radius: 10px; margin-top: 10px;">
        <h4 style="margin:0; color: #0f766e;">{u['persona']}</h4>
        <p style="font-size: 0.9rem; margin: 5px 0;">
        <b>Age:</b> {u['age']} | <b>Gender:</b> {u['gender']}<br>
        <b>Region:</b> {u['region']}<br>
        <b>Income:</b> ₹{u['income']:,}
        </p>
        <hr style="margin: 10px 0;">
        <small><b>Prefers:</b> {u['preferred_category']} & {u['preferred_scent']}</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning("No users match the selected filters.")
    selected_user = None

# ======================================================================
# HELPER: PRODUCT DIALOG
# ======================================================================
@st.dialog("Product Details")
def show_product_details(pid):
    p = products[products[PRODUCT_ID_COL] == pid].iloc[0]
    col1, col2 = st.columns([1, 1.5])
    with col1:
        if pd.notna(p.get(IMAGE_URL_COL)):
            st.image(p[IMAGE_URL_COL], width="stretch")
    with col2:
        st.subheader(p[PRODUCT_NAME_COL])
        st.markdown(f"**Price:** ₹{p['Price']}")
        st.markdown(f"**Scent:** {p['Scent Notes']}")
        st.markdown(f"**Rating:** {'⭐'*int(p['Rating'])} ({p['Rating']})")
        
        # Description from "Short Marketing Description"
        desc = p.get("Short Marketing Description", "No description available.")
        if pd.isna(desc): desc = "No description available."
        st.markdown(f"_{desc}_")

# ======================================================================
# TABS
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Segments",
    "✨ User Recommendations",
    "📈 Engagement KPIs",
    "💼 Managerial Insights",
    "🛍️ Product Portfolio"
])

# ... (Existing Tab 1, 2, 3, 4 code remains unchanged, I will append Tab 5 at the end)

# ======================================================================
# PAGE 1: SEGMENTS
# ======================================================================
with tab1:
    st.markdown("### 🌍 Market Segmentation Analysis")
    st.markdown(f"**Showing {len(seg_users)} users** based on active filters.")
    
    if not seg_users.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### User Distribution by Persona")
            fig_pie = px.pie(seg_users, names="persona", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Income vs Age (by Persona)")
            fig_scat = px.scatter(seg_users, x="age", y="income", color="persona", size="income", 
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_scat.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scat, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Regional Preferences")
        region_cat = seg_users.groupby(["region", "preferred_category"]).size().reset_index(name="count")
        fig_bar = px.bar(region_cat, x="region", y="count", color="preferred_category", barmode="group",
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📋 Detailed User Data")
        
        # Display enriched table
        display_cols = [
            "user_id", "persona", "age", "region", "income", 
            "views", "add_to_cart", "purchase_count", "total_spend", "avg_order_value", "cart_preview"
        ]
        
        st.dataframe(
            seg_users[display_cols].style.format({
                "income": "₹{:,.0f}", 
                "total_spend": "₹{:,.0f}", 
                "avg_order_value": "₹{:,.0f}"
            }), 
            width="stretch"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Activity Analysis")
        
        g1, g2, g3 = st.columns(3)
        
        with g1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Activity by Persona")
            # Group by persona
            act_persona = seg_users.groupby("persona")[["views", "add_to_cart", "purchase_count"]].mean().reset_index()
            act_persona = act_persona.melt(id_vars="persona", var_name="Activity", value_name="Avg Count")
            
            fig_act = px.bar(act_persona, x="persona", y="Avg Count", color="Activity", barmode="group",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_act.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_act, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with g2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Spend Distribution")
            fig_hist = px.histogram(seg_users, x="total_spend", nbins=20, color_discrete_sequence=["#ef4444"])
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Total Spend (₹)")
            st.plotly_chart(fig_hist, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with g3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Purchase Frequency")
            freq_counts = seg_users["purchase_count"].value_counts().reset_index()
            freq_counts.columns = ["Purchases", "Count"]
            fig_freq = px.bar(freq_counts, x="Purchases", y="Count", color_discrete_sequence=["#0ea5e9"])
            fig_freq.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_freq, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### 🏆 Product Performance")
        
        # Aggregate product data
        prod_perf = activity_df.groupby("product_id").agg({
            "purchase_count": "sum",
            "price": "sum" # This is total revenue
        }).reset_index()
        
        prod_perf = prod_perf.merge(products[[PRODUCT_ID_COL, PRODUCT_NAME_COL, "Category"]], left_on="product_id", right_on=PRODUCT_ID_COL)
        prod_perf = prod_perf.rename(columns={"purchase_count": "Units Sold", "price": "Revenue"})
        
        p1, p2 = st.columns(2)
        
        with p1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Top 10 Best Sellers")
            top_sold = prod_perf.sort_values("Units Sold", ascending=False).head(10)
            fig_top = px.bar(top_sold, x="Units Sold", y=PRODUCT_NAME_COL, orientation='h', 
                             color="Units Sold", color_continuous_scale="Reds")
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_top, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with p2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Revenue Landscape (Treemap)")
            fig_tree = px.treemap(prod_perf, path=[px.Constant("All Products"), "Category", PRODUCT_NAME_COL], values='Revenue',
                                  color='Revenue', color_continuous_scale='RdBu')
            fig_tree.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_tree, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Category Sales Share")
        cat_share = prod_perf.groupby("Category")["Revenue"].sum().reset_index()
        fig_donut = px.pie(cat_share, names="Category", values="Revenue", hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
# ======================================================================
# PAGE 2: RECOMMENDATIONS
# ======================================================================
with tab2:
    if selected_user:
        st.markdown(f"### Personalized Recommendations")
        
        # User Stats Row (Matched to Third Image)
        u_acts = activity_df[activity_df["user_id"] == selected_user]
        total_spend = (u_acts["purchase_count"] * u_acts["product_id"].map(lambda x: products.set_index(PRODUCT_ID_COL).loc[x, "Price"])).sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Age", u["age"])
        c2.metric("Region", u["region"])
        c3.metric("Total Purchases", int(u_acts["purchase_count"].sum()))
        c4.metric("Total Spend", f"₹{total_spend:,.0f}")
        
        # Model Selection Logic
        purchase_count = int(u_acts["purchase_count"].sum())
        if purchase_count == 0:
            best_model = "Content-Based"
            reason = "User has no purchase history. Content-based filtering is used to recommend items similar to those viewed."
            why_not = "Collaborative Filtering requires purchase history to find similar users."
        elif purchase_count < 5:
            best_model = "Hybrid"
            reason = "User has limited history. Hybrid approach balances discovery (Content) with peer trends (CF)."
            why_not = "Pure CF might be too sparse, and pure Content might limit diversity."
        else:
            best_model = "Collaborative Filtering"
            reason = "User has rich history. CF effectively finds similar power users for high-precision recommendations."
            why_not = "Content-based might be too repetitive. CF leverages community patterns better."
            
        st.markdown(f"**Best Model for this user:** `{best_model}`")
        with st.expander("ℹ️ Why this model?"):
            st.markdown(f"**Reasoning**: {reason}")
            st.markdown(f"**Why not others?**: {why_not}")
        
        st.markdown("---")
        
        # 1. Top Recommendations (Matched to Second Image)
        st.markdown("### 🎁 You may also like these....")
        recs = hybrid_predict(selected_user, top_n=4)
        if recs:
            cols = st.columns(4)
            for i, (pid, score) in enumerate(recs):
                p = products[products[PRODUCT_ID_COL] == pid].iloc[0]
                with cols[i]:
                    st.markdown("<div class='glass-card' style='height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>", unsafe_allow_html=True)
                    
                    content_html = ""
                    if pd.notna(p.get(IMAGE_URL_COL)):
                        content_html += f"<img src='{p[IMAGE_URL_COL]}' class='rec-img'>"
                    
                    content_html += f"<div class='rec-title'>{p[PRODUCT_NAME_COL]}</div>"
                    
                    # Stars
                    rating = float(p.get("Rating", 4.0))
                    stars = "⭐" * int(round(rating))
                    content_html += f"<div style='color: #f59e0b; font-size: 0.9rem; margin-bottom: 4px;'>{stars} ({rating})</div>"
                    
                    content_html += f"<div class='rec-price'>₹{p['Price']}</div>"
                    content_html += f"<span class='rec-tag'>{p['Category']} • {p['Scent Notes']}</span>"
                    
                    st.markdown(content_html, unsafe_allow_html=True)
                    
                    # Full Width Blue Button
                    if st.button("View details", key=f"btn_rec_{pid}", width="stretch"):
                        show_product_details(pid)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. Because you bought X (Restored Split Layout)
        st.markdown("### 🛒 Because you bought...")
        
        purchased = u_acts[u_acts["purchase_count"] > 0]
        if not purchased.empty:
            last_pid = purchased.iloc[-1]["product_id"]
            last_p = products[products[PRODUCT_ID_COL] == last_pid].iloc[0]
            
            # Find related products (same category/scent)
            related = products[
                (products["Category"] == last_p["Category"]) & 
                (products[PRODUCT_ID_COL] != last_pid)
            ].head(3)
            
            col_main, col_recs = st.columns([1, 3])
            
            with col_main:
                st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
                if pd.notna(last_p.get(IMAGE_URL_COL)):
                    st.markdown(f"<img src='{last_p[IMAGE_URL_COL]}' style='width: 100%; border-radius: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                st.markdown(f"**{last_p[PRODUCT_NAME_COL]}**")
                st.caption("Your Last Purchase")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_recs:
                if not related.empty:
                    r_cols = st.columns(3)
                    for i, (_, rp) in enumerate(related.iterrows()):
                        with r_cols[i]:
                            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                            if pd.notna(rp.get(IMAGE_URL_COL)):
                                st.markdown(f"<img src='{rp[IMAGE_URL_COL]}' class='rec-img' style='height: 120px;'>", unsafe_allow_html=True)
                            st.markdown(f"<div class='rec-title' style='font-size: 0.9rem;'>{rp[PRODUCT_NAME_COL]}</div>", unsafe_allow_html=True)
                            
                            # Stars
                            rating = float(rp.get("Rating", 4.0))
                            stars = "⭐" * int(round(rating))
                            st.markdown(f"<div style='color: #f59e0b; font-size: 0.8rem; margin-bottom: 4px;'>{stars} ({rating})</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"<div class='rec-price' style='font-size: 1rem;'>₹{rp['Price']}</div>", unsafe_allow_html=True)
                            
                            if st.button("View details", key=f"btn_rel_{rp[PRODUCT_ID_COL]}", width="stretch"):
                                show_product_details(rp[PRODUCT_ID_COL])
                            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Make a purchase to see related recommendations!")

# ======================================================================
# PAGE 3: ENGAGEMENT
# ======================================================================
with tab3:
    st.markdown("### 📊 Engagement KPIs & Funnel")
    
    # Filter activity_df based on current segment
    seg_user_ids = seg_users["user_id"].unique()
    seg_activities = activity_df[activity_df["user_id"].isin(seg_user_ids)]
    
    # --- Aggregate Metrics ---
    st.markdown(f"#### Segment Overview ({len(seg_user_ids)} Users)")
    
    agg_views = seg_activities["views"].sum()
    agg_carts = seg_activities["add_to_cart"].sum()
    agg_purchases = seg_activities["purchase_count"].sum()
    agg_conversion = (agg_purchases / agg_views * 100) if agg_views > 0 else 0
    agg_avg_eng = seg_activities["engagement_score"].mean() if not seg_activities.empty else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Views", f"{agg_views:,}")
    c2.metric("Total Carts", f"{agg_carts:,}")
    c3.metric("Total Purchases", f"{agg_purchases:,}")
    c4.metric("Conversion Rate", f"{agg_conversion:.1f}%")
    
    st.markdown("---")
    
    # --- Individual vs Segment Comparison ---
    if selected_user:
        st.markdown(f"#### 👤 User vs. Segment: {selected_user}")
        
        u_acts = activity_df[activity_df["user_id"] == selected_user]
        u_views = u_acts["views"].sum()
        u_carts = u_acts["add_to_cart"].sum()
        u_purchases = u_acts["purchase_count"].sum()
        u_conversion = (u_purchases / u_views * 100) if u_views > 0 else 0
        u_eng = u_acts["engagement_score"].mean() if not u_acts.empty else 0
        
        col_u1, col_u2, col_u3, col_u4 = st.columns(4)
        
        col_u1.metric("User Views", u_views, delta=f"{u_views - (agg_views/len(seg_user_ids)):.1f} vs Avg")
        col_u2.metric("User Carts", u_carts, delta=f"{u_carts - (agg_carts/len(seg_user_ids)):.1f} vs Avg")
        col_u3.metric("User Purchases", int(u_purchases), delta=f"{u_purchases - (agg_purchases/len(seg_user_ids)):.1f} vs Avg")
        col_u4.metric("User Conv. Rate", f"{u_conversion:.1f}%", delta=f"{u_conversion - agg_conversion:.1f}%")
        
        st.markdown("---")

    # --- Funnel Visualization ---
    st.markdown("#### 📉 Engagement Funnel (Segment)")
    fig_funnel = px.funnel(
        dict(number=[agg_views, agg_carts, agg_purchases], stage=["Views", "Add to Cart", "Purchases"]),
        x='number', y='stage', color_discrete_sequence=["#ef4444"]
    )
    fig_funnel.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig_funnel, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================
# PAGE 4: MANAGERIAL INSIGHTS (PERSONALIZED)
# ======================================================================
with tab4:
    if selected_user:
        st.markdown(f"### 💼 CLV Analysis for {selected_user}")
        
        # Calculate Individual CLV
        user_rev = total_spend
        avg_order_val = user_rev / int(u_acts["purchase_count"].sum()) if int(u_acts["purchase_count"].sum()) > 0 else 0
        
        # Projected CLV based on Persona
        persona_mult = {
            "The Loyalist": 3.0, "The Big Spender": 2.5, 
            "The Window Shopper": 1.2, "The Newbie": 1.5, "The Deal Hunter": 1.8
        }
        mult = persona_mult.get(u["persona"], 1.5)
        projected_clv = user_rev * mult + 5000 # Base potential
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Value (Revenue)", f"₹{user_rev:,.0f}")
        c2.metric("Projected Lifetime Value", f"₹{projected_clv:,.0f}")
        c3.metric("Growth Potential", f"{(mult-1)*100:.0f}%")
        
        st.markdown("---")
        
        col_ins, col_act = st.columns([1, 2])
        
        with col_ins:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 📊 CLV Calculation Logic")
            
            st.markdown(f"""
            **Formula**:
            `CLV = (Current Revenue × Multiplier) + Base Potential`
            
            **User Values**:
            - Revenue: ₹{user_rev:,.0f}
            - Multiplier: {mult}x ({u['persona']})
            - Base Potential: ₹5,000
            
            **Projection**:
            ₹{user_rev:,.0f} × {mult} + ₹5,000 = **₹{projected_clv:,.0f}**
            """)
            
            # Segment Comparison
            # Calculate average revenue for this persona in the current segment
            persona_peers = seg_users[seg_users['persona'] == u['persona']]
            avg_peer_rev = persona_peers['total_spend'].mean() if not persona_peers.empty else 0
            seg_avg_clv = avg_peer_rev * mult + 5000
            
            diff = projected_clv - seg_avg_clv
            color = "#10b981" if diff >= 0 else "#ef4444" # Green or Red
            
            st.markdown("---")
            st.markdown(f"**Vs. {u['persona']} Average**:")
            st.markdown(f"Avg CLV: ₹{seg_avg_clv:,.0f}")
            st.markdown(f"Difference: <span style='color:{color}; font-weight:bold;'>{'▲' if diff>=0 else '▼'} ₹{abs(diff):,.0f}</span>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_act:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🚀 Recommended Action")
            if u["persona"] == "The Loyalist":
                st.success("🌟 **VIP Treatment**: Send a personalized 'Thank You' note and early access to new Floral collections.")
            elif u["persona"] == "The Window Shopper":
                st.warning("🏷️ **Conversion Push**: Send a limited-time 15% discount code for perfumes under ₹2000.")
            elif u["persona"] == "The Big Spender":
                st.info("💎 **Upsell**: Recommend the 'Luxury Gift Set' (₹5000+) as a complementary purchase.")
            else:
                st.info("📧 **Nurture**: Send a 'Welcome Guide' to fragrances to increase engagement.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            
            # Restore Insights Section
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🧠 User & Segment Profile Insights")
            st.markdown(f"""
            **Why this segment?**
            - **Spending Pattern**: This user fits the **{u['persona']}** profile due to their purchase frequency and average order value.
            - **Preferences**: Their activity shows a strong affinity for **{u['preferred_category']}** products, aligning with the segment's dominant preference.
            - **Engagement**: With **{int(u_acts['views'].sum())} views**, they are highly engaged, suggesting high potential for cross-selling **{u['preferred_scent']}** notes.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.info("Select a user to see personalized CLV insights.")

    # Segment Summary Section (Dynamic based on filters)
    st.markdown("---")
    st.markdown("### 🧩 Segment-Level Strategic Recommendations")
    st.markdown("Based on your current filters, here are strategies for the visible segments:")
    
    if not seg_users.empty:
        # Group by Persona to get stats
        seg_stats = seg_users.groupby("persona").agg({
            "income": "mean",
            "preferred_category": lambda x: x.mode()[0] if not x.mode().empty else "N/A"
        }).reset_index()
        
        # Strategy Map
        strategies = {
            "The Loyalist": "🏆 **Retention Focus**: These users are your backbone. Offer a loyalty program tier upgrade or early access to new launches.",
            "The Window Shopper": "🏷️ **Conversion Focus**: High interest, low purchase. Retarget with 'Price Drop' alerts or limited-time 10% off coupons.",
            "The Big Spender": "💎 **VIP Focus**: High value. Offer concierge services, exclusive bundles, or 'Private Reserve' collections.",
            "The Newbie": "🌱 **Onboarding Focus**: New to the brand. Send educational content (e.g., 'How to wear perfume') and a welcome discount.",
            "The Deal Hunter": "⚡ **Volume Focus**: Price sensitive. Promote 'Buy 2 Get 1 Free' offers or clearance sales to clear inventory."
        }
        
        cols = st.columns(len(seg_stats))
        for idx, row in seg_stats.iterrows():
            p_name = row['persona']
            with cols[idx % 4]: # Wrap if too many columns, though max is 5
                st.markdown(f"<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown(f"#### {p_name}")
                st.caption(f"Avg Income: ₹{row['income']:,.0f}")
                st.caption(f"Top Cat: {row['preferred_category']}")
                st.markdown(f"<div style='margin-top: 10px; font-size: 0.9rem;'>{strategies.get(p_name, 'General engagement strategy.')}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No segments visible with current filters.")

# ======================================================================
# PAGE 5: PRODUCT PORTFOLIO
# ======================================================================
with tab5:
    st.markdown("### 🛍️ Product Portfolio")
    
    # --- Top Bar: Collections (Scent Families) ---
    # We'll use Scent Notes as "Collections"
    all_scents = sorted(products["Scent Notes"].dropna().unique().tolist())
    # Add an "All" option
    collections = ["All"] + all_scents
    
    # Use columns for buttons to simulate a nav bar
    # We'll use session state to track the active collection
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = "All"
        
    st.markdown("##### 🌿 Collections")
    cols = st.columns(len(collections))
    for i, coll in enumerate(collections):
        if cols[i].button(coll, key=f"coll_btn_{i}", use_container_width=True):
            st.session_state.selected_collection = coll
            
    # --- Sidebar Filters for Portfolio ---
    # We use a separate expander or just add to sidebar if tab5 is active?
    # Since sidebar is global, we can add a section specific to Tab 5
    # But Streamlit sidebar is linear. We'll put filters in the main area (left col) or just use the global sidebar?
    # User asked for "side bar of filters". Let's put them in a collapsible sidebar section or just distinct area.
    # To avoid cluttering the main sidebar which has User Segmentation, let's use an expander in the main area or 
    # columns: [Filters (1) | Grid (3)]
    
    st.markdown("---")
    
    col_filters, col_grid = st.columns([1, 3])
    
    with col_filters:
        st.markdown("#### 🌪️ Filters")
        
        # Category
        all_cats = sorted(products["Category"].dropna().unique().tolist())
        sel_cats = st.multiselect("Category", all_cats, default=all_cats, key="port_cat")
        
        # Price
        min_p, max_p = int(products["Price"].min()), int(products["Price"].max())
        sel_price = st.slider("Price Range", min_p, max_p, (min_p, max_p), key="port_price")
        
        # Rating
        sel_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0, step=0.5, key="port_rating")
        
        # Quantity (Bottle Size)
        all_sizes = sorted(products["BottleML"].unique().tolist())
        sel_sizes = st.multiselect("Size (mL)", all_sizes, default=all_sizes, key="port_size")
        
    with col_grid:
        # Filter Logic
        filtered_prods = products[
            (products["Category"].isin(sel_cats)) &
            (products["Price"].between(sel_price[0], sel_price[1])) &
            (products["Rating"] >= sel_rating) &
            (products["BottleML"].isin(sel_sizes))
        ]
        
        # Apply Collection Filter (Top Bar)
        if st.session_state.selected_collection != "All":
            filtered_prods = filtered_prods[filtered_prods["Scent Notes"] == st.session_state.selected_collection]
            
        st.markdown(f"**Showing {len(filtered_prods)} products** in *{st.session_state.selected_collection}* collection")
        
        if not filtered_prods.empty:
            # Grid Layout
            n_cols = 3
            rows = [filtered_prods.iloc[i:i+n_cols] for i in range(0, len(filtered_prods), n_cols)]
            
            for row in rows:
                cols = st.columns(n_cols)
                for i, (_, p) in enumerate(row.iterrows()):
                    with cols[i]:
                        st.markdown("<div class='glass-card' style='height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>", unsafe_allow_html=True)
                        
                        if pd.notna(p.get(IMAGE_URL_COL)):
                            st.markdown(f"<img src='{p[IMAGE_URL_COL]}' class='rec-img' style='height: 150px;'>", unsafe_allow_html=True)
                            
                        st.markdown(f"<div class='rec-title' style='font-size: 0.9rem;'>{p[PRODUCT_NAME_COL]}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-weight:bold; color:#0f766e;'>₹{p['Price']}</div>", unsafe_allow_html=True)
                        st.caption(f"{p['Category']} | {p['BottleML']}ml")
                        st.markdown(f"⭐ {p['Rating']}")
                        
                        if st.button("View", key=f"btn_port_{p[PRODUCT_ID_COL]}", use_container_width=True):
                            show_product_details(p[PRODUCT_ID_COL])
                            
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No products match your filters.")
