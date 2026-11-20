from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.optimization import (
    prepare_overlap_recommendations,
    prepare_split_candidates,
    build_split_points,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
GTFS = DATA / "gtfs"

st.set_page_config(
    page_title="Route Ranker | Transit Intelligence",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(180deg,#f5f8fc 0%,#f8fafc 45%,#f3f6fb 100%);}
.block-container {padding-top:1.4rem;padding-bottom:2.5rem;max-width:1500px;}
[data-testid="stMetric"] {background:#ffffff;border:1px solid #e5eaf2;padding:16px 18px;border-radius:16px;box-shadow:0 4px 18px rgba(20,40,80,.045);}
[data-testid="stMetricLabel"] {color:#64748b;font-weight:600;}
[data-testid="stMetricValue"] {color:#10233f;}
section[data-testid="stSidebar"] {background:#101f36;}
section[data-testid="stSidebar"] * {color:#edf4ff;}
div[data-testid="stTabs"] button {font-weight:700;}
.hero {background:linear-gradient(115deg,#10233f 0%,#173d63 55%,#146b73 100%);padding:28px 32px;border-radius:22px;color:white;margin-bottom:20px;box-shadow:0 12px 35px rgba(16,35,63,.16);}
.hero h1 {font-size:2.05rem;line-height:1.15;margin:0 0 8px 0;color:white;}
.hero p {font-size:1rem;color:#d8e7f4;margin:0;max-width:850px;}
.section-kicker {color:#16828a;font-weight:800;letter-spacing:.09em;text-transform:uppercase;font-size:.72rem;margin-bottom:4px;}
.note {background:#eef6fb;border-left:4px solid #16828a;padding:12px 15px;border-radius:8px;color:#26445b;}
.warn {background:#fff7e8;border-left:4px solid #e5a12b;padding:12px 15px;border-radius:8px;color:#684813;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        return None
    return pd.read_csv(path)

def require_data():
    names = ["route_features.csv","overlap_df.csv","route_stop_map.csv","route_hourly_usage.csv","routes_preprocessed.csv"]
    missing = [n for n in names if not (DATA/n).exists()]
    if missing:
        st.error("Missing required data files in the data folder: " + ", ".join(missing))
        st.stop()

require_data()
features = load_csv("route_features.csv")
overlap_raw = load_csv("overlap_df.csv")
stop_map = load_csv("route_stop_map.csv")
hourly = load_csv("route_hourly_usage.csv")
Xdf = load_csv("routes_preprocessed.csv")

if features is None or overlap_raw is None or stop_map is None or hourly is None or Xdf is None:
    st.error("One or more required CSV files could not be loaded.")
    st.stop()

# Validate the row alignment expected by the original clustering notebook.
if len(features) != len(Xdf):
    st.error(f"Route feature rows ({len(features):,}) and preprocessed rows ({len(Xdf):,}) do not match. Check notebook export alignment.")
    st.stop()

st.sidebar.markdown("## 🚌 ROUTE RANKER")
st.sidebar.caption("Transit network intelligence")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview","Network Analytics","Route Optimization","Model Insights"],
    label_visibility="visible"
)
st.sidebar.markdown("---")
st.sidebar.caption("Processed GTFS datasets • Route-level analysis")
st.sidebar.markdown("<div style='font-size:.75rem;color:#b8c8dc'>Built for operational exploration, not automatic service-change approval.</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="font-size:.75rem;font-weight:800;letter-spacing:.14em;color:#8ce0d8;margin-bottom:9px">PUBLIC TRANSPORT • DATA INTELLIGENCE</div>
  <h1>Route Ranker</h1>
  <p>Explore network structure, discover overlapping corridors, identify long-route split candidates and compare machine-learning segments in one place.</p>
</div>
""", unsafe_allow_html=True)

# Common clean-up for optimization output.
merge_df = prepare_overlap_recommendations(overlap_raw, features)
split_candidates, length_threshold, stop_threshold = prepare_split_candidates(features)

if page == "Overview":
    st.markdown('<div class="section-kicker">Network snapshot</div>', unsafe_allow_html=True)
    st.subheader("Your transit network at a glance")
    a,b,c,d = st.columns(4)
    a.metric("Routes analysed", f"{features['route_id'].nunique():,}")
    b.metric("Unique stops", f"{stop_map['stop_id'].nunique():,}")
    c.metric("Route-stop links", f"{len(stop_map):,}")
    d.metric("Route pairs assessed", f"{len(merge_df):,}")

    high = int((merge_df["overlap_percentage"] >= 70).sum())
    moderate = int(((merge_df["overlap_percentage"] >= 40) & (merge_df["overlap_percentage"] < 70)).sum())
    split_count = len(split_candidates)
    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Route length distribution")
        fig = px.histogram(features, x="avg_route_length_km", nbins=32,
                           color_discrete_sequence=["#16828a"])
        fig.update_layout(xaxis_title="Average route length (km)",yaxis_title="Number of routes",
                          plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Optimization signals")
        summary = pd.DataFrame({
            "Signal":["High overlap pairs","Moderate overlap pairs","Long-route split candidates"],
            "Count":[high,moderate,split_count]
        })
        fig = px.bar(summary,x="Count",y="Signal",orientation="h",text="Count",
                     color="Signal",color_discrete_sequence=["#e18b43","#3b82a0","#16828a"])
        fig.update_layout(showlegend=False,plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=10,t=15,b=5))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="note">Interpretation: scheduled stop records and route features are operational indicators. They are not passenger counts, and overlap alone does not prove that routes should be merged.</div>',unsafe_allow_html=True)

elif page == "Network Analytics":
    st.markdown('<div class="section-kicker">Explore the data</div>', unsafe_allow_html=True)
    st.subheader("Route characteristics")
    numeric = features.select_dtypes(include="number").columns.tolist()
    default_x = numeric.index("avg_route_length_km") if "avg_route_length_km" in numeric else 0
    default_y = numeric.index("total_trips") if "total_trips" in numeric else min(1,len(numeric)-1)
    xcol,ycol = st.columns(2)
    x = xcol.selectbox("Horizontal feature",numeric,index=default_x)
    y = ycol.selectbox("Vertical feature",numeric,index=default_y)
    color_by = st.selectbox("Group routes by",["agency_id","route_type","None"])
    fig = px.scatter(features,x=x,y=y,color=None if color_by=="None" else color_by,
                     hover_data=["route_id"],opacity=.72,
                     color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Hourly scheduled activity")
    route_ids = sorted(hourly["route_id"].astype(str).unique().tolist())
    chosen = st.selectbox("Select a route",route_ids)
    h = hourly[hourly["route_id"].astype(str)==chosen].sort_values("hour")
    fig = px.line(h,x="hour",y="total_stop_records",markers=True,
                  color_discrete_sequence=["#16828a"])
    fig.update_layout(xaxis_title="Hour of day",yaxis_title="Scheduled stop records",
                      plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Dataset browser")
    st.dataframe(features.head(100),use_container_width=True,hide_index=True)

elif page == "Route Optimization":
    st.markdown('<div class="section-kicker">Operational decision support</div>', unsafe_allow_html=True)
    st.subheader("Route optimization workspace")
    merge_tab,split_tab = st.tabs(["↔ Route merging analysis","✂ Route splitting analysis"])

    with merge_tab:
        st.markdown("#### Identify route pairs with substantial shared-stop overlap")
        st.markdown('<div class="note">Notebook rules: High overlap ≥ 70%; Moderate overlap ≥ 40% and &lt; 70%; Low overlap &lt; 40%.</div>',unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("High overlap pairs",f"{(merge_df['overlap_category']=='High').sum():,}")
        m2.metric("Moderate overlap pairs",f"{(merge_df['overlap_category']=='Moderate').sum():,}")
        m3.metric("All route pairs",f"{len(merge_df):,}")

        threshold = st.slider("Minimum overlap percentage",0,100,70,step=5)
        categories = st.multiselect("Show recommendation categories",
                                    ["High","Moderate","Low"],default=["High","Moderate"])
        filtered = merge_df[(merge_df["overlap_percentage"]>=threshold) &
                            (merge_df["overlap_category"].isin(categories))].copy()
        top_n = st.slider("Maximum route pairs to display",10,100,30,step=10)
        filtered = filtered.sort_values(["overlap_percentage","shared_stops"],ascending=False).head(top_n)
        fig = px.histogram(merge_df,x="overlap_percentage",nbins=30,
                           color_discrete_sequence=["#16828a"])
        fig.add_vline(x=40,line_dash="dash",line_color="#e5a12b",annotation_text="Moderate 40%")
        fig.add_vline(x=70,line_dash="dash",line_color="#d65f59",annotation_text="High 70%")
        fig.update_layout(xaxis_title="Overlap percentage",yaxis_title="Route-pair count",
                          plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=35,b=10))
        st.plotly_chart(fig,use_container_width=True)

        display_cols = [c for c in ["route_id_1","route_id_2","shared_stops","overlap_percentage",
                                    "route_1_length_km","route_2_length_km","route_1_total_trips",
                                    "route_2_total_trips","recommendation","reason"] if c in filtered.columns]
        st.dataframe(filtered[display_cols],use_container_width=True,hide_index=True)
        st.download_button("Download route merging recommendations",
                           filtered.to_csv(index=False).encode("utf-8"),
                           "route_merging_recommendations.csv","text/csv")
        st.caption("A high-overlap pair is an investigation candidate, not an automatic merge decision. Check service patterns, directionality, passenger demand and operational constraints before changing routes.")

    with split_tab:
        st.markdown("#### Identify unusually long routes for split investigation")
        s1,s2,s3 = st.columns(3)
        s1.metric("90th percentile length",f"{length_threshold:.2f} km")
        s2.metric("Median unique stops",f"{stop_threshold:.0f}")
        s3.metric("Split candidates",f"{len(split_candidates):,}")
        st.markdown('<div class="note">Notebook rule: select routes at or above the 90th percentile of average route length, then retain those whose unique-stop count is at least the network median.</div>',unsafe_allow_html=True)

        if not split_candidates.empty:
            fig = px.scatter(split_candidates,x="avg_route_length_km",y="total_unique_stops",
                             size="total_trips",hover_name="route_id",
                             color_discrete_sequence=["#16828a"])
            fig.add_vline(x=length_threshold,line_dash="dash",line_color="#d65f59",
                          annotation_text="90th percentile")
            fig.add_hline(y=stop_threshold,line_dash="dash",line_color="#e5a12b",
                          annotation_text="Median stops")
            fig.update_layout(xaxis_title="Average route length (km)",yaxis_title="Unique stops",
                              plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig,use_container_width=True)
            split_display = split_candidates[["route_id","avg_route_length_km","total_unique_stops","total_trips"]].copy()
            st.dataframe(split_display,use_container_width=True,hide_index=True)
            st.download_button("Download route split candidates",
                               split_display.to_csv(index=False).encode("utf-8"),
                               "route_split_candidates.csv","text/csv")
        else:
            st.info("No routes met both route-splitting screening conditions.")

        trips_path = GTFS / "trips.txt"
        stop_times_path = GTFS / "stop_times.txt"
        if trips_path.exists() and stop_times_path.exists() and not split_candidates.empty:
            st.success("Optional GTFS trips.txt and stop_times.txt detected. Stop-level midpoint analysis is available.")
            with st.spinner("Building representative-trip split points..."):
                split_points = build_split_points(split_candidates,trips_path,stop_times_path)
            if not split_points.empty:
                st.dataframe(split_points,use_container_width=True,hide_index=True)
                st.download_button("Download candidate split points",
                                   split_points.to_csv(index=False).encode("utf-8"),
                                   "route_split_point_suggestions.csv","text/csv")
            else:
                st.warning("No representative trip had enough stop-time records to create a midpoint split.")
        else:
            st.markdown('<div class="warn"><b>Stop-level split points are not yet available.</b><br>The optimization notebook needs the original GTFS <code>trips.txt</code> and <code>stop_times.txt</code> to select each candidate route’s representative trip and calculate its midpoint stop. The processed route-stop map does not contain stop sequence order. If you want this additional analysis, place those two files in <code>data/gtfs/</code>.</div>',unsafe_allow_html=True)

elif page == "Model Insights":
    st.markdown('<div class="section-kicker">Machine learning</div>', unsafe_allow_html=True)
    st.subheader("Route segmentation and classification")
    model_tab1,model_tab2,model_tab3 = st.tabs(["K-Means","DBSCAN","XGBoost"])
    X = Xdf.to_numpy(dtype=float)
    with model_tab1:
        k = st.slider("K-Means cluster count",2,10,4,key="kmeans_k")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        with st.spinner("Fitting K-Means..."):
            km = KMeans(n_clusters=k,random_state=42,n_init=10)
            labels = km.fit_predict(X)
        result = features[["route_id","avg_route_length_km","total_trips","total_unique_stops"]].copy()
        result["cluster"] = labels.astype(str)
        counts = result["cluster"].value_counts().rename_axis("Cluster").reset_index(name="Routes")
        st.metric("Inertia (WCSS)",f"{km.inertia_:,.2f}")
        st.plotly_chart(px.bar(counts,x="Cluster",y="Routes",color="Cluster",
                               color_discrete_sequence=px.colors.qualitative.Safe),use_container_width=True)
        if len(set(labels)) > 1:
            q1,q2,q3 = st.columns(3)
            q1.metric("Silhouette",f"{silhouette_score(X,labels):.4f}")
            q2.metric("Davies-Bouldin",f"{davies_bouldin_score(X,labels):.4f}")
            q3.metric("Calinski-Harabasz",f"{calinski_harabasz_score(X,labels):.2f}")
        st.dataframe(result,use_container_width=True,hide_index=True)
        st.download_button("Download K-Means assignments",result.to_csv(index=False).encode("utf-8"),
                           "kmeans_route_assignments.csv","text/csv")
    with model_tab2:
        eps = st.slider("DBSCAN epsilon",0.1,3.0,1.0,0.1,key="db_eps")
        min_samples = st.slider("Minimum samples",2,20,5,key="db_min")
        from sklearn.cluster import DBSCAN
        with st.spinner("Fitting DBSCAN..."):
            db = DBSCAN(eps=eps,min_samples=min_samples)
            labels = db.fit_predict(X)
        noise = int((labels==-1).sum())
        st.columns(3)[0].metric("Clusters excluding noise",len(set(labels)-{-1}))
        st.columns(3)[1].metric("Noise routes",f"{noise:,}")
        st.columns(3)[2].metric("Noise share",f"{noise/len(labels):.1%}")
        result = features[["route_id","avg_route_length_km","total_trips","total_unique_stops"]].copy()
        result["cluster"] = labels
        fig = px.scatter(result,x="avg_route_length_km",y="total_trips",
                         color=result["cluster"].astype(str),hover_name="route_id")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("DBSCAN noise means a route was not assigned to a dense cluster for the selected parameters. It does not mean the route is invalid or poor-performing.")
        st.download_button("Download DBSCAN assignments",result.to_csv(index=False).encode("utf-8"),
                           "dbscan_route_assignments.csv","text/csv")
    with model_tab3:
        st.write("XGBoost predicts Short, Medium and Long route-length categories using the feature engineering in your model notebook.")
        st.info("Average route length defines the target and is excluded from predictors to avoid target leakage.")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
        from xgboost import XGBClassifier
        df = features.replace([np.inf,-np.inf],np.nan).dropna(subset=["avg_route_length_km"]).copy()
        numeric = ["total_trips","total_scheduled_stop_records","active_hours","avg_hourly_stop_records",
                   "total_unique_stops","overlapping_routes","total_shared_stops"]
        categorical = [c for c in ["agency_id","route_type"] if c in df.columns]
        lower = df["avg_route_length_km"].quantile(.33)
        upper = df["avg_route_length_km"].quantile(.66)
        df["target"] = np.where(df["avg_route_length_km"]<=lower,0,
                         np.where(df["avg_route_length_km"]<=upper,1,2))
        X_model = pd.get_dummies(df[numeric+categorical],columns=categorical)
        X_model = X_model.apply(pd.to_numeric,errors="coerce")
        X_model = X_model.fillna(X_model.median(numeric_only=True)).dropna(axis=1,how="all")
        Xtr,Xte,ytr,yte = train_test_split(X_model,df["target"],test_size=.20,random_state=42,stratify=df["target"])
        with st.spinner("Training XGBoost..."):
            xgb = XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.05,subsample=.8,
                                colsample_bytree=.8,objective="multi:softprob",num_class=3,
                                eval_metric="mlogloss",random_state=42,n_jobs=-1)
            xgb.fit(Xtr,ytr)
            pred = xgb.predict(Xte)
        st.metric("Validation accuracy",f"{accuracy_score(yte,pred):.4f}")
        names=["Short","Medium","Long"]
        report=classification_report(yte,pred,labels=[0,1,2],target_names=names,output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)
        cm= pd.DataFrame(confusion_matrix(yte,pred,labels=[0,1,2]),index=names,columns=names)
        st.plotly_chart(px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),
                                  color_continuous_scale="Tealgrn"),use_container_width=True)

st.markdown("---")
st.caption("Route Ranker • Route-level analytics and optimization decision support")
