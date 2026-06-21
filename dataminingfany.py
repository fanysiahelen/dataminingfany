import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="K-Means Clustering APM", layout="wide")

st.title("Klasterisasi Provinsi Berdasarkan APM Pendidikan")

dataset = st.file_uploader("Upload Dataset Excel", type=["xlsx"])
geojson = st.file_uploader("Upload GeoJSON Indonesia", type=["geojson"])

if dataset:

    df = pd.read_excel(dataset)

    df.rename(columns={
        "38 Provinsi":"Provinsi",
        "SD/sederajat":"APM_SD",
        "SMP/sederajat":"APM_SMP",
        "SMA/sederajat":"APM_SMA"
    }, inplace=True)

    st.header("Data Understanding")
    st.dataframe(df.head())
    st.dataframe(df.describe())

    st.header("Data Preparation")

    st.write("Jumlah Duplikat:", df.duplicated().sum())
    st.dataframe(df.isnull().sum().to_frame("Missing Value"))

    X = df[["APM_SD","APM_SMP","APM_SMA"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    st.subheader("Data Setelah Scaling")
    st.dataframe(scaled_df.head())

    fig, axes = plt.subplots(1,3,figsize=(15,4))

    for i,col in enumerate(scaled_df.columns):
        sns.histplot(scaled_df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)

    st.pyplot(fig)

    st.header("Modeling")

    inertia = []

    for k in range(1,11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig2, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1,11), inertia, marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("Inertia")

    st.pyplot(fig2)

    k = st.slider("Jumlah Cluster",2,10,3)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X_scaled)

    st.subheader("Hasil Klasterisasi")
    st.dataframe(df)

    st.header("Analisis Cluster")

    cluster_means = df.groupby("Cluster")[["APM_SD","APM_SMP","APM_SMA"]].mean()
    st.dataframe(cluster_means)

    cluster_counts = df["Cluster"].value_counts().sort_index()
    st.dataframe(cluster_counts)

    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df,x="APM_SD",y="APM_SMP",hue="Cluster",ax=ax3)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df,x="APM_SMP",y="APM_SMA",hue="Cluster",ax=ax4)
    st.pyplot(fig4)

    for c in sorted(df["Cluster"].unique()):
        st.subheader(f"Cluster {c}")
        st.write(df[df["Cluster"]==c]["Provinsi"].tolist())
        
    # Menentukan kategori cluster berdasarkan rata-rata APM

cluster_means["Rata_Rata_APM"] = (
    cluster_means["APM_SD"] +
    cluster_means["APM_SMP"] +
    cluster_means["APM_SMA"]
) / 3

ranking = cluster_means["Rata_Rata_APM"].sort_values()

cluster_label = {}

cluster_label[ranking.index[0]] = "Rendah"
cluster_label[ranking.index[1]] = "Sedang"
cluster_label[ranking.index[2]] = "Tinggi"

st.subheader("Interpretasi Cluster")

for cluster in sorted(cluster_means.index):

    kategori = cluster_label[cluster]

    st.markdown(f"### Cluster {cluster}")

    st.write(
        f"Kategori APM : **{kategori}**"
    )

    st.write(
        f"Rata-rata APM SD : {cluster_means.loc[cluster,'APM_SD']:.2f}"
    )

    st.write(
        f"Rata-rata APM SMP : {cluster_means.loc[cluster,'APM_SMP']:.2f}"
    )

    st.write(
        f"Rata-rata APM SMA : {cluster_means.loc[cluster,'APM_SMA']:.2f}"
    )

    if kategori == "Tinggi":
        st.success(
            "Provinsi dalam cluster ini memiliki tingkat partisipasi pendidikan yang tinggi pada hampir semua jenjang pendidikan."
        )

    elif kategori == "Sedang":
        st.info(
            "Provinsi dalam cluster ini memiliki tingkat partisipasi pendidikan yang cukup baik, namun masih terdapat ruang untuk peningkatan."
        )

    else:
        st.error(
            "Provinsi dalam cluster ini memiliki tingkat partisipasi pendidikan yang relatif rendah dan memerlukan perhatian lebih dalam kebijakan pendidikan."
        )

    if geojson:

        def clean_name(x):
            x = str(x).upper().strip()
            x = x.replace("DAERAH KHUSUS IBUKOTA","DKI")
            x = x.replace("DAERAH ISTIMEWA","DI")
            x = x.replace("KEPULAUAN","KEP.")
            return x

        gdf = gpd.read_file(geojson)

        df["Provinsi_clean"] = df["Provinsi"].apply(clean_name)
        gdf["Provinsi_clean"] = gdf["PROVINSI"].apply(clean_name)

        merged = gdf.merge(
            df,
            on="Provinsi_clean",
            how="left"
        )

        st.header("Peta Cluster")

        fig_map = px.choropleth_mapbox(
            merged,
            geojson=merged.geometry,
            locations=merged.index,
            color="Cluster",
            hover_name="Provinsi",
            mapbox_style="carto-positron",
            zoom=3,
            center={"lat":-2.5,"lon":118},
            opacity=0.7
        )

        st.plotly_chart(fig_map, use_container_width=True)

    st.header("Evaluation")

    score = silhouette_score(X_scaled, df["Cluster"])
    st.metric("Silhouette Score", f"{score:.4f}")

else:
    st.info("Silakan upload dataset terlebih dahulu.")
