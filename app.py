import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import requests
import io
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


lang_dict = {
    "en": {
        "app_title": "📈 CSV Analyzer",
        "upload_csv": "Upload CSV File(s)",
        "load_csv_url": "Paste direct CSV URL:",
        "load_csv_btn": "Load CSV from URL",
        "files_uploaded": "✅ Files uploaded",
        "select_csv": "Select CSV(s) to view",
        "no_file_selected": "Please select at least one file.",
        "no_files": "📂 Upload or load a CSV file to get started.",
        "missing_values": "🧼 Missing Values",
        "handle_missing": "How to handle missing data?",
        "drop_rows": "Drop Rows",
        "fill_mean": "Fill with Mean",
        "fill_median": "Fill with Median",
        "tabs": ["📄 Overview", "📊 Charts", "📈 Predict", "⬇️ Download"],
        "dataset_overview": "📄 Dataset Overview",
        "rows_cols": "**Rows:** {rows}  |  **Columns:** {cols}",
        "missing_col": "**Missing values per column:**",
        "simple_visualizations": "📊 Simple Visualizations",
        "x_axis": "X-Axis",
        "y_axis": "Y-Axis",
        "group_by": "Group by (Optional)",
        "scatter_plot": "📍 Scatter Plot",
        "correlation_heatmap": "📉 Correlation Heatmap",
        "not_enough_data_scatter": "Not enough data points to plot scatter chart.",
        "not_enough_numeric_heatmap": "Not enough numeric columns to display correlation heatmap.",
        "predict_header": "🧠 Predict with Simple Models",
        "pick_features": "Pick Input Columns (X)",
        "pick_target": "Pick Target Column (Y)",
        "choose_model": "Choose Model",
        "linear": "Linear",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "test_set_size": "Test Set Size (%)",
        "train_model": "🚀 Train Model",
        "target_in_features_err": "Target column cannot also be a feature.",
        "not_enough_train_data": "Not enough data to train. Minimum 5 rows required.",
        "model_trained": "✅ Model Trained Successfully",
        "r2_score": "R² Score",
        "rmse": "RMSE",
        "pred_vs_actual": "📌 Predictions vs Actual",
        "download_cleaned": "⬇️ Download Cleaned Data",
        "download_csv": "📥 Download CSV",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language": "Language",
        "english": "English",
        "hindi": "Hindi",
        "large_dataset_info": "Dataset too large ({n} rows). Using a random sample of {max_n} rows for plotting."
    },
    "hi": {
        "app_title": "📈 CSV विश्लेषक",
        "upload_csv": "CSV फ़ाइल(ओं) अपलोड करें",
        "load_csv_url": "सीधे CSV URL चिपकाएँ:",
        "load_csv_btn": "URL से CSV लोड करें",
        "files_uploaded": "✅ फ़ाइलें अपलोड हुईं",
        "select_csv": "दृश्यमान CSV चुनें",
        "no_file_selected": "कम से कम एक फ़ाइल चुनें।",
        "no_files": "📂 शुरू करने के लिए CSV फ़ाइल अपलोड करें या लोड करें।",
        "missing_values": "🧼 गायब मान",
        "handle_missing": "गायब डेटा कैसे संभालें?",
        "drop_rows": "पंक्तियाँ हटाएं",
        "fill_mean": "औसत से भरें",
        "fill_median": "माध्यिका से भरें",
        "tabs": ["📄 अवलोकन", "📊 चार्ट", "📈 भविष्यवाणी", "⬇️ डाउनलोड"],
        "dataset_overview": "📄 डेटा सेट अवलोकन",
        "rows_cols": "**पंक्तियाँ:** {rows}  |  **स्तंभ:** {cols}",
        "missing_col": "**स्तंभों में गायब मान:**",
        "simple_visualizations": "📊 सरल दृश्य",
        "x_axis": "X-अक्ष",
        "y_axis": "Y-अक्ष",
        "group_by": "समूह बनाएँ (वैकल्पिक)",
        "scatter_plot": "📍 स्कैटर प्लॉट",
        "correlation_heatmap": "📉 सहसंबंध हीटमैप",
        "not_enough_data_scatter": "स्कैटर चार्ट के लिए पर्याप्त डेटा नहीं है।",
        "not_enough_numeric_heatmap": "सहसंबंध हीटमैप दिखाने के लिए पर्याप्त संख्यात्मक स्तंभ नहीं हैं।",
        "predict_header": "🧠 सरल मॉडल के साथ भविष्यवाणी करें",
        "pick_features": "इनपुट स्तंभ चुनें (X)",
        "pick_target": "लक्ष्य स्तंभ चुनें (Y)",
        "choose_model": "मॉडल चुनें",
        "linear": "लिनियर",
        "ridge": "रिज",
        "lasso": "लासो",
        "test_set_size": "टेस्ट सेट आकार (%)",
        "train_model": "🚀 मॉडल प्रशिक्षित करें",
        "target_in_features_err": "लक्ष्य स्तंभ फीचर भी नहीं हो सकता।",
        "not_enough_train_data": "प्रशिक्षण के लिए पर्याप्त डेटा नहीं है। न्यूनतम 5 पंक्तियाँ आवश्यक हैं।",
        "model_trained": "✅ मॉडल सफलतापूर्वक प्रशिक्षित हुआ",
        "r2_score": "R² स्कोर",
        "rmse": "RMSE",
        "pred_vs_actual": "📌 पूर्वानुमान बनाम वास्तविक",
        "download_cleaned": "⬇️ साफ़ डेटा डाउनलोड करें",
        "download_csv": "📥 CSV डाउनलोड करें",
        "theme": "थीम",
        "light": "लाइट",
        "dark": "डार्क",
        "language": "भाषा",
        "english": "अंग्रेज़ी",
        "hindi": "हिन्दी",
        "large_dataset_info": "डेटासेट बहुत बड़ा है ({n} पंक्तियाँ)। प्लॉट के लिए {max_n} पंक्तियाँ यादृच्छिक रूप से चुनी गई हैं।"
    }
}

def _(key):
    lang = st.session_state.get("lang", "en")
    return lang_dict.get(lang, lang_dict["en"]).get(key, key)



def get_font_sizes(row_count):
    if row_count < 500:
        return 18, 14, 12, 10
    elif row_count < 2000:
        return 14, 11, 9, 7
    elif row_count < 10000:
        return 10, 8, 6, 5
    else:
        return 8, 6, 5, 4



st.set_page_config(page_title=" CSV Analyzer", layout="wide")

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

with st.sidebar:
    st.header(_("app_title"))
    lang_choice = st.selectbox(_("language"), ["English", "Hindi"], index=0 if st.session_state["lang"]=="en" else 1)
    st.session_state["lang"] = "en" if lang_choice == "English" else "hi"
    theme_choice = st.radio(_("theme"), [_("light"), _("dark")], index=0 if st.session_state["theme"]=="light" else 1)
    st.session_state["theme"] = "light" if theme_choice == _("light") else "dark"


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_files(files):
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.name)
        if not os.path.exists(path):
            with open(path, "wb") as out:
                out.write(f.getbuffer())

@st.cache_data(show_spinner=False)
def load_and_merge(paths):
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        if "Year" in df.columns:
            df["Year"] = pd.to_datetime(df["Year"].astype(str), errors="coerce", format="%Y")
        return df
    else:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_csv_from_url(url):
    try:
        if not url.lower().startswith("http"):
            raise ValueError(_("load_csv_url"))
        res = requests.get(url)
        res.raise_for_status()
        return pd.read_csv(io.StringIO(res.text))
    except Exception as e:
        st.error(f"❌ Failed to load CSV from URL: {e}")
        return None

with st.sidebar.expander(_("upload_csv")):
    uploaded = st.file_uploader("", type="csv", accept_multiple_files=True)
    if uploaded:
        save_files(uploaded)
        st.success(_("files_uploaded"))

with st.sidebar.expander("🌍 Public CSV URL"):
    dataset_url = st.text_input(_("load_csv_url"))
    if st.button(_("load_csv_btn")) and dataset_url:
        df_remote = load_csv_from_url(dataset_url)
        if df_remote is not None:
            remote_name = dataset_url.split("/")[-1] or "remote_file.csv"
            df_remote.to_csv(os.path.join(UPLOAD_DIR, remote_name), index=False)
            st.success(f"✅ Loaded and saved: {remote_name}")
            st.experimental_rerun()

csv_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
if not csv_files:
    st.info(_("no_files"))
    st.stop()

selected = st.sidebar.multiselect(_("select_csv"), csv_files, default=csv_files)
if not selected:
    st.warning(_("no_file_selected"))
    st.stop()

df = load_and_merge([os.path.join(UPLOAD_DIR, f) for f in selected])
if df.empty:
    st.warning("Data is empty or invalid.")
    st.stop()


st.sidebar.subheader(_("missing_values"))
missing_strategy = st.sidebar.radio(_("handle_missing"), [_("drop_rows"), _("fill_mean"), _("fill_median")])
if missing_strategy == _("drop_rows"):
    df = df.dropna()
elif missing_strategy == _("fill_mean"):
    df = df.fillna(df.mean(numeric_only=True))
elif missing_strategy == _("fill_median"):
    df = df.fillna(df.median(numeric_only=True))


if st.session_state["theme"] == "dark":
    plt.style.use("dark_background")
    plotly_template = "plotly_dark"
    bg_color = "#0E1117"
else:
    plt.style.use("default")
    plotly_template = "plotly_white"
    bg_color = "white"


tab1, tab2, tab3, tab4 = st.tabs(_("tabs"))


with tab1:
    st.header(_("dataset_overview"))
    st.dataframe(df.head(100), use_container_width=True)
    st.markdown(_(
        "rows_cols").format(rows=df.shape[0], cols=df.shape[1]))
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        st.markdown(_("missing_col"))
        st.dataframe(missing_counts)


with tab2:
    st.header(_("simple_visualizations"))
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    MAX_SAMPLE_SIZE = 1000
    if df.shape[0] > MAX_SAMPLE_SIZE:
        st.info(_(
            "large_dataset_info").format(n=df.shape[0], max_n=MAX_SAMPLE_SIZE))
        df_sample = df.sample(n=MAX_SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df

    title_fs, axislabel_fs, ticklabel_fs, annot_fs = get_font_sizes(df_sample.shape[0])

    if len(num_cols) >= 2:
        x = st.selectbox(_("x_axis"), num_cols, key="xaxis")
        y = st.selectbox(_("y_axis"), [col for col in num_cols if col != x], key="yaxis")
        color = st.selectbox(_("group_by"), ["None"] + cat_cols, key="color")

        if df_sample[[x, y]].dropna().shape[0] >= 2:
            st.subheader(_("scatter_plot"))
            fig = px.scatter(
                df_sample,
                x=x,
                y=y,
                color=None if color == "None" else df_sample[color],
                opacity=0.6,
                size_max=7,
                title=f"Scatter plot of {y} vs {x}",
                template=plotly_template
            )
            fig.update_layout(
                title_font_size=title_fs,
                xaxis_title_font_size=axislabel_fs,
                yaxis_title_font_size=axislabel_fs,
                xaxis_tickfont_size=ticklabel_fs,
                yaxis_tickfont_size=ticklabel_fs,
                plot_bgcolor=bg_color,
                paper_bgcolor=bg_color,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(_("not_enough_data_scatter"))

    st.subheader(_("correlation_heatmap"))
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        top_corr_cols = corr_matrix.sum().sort_values(ascending=False).head(10).index.tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df[top_corr_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax,
            square=True,
            linewidths=0.5,
            annot_kws={"size": annot_fs}
        )
        ax.set_title(_("correlation_heatmap"), fontsize=title_fs)
        ax.xaxis.label.set_size(axislabel_fs)
        ax.yaxis.label.set_size(axislabel_fs)
        ax.tick_params(axis='x', labelsize=ticklabel_fs)
        ax.tick_params(axis='y', labelsize=ticklabel_fs)
        fig.patch.set_facecolor(bg_color)
        st.pyplot(fig)
    else:
        st.warning(_("not_enough_numeric_heatmap"))


with tab3:
    st.header(_("predict_header"))
    if len(num_cols) >= 2:
        features = st.multiselect(_("pick_features"), num_cols[:-1])
        target = st.selectbox(_("pick_target"), num_cols)
        model_choice = st.radio(_("choose_model"), [_("linear"), _("ridge"), _("lasso")])
        size = st.slider(_("test_set_size"), 10, 40, 20)

        if st.button(_("train_model")):
            if target in features:
                st.error(_("target_in_features_err"))
            else:
                data = df[features + [target]].dropna()
                if data.shape[0] < 5:
                    st.warning(_("not_enough_train_data"))
                else:
                    X = data[features]
                    y = data[target]

                    
                    try:
                        X = np.array(X)
                        y = np.array(y)
                    except Exception as e:
                        st.error(f"Error converting data: {e}")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size/100, random_state=0)

                    if model_choice == _("linear"):
                        model = LinearRegression()
                    elif model_choice == _("ridge"):
                        model = Ridge()
                    else:
                        model = Lasso()

                    try:
                        model.fit(X_train, y_train)
                    except Exception as e:
                        st.error(f"Error training model: {e}")
                        st.stop()

                    pred = model.predict(X_test)

                    st.success(_("model_trained"))
                    st.metric(_("r2_score"), f"{r2_score(y_test, pred):.3f}")
                    st.metric(_("rmse"), f"{np.sqrt(mean_squared_error(y_test, pred)):.3f}")

                    st.subheader(_("pred_vs_actual"))
                    fig = px.scatter(x=y_test, y=pred,
                                     labels={"x": "Actual", "y": "Predicted"},
                                     title="Prediction Results",
                                     template=plotly_template)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data with at least 2 numeric columns to use this feature.")


with tab4:
    st.header(_("download_cleaned"))
    st.download_button(_("download_csv"), df.to_csv(index=False).encode(), file_name="cleaned_data.csv")


st.markdown("---")
st.markdown("Made beginner-friendly with ❤️ using Streamlit")
