
import streamlit as st
import pandas as pd
import glob
import os
import io
import json
from app.processing import split_data, create_binary_dataset, train_xgboost_cv, run_inference, generate_shap_summary_plot
import plotly.express as px

DATA_DIR = "data_TOI/data/"
HEADER_INFO_FILE = "data_TOI/data/data7.txt"
SCORER_CONF_FILE = "data_TOI/scorer_conf.json"

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def initialize_session_state():
    st.session_state.setdefault('raw_data', None)
    st.session_state.setdefault('header_info', "")
    st.session_state.setdefault('train_data', None)
    st.session_state.setdefault('test_data', None)
    st.session_state.setdefault('selected_features', None)
    st.session_state.setdefault('filtered_train_data', None)
    st.session_state.setdefault('filtered_test_data', None)
    st.session_state.setdefault('binary_train_data', None)
    st.session_state.setdefault('model_pack', None)
    st.session_state.setdefault('inference_results', None)
    st.session_state.setdefault('shap_plot', None)

def ui_step_1_select_data():
    st.subheader("1. Select Raw Data")
    if st.session_state.raw_data is not None and st.button("Clear Loaded Data"):
        for key in st.session_state.keys():
            st.session_state[key] = None
        st.rerun()

    if st.session_state.raw_data is None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Option A: Select an existing file")
            try:
                csv_files = [""] + sorted([os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))])
                selected_file = st.selectbox("Choose a CSV file", csv_files)
                if selected_file:
                    file_path = os.path.join(DATA_DIR, selected_file)
                    df = pd.read_csv(file_path, comment='#')
                    st.session_state.raw_data = df
                    st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")
        with col2:
            st.markdown("#### Option B: Upload a new file")
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, comment='#')
                st.session_state.raw_data = df
                st.rerun()
        return False
    
st.success("Raw data loaded successfully.")
st.dataframe(st.session_state.raw_data.head())
return True

def ui_step_2_header_info():
    st.subheader("2. Provide Header Info")
    try:
        with open(HEADER_INFO_FILE, 'r') as f:
            default_headers = "".join([line for line in f.readlines() if line.startswith("# COLUMN")])
    except FileNotFoundError:
        default_headers = "Header information file not found."

    current_headers = "\n".join([f"# COLUMN {col}:" for col in st.session_state.raw_data.columns])
    header_options = ["Extracted from data", "Default from data7.txt"]
    selected_option = st.radio("Choose header source:", header_options, horizontal=True)
    
    header_text_value = current_headers if selected_option == "Extracted from data" else default_headers
    header_text = st.text_area("Header Information", value=header_text_value, height=200)
    
    if st.button("Save Header Info"):
        st.session_state.header_info = header_text
        st.success("Header information saved.")
    
    if st.session_state.header_info:
        with st.expander("Show Saved Header Information"):
            st.text(st.session_state.header_info)

def ui_step_3_split_data():
    st.subheader("3. Split Data to Train and Test")
    if st.session_state.train_data is not None and st.button("Clear Split Data"):
        st.session_state.train_data = None
        st.session_state.test_data = None
        st.rerun()

    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.number_input("Random seed", value=42)

    if st.button("Split Data"):
        try:
            train_df, test_df = split_data(st.session_state.raw_data, 1 - test_size, random_seed)
            st.session_state.train_data = train_df
            st.session_state.test_data = test_df
            st.success("Data split successfully!")
        except ValueError as e:
            st.error(f"Error: {e}. Make sure a 'tid' column exists.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    if st.session_state.train_data is not None:
        st.write("Train Set:", st.session_state.train_data.shape)
        st.dataframe(st.session_state.train_data.head(3))
        st.download_button("Download Train CSV", convert_df_to_csv(st.session_state.train_data), "train.csv", "text/csv")
        
        st.write("Test Set:", st.session_state.test_data.shape)
        st.dataframe(st.session_state.test_data.head(3))
        st.download_button("Download Test CSV", convert_df_to_csv(st.session_state.test_data), "test.csv", "text/csv")
        return True
    return False

def ui_step_4_visualize_data():
    with st.expander("4. Column Visualization", expanded=False):
        plot_df = st.session_state.train_data
        columns = plot_df.columns.tolist()
        tab1, tab2, tab3 = st.tabs(["1D Distribution", "2D Scatter", "2D Scatter with Target"])
        with tab1:
            col = st.selectbox("Select column", columns, key="vis_1d")
            if col:
                fig = px.histogram(plot_df, x=col, title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            x_col = st.selectbox("X-axis", columns, key="vis_2d_x")
            y_col = st.selectbox("Y-axis", columns, index=1 if len(columns)>1 else 0, key="vis_2d_y")
            if x_col and y_col:
                fig = px.scatter(plot_df.sample(min(1000, len(plot_df))), x=x_col, y=y_col, title=f'{x_col} vs. {y_col}')
                st.plotly_chart(fig, use_container_width=True)
        with tab3:
            x_col_t = st.selectbox("X-axis", columns, key="vis_3d_x")
            y_col_t = st.selectbox("Y-axis", columns, index=1 if len(columns)>1 else 0, key="vis_3d_y")
            symbol_col = st.selectbox("Target/Symbol", columns, key="vis_3d_s")
            if x_col_t and y_col_t and symbol_col:
                fig = px.scatter(plot_df.sample(min(1000, len(plot_df))), x=x_col_t, y=y_col_t, symbol=symbol_col, title=f'{x_col_t} vs. {y_col_t} by {symbol_col}')
                st.plotly_chart(fig, use_container_width=True)

def ui_step_5_select_features():
    st.subheader("5. Column Selection")
    all_columns = st.session_state.train_data.columns.tolist()
    default_features = ["pl_orbper","pl_trandurh","pl_trandep","st_teff","st_logg","st_rad","st_tmag","st_dist"]
    valid_defaults = [f for f in default_features if f in all_columns]

    selected = st.multiselect("Select feature columns", all_columns, default=valid_defaults)
    if st.button("Save Selected Features"):
        st.session_state.selected_features = selected
        st.success(f"Saved {len(selected)} features.")
    
    if st.session_state.selected_features:
        st.write("Selected features:", st.session_state.selected_features)
        return True
    return False

def ui_step_6_filter_csv():
    st.subheader("6. Create Filtered CSV")
    if st.session_state.filtered_train_data is not None and st.button("Clear Filtered Data"):
        st.session_state.filtered_train_data = None
        st.session_state.filtered_test_data = None
        st.rerun()

    all_columns = st.session_state.train_data.columns.tolist()
    default_keys = ['toi', 'tid', 'tfopwg_disp']
    valid_defaults = [k for k in default_keys if k in all_columns]
    
    key_cols = st.multiselect("Select key/target columns to keep", all_columns, default=valid_defaults)

    if st.button("Create Filtered Data"):
        final_cols = list(set(key_cols + st.session_state.selected_features))
        
        missing_cols = [col for col in final_cols if col not in all_columns]
        if missing_cols:
            st.error(f"The following columns are not in the data: {', '.join(missing_cols)}")
        else:
            st.session_state.filtered_train_data = st.session_state.train_data[final_cols]
            st.session_state.filtered_test_data = st.session_state.test_data[final_cols]
            st.success("Filtered datasets created.")

    if st.session_state.filtered_train_data is not None:
        st.write("Filtered Train Set:", st.session_state.filtered_train_data.shape)
        st.dataframe(st.session_state.filtered_train_data.head(3))
        st.download_button("Download Filtered Train CSV", convert_df_to_csv(st.session_state.filtered_train_data), "train_filtered.csv", "text/csv")
        
        st.write("Filtered Test Set:", st.session_state.filtered_test_data.shape)
        st.dataframe(st.session_state.filtered_test_data.head(3))
        st.download_button("Download Filtered Test CSV", convert_df_to_csv(st.session_state.filtered_test_data), "test_filtered.csv", "text/csv")
        return True
    return False

def ui_step_7_create_binary_set():
    st.subheader("7. Create Binary Training Set")
    if st.session_state.binary_train_data is not None and st.button("Clear Binary Data"):
        st.session_state.binary_train_data = None
        st.rerun()

    try:
        with open(SCORER_CONF_FILE, 'r') as f:
            scorer_conf = json.load(f)
        default_target = scorer_conf.get("target_column", "")
        default_mapping = scorer_conf.get("mapping", {})
    except Exception:
        default_target = "tfopwg_disp"
        default_mapping = {"PC": "dontcare", "KP": "positive", "CP": "positive", "FP": "negative"}

    all_columns = st.session_state.filtered_train_data.columns.tolist()
    target_col_index = all_columns.index(default_target) if default_target in all_columns else 0
    target_column = st.selectbox("Select the target column", all_columns, index=target_col_index)
    
    mapping_str = st.text_area("Define target mapping", value=json.dumps(default_mapping, indent=2), height=200)

    if st.button("Create Binary Dataset"):
        try:
            mapping = json.loads(mapping_str)
            binary_df = create_binary_dataset(st.session_state.filtered_train_data, target_column, mapping)
            st.session_state.binary_train_data = binary_df
            st.success("Binary training dataset created successfully.")
        except Exception as e:
            st.error(f"Error creating binary dataset: {e}")

    if st.session_state.binary_train_data is not None:
        st.write("Binary Train Set Preview:", st.session_state.binary_train_data.shape)
        st.dataframe(st.session_state.binary_train_data.head())
        st.download_button("Download Binary Train CSV", convert_df_to_csv(st.session_state.binary_train_data), "train_binary.csv", "text/csv")
        return True
    return False

def ui_step_8_run_xgboost():
    st.subheader("8. Run XGBoost with Cross-Validation")
    if st.session_state.model_pack is not None and st.button("Clear Model"):
        st.session_state.model_pack = None
        st.rerun()

    st.markdown("#### XGBoost Hyperparameters")
    with st.form("xgb_params"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("n_estimators", 100, 1000, 300, 50)
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
            max_depth = st.slider("max_depth", 3, 10, 4, 1)
        with col2:
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1)
            colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1)
            min_child_weight = st.slider("min_child_weight", 1, 10, 5, 1)
        with col3:
            reg_lambda = st.slider("reg_lambda (L2)", 0.0, 10.0, 5.0, 0.5)
            reg_alpha = st.slider("reg_alpha (L1)", 0.0, 10.0, 0.0, 0.5)
        
        submitted = st.form_submit_button("Run Training")

    if submitted:
        with st.spinner("Training XGBoost model..."):
            try:
                model, metrics_df, medians = train_xgboost_cv(
                    df=st.session_state.binary_train_data,
                    features=st.session_state.selected_features,
                    target_col='target_binary',
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha
                )
                st.session_state.model_pack = {'model': model, 'metrics': metrics_df, 'medians': medians}
                st.success("Model training complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.session_state.model_pack is not None:
        st.markdown("#### Cross-Validation Metrics")
        st.dataframe(st.session_state.model_pack['metrics'])
        return True
    return False

def ui_step_9_run_inference():
    st.subheader("9. Run Inference on Test Set")
    if st.session_state.inference_results is not None and st.button("Clear Inference Results"):
        st.session_state.inference_results = None
        st.rerun()

    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("Run Inference"):
        with st.spinner("Running inference..."):
            try:
                results_df = run_inference(
                    model=st.session_state.model_pack['model'],
                    df=st.session_state.filtered_test_data,
                    features=st.session_state.selected_features,
                    medians=st.session_state.model_pack['medians'],
                    threshold=threshold
                )
                st.session_state.inference_results = results_df
                st.success("Inference complete.")
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

    if st.session_state.inference_results is not None:
        st.write("Inference Results Preview:", st.session_state.inference_results.shape)
        st.dataframe(st.session_state.inference_results.head())
        st.download_button("Download Inference Results", convert_df_to_csv(st.session_state.inference_results), "inference_results.csv")
        return True
    return False

def ui_step_10_show_shap():
    st.subheader("10. SHAP Analysis")
    if st.session_state.get('shap_plot') is not None and st.button("Clear SHAP Plot"):
        st.session_state.shap_plot = None
        st.rerun()
    
    if st.button("Generate SHAP Summary Plot"):
        with st.spinner("Calculating SHAP values..."):
            try:
                fig = generate_shap_summary_plot(
                    model=st.session_state.model_pack['model'],
                    df=st.session_state.binary_train_data,
                    features=st.session_state.selected_features,
                    medians=st.session_state.model_pack['medians']
                )
                st.session_state.shap_plot = fig
                st.success("SHAP plot generated.")
            except Exception as e:
                st.error(f"An error occurred during SHAP analysis: {e}")

    if st.session_state.shap_plot is not None:
        st.pyplot(st.session_state.shap_plot)

def show_page():
    st.header("Exoplanet Analysis Workflow")
    initialize_session_state()

    if not ui_step_1_select_data(): return
    st.markdown("---")
    ui_step_2_header_info()
    st.markdown("---")
    if not ui_step_3_split_data(): return
    st.markdown("---")
    ui_step_4_visualize_data()
    st.markdown("---")
    if not ui_step_5_select_features(): return
    st.markdown("---")
    if not ui_step_6_filter_csv(): return
    st.markdown("---")
    if not ui_step_7_create_binary_set(): return
    st.markdown("---")
    if not ui_step_8_run_xgboost(): return
    st.markdown("---")
    if not ui_step_9_run_inference(): return
    st.markdown("---")
    ui_step_10_show_shap()
