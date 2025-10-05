
import streamlit as st
import pandas as pd
import glob
import os
import io

# Define the path to the data directory
DATA_DIR = "data_TOI/data/"

def show_page():
    st.header("1. Select Raw Data")

    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None

    # Option to clear the loaded data
    if st.session_state.raw_data is not None:
        if st.button("Clear Loaded Data"):
            st.session_state.raw_data = None
            st.rerun()

    if st.session_state.raw_data is None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Option A: Select an existing file")
            try:
                csv_files = [""] + [os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))]
                selected_file = st.selectbox("Choose a CSV file", csv_files)

                if selected_file:
                    file_path = os.path.join(DATA_DIR, selected_file)
                    try:
                        df = pd.read_csv(file_path, comment='#')
                        st.session_state.raw_data = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading file: {e}")

            except Exception as e:
                st.error(f"An error occurred while trying to find CSV files: {e}")

        with col2:
            st.subheader("Option B: Upload a new file")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    # To read file as string:
                    string_data = uploaded_file.getvalue().decode("utf-8")
                    # Can be used wherever a "file-like" object is accepted:
                    df = pd.read_csv(io.StringIO(string_data), comment='#')
                    st.session_state.raw_data = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    if st.session_state.raw_data is not None:
        st.success("Raw data loaded successfully.")
        st.write("### Data Preview")
        st.dataframe(st.session_state.raw_data.head())

