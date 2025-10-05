
import streamlit as st
import pandas as pd
import glob
import os
import io

# Define the path to the data directory
DATA_DIR = "data_TOI/data/"
HEADER_INFO_FILE = "data_TOI/data/data7.txt"

def get_default_header():
    try:
        with open(HEADER_INFO_FILE, 'r') as f:
            lines = f.readlines()
        header_lines = [line for line in lines if line.startswith("# COLUMN")]
        return "".join(header_lines)
    except FileNotFoundError:
        return "Header information file not found."

def show_page():
    st.header("Exoplanet Analysis Workflow")

    # --- 1. Select Raw Data ---
    st.subheader("1. Select Raw Data")

    # Initialize session state
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'header_info' not in st.session_state:
        st.session_state.header_info = ""

    # Option to clear the loaded data
    if st.session_state.raw_data is not None:
        if st.button("Clear Loaded Data"):
            st.session_state.raw_data = None
            st.session_state.header_info = ""
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
                    try:
                        df = pd.read_csv(file_path, comment='#')
                        st.session_state.raw_data = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            except Exception as e:
                st.error(f"An error occurred while trying to find CSV files: {e}")

        with col2:
            st.markdown("#### Option B: Upload a new file")
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, comment='#')
                    st.session_state.raw_data = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    if st.session_state.raw_data is not None:
        st.success("Raw data loaded successfully.")
        st.write("##### Data Preview")
        st.dataframe(st.session_state.raw_data.head())

        # --- 2. Provide HEADER info ---
        st.subheader("2. Provide Header Info")
        
        default_headers = get_default_header()
        
        # Automatically get headers from the dataframe if they exist
        current_headers = "\n".join([f"# COLUMN {col}:" for col in st.session_state.raw_data.columns])

        header_options = ["Extracted from data", "Default from data7.txt"]
        selected_header_option = st.radio("Choose header source:", header_options)

        if selected_header_option == "Extracted from data":
            header_text = st.text_area("Header Information", value=current_headers, height=250)
        else:
            header_text = st.text_area("Header Information", value=default_headers, height=250)

        if st.button("Save Header Info"):
            st.session_state.header_info = header_text
            st.success("Header information saved.")

        if st.session_state.header_info:
            st.write("##### Saved Header Information Preview")
            st.text(st.session_state.header_info[:500] + "...")

