
import streamlit as st
import pandas as pd
import glob
import os

# Define the path to the data directory
DATA_DIR = "data_TOI/data/"

def show_page():
    st.header("TOI Data Analysis")
    st.write("Select a CSV file to view its contents.")

    # Find all CSV files in the data directory
    try:
        csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))]
        if not csv_files:
            st.warning(f"No CSV files found in `{DATA_DIR}`. Please make sure your data is there.")
            return

        selected_file = st.selectbox("Choose a CSV file", csv_files)

        if selected_file:
            file_path = os.path.join(DATA_DIR, selected_file)
            st.write(f"### Preview of `{selected_file}`")
            
            try:
                df = pd.read_csv(file_path)
                st.data_editor(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {e}")

    except Exception as e:
        st.error(f"An error occurred while trying to find CSV files: {e}")

