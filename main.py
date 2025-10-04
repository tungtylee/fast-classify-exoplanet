
import streamlit as st
from app import toi_page

st.set_page_config(
    page_title="Exoplanet Classification App",
    layout="wide"
)

# A simple dictionary to map page names to their corresponding functions
PAGES = {
    "TOI Data Analysis": toi_page.show_page,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
