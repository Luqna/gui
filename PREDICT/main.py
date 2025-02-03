import streamlit as st
from pages.home import show_home_page
from pages.prediction import prediction_page
from utils.styles import CUSTOM_STYLES

def main():
    st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_home_page()
    else:
        prediction_page()

if __name__ == '__main__':
    main()