import streamlit as st

from DemoML import page_DemoML
from DemoNN import page_DemoNN
from DescripNN import page_DescripNN
from DescriptML import page_DescriptML

st.set_page_config(page_title="IS Model", page_icon="🔭")

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

pages = {
    "Home": "Home",
    "Demo_ML": page_DemoML,
    "Demo_NN": page_DemoNN,
    "Description ML": page_DescriptML,
    "Description NN": page_DescripNN
}

st.sidebar.title("Model")
selection = st.sidebar.radio("", list(pages.keys()), index=list(pages.keys()).index(st.session_state.selected_page))

if selection != st.session_state.selected_page:
    st.session_state.selected_page = selection
    st.rerun()

def page_Home():
    st.title("Model Application")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Demo ML", expanded=True):
            st.markdown("""### Demo ML Model  
            This example demonstrates the use of machine learning models to predict house prices using algorithms such as Random Forest and Gradient Boosting.""")
            if st.button("View", key="demo_ml"):
                st.session_state.selected_page = "Demo_ML"
                st.rerun()

    with col2:
        with st.expander("Demo NN", expanded=True):
            st.markdown("""### Demo NN Model  
            This example demonstrates the use of a neural network model to predict the impact of economic changes.""")
            if st.button("View", key="demo_nn"):
                st.session_state.selected_page = "Demo_NN"
                st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        with st.expander("Description ML", expanded=True):
            st.markdown("""### Description of ML Model  
            An explanation of the Machine Learning model, from data preprocessing to how the model works in predicting house prices using various factors.""")
            if st.button("View", key="desc_ml"):
                st.session_state.selected_page = "Description ML"
                st.rerun()

    with col4:
        with st.expander("Description NN", expanded=True):
            st.markdown("""### Description of NN Model  
            An explanation of the Neural Network model, from data preprocessing to the model's working process in predicting the impact of economic changes using various factors such as interest rates, GDP growth, inflation, unemployment rates, and monetary policy announcements.""")
            if st.button("View", key="desc_nn"):
                st.session_state.selected_page = "Description NN"
                st.rerun()

if st.session_state.selected_page == "Home":
    page_Home()
else:
    pages[st.session_state.selected_page]()
