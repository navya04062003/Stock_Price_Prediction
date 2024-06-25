import streamlit as st


st.title("Welcome to the Stock Price Prediction Models")
st.write("""
Welcome to the Stock Price Prediction Models app. Navigate through the pages to see the model details and descriptions.
""")
# Online image URL
image_url = "https://cdn.pixabay.com/photo/2023/07/28/08/06/finance-8154775_640.jpg"
st.image(image_url, caption="Stock Prediction Models", use_column_width=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Model Information", "Model Description"])

if page == "Home":
    st.write("""
    ## Home
    Welcome to the Stock Price Prediction Models app. Use the sidebar to navigate to different pages.
    """)
elif page == "Model Information":
    st.write("## Model Information")
    exec(open("pages/table.py").read())
elif page == "Model Description":
    st.write("## Model Description")
    exec(open("pages/description.py").read())
