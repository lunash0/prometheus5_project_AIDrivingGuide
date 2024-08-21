import streamlit as st
import os
import sys
from home import page_home
from feedback import page_feedback

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

st.set_page_config(
    page_title="AI Driving Guide Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

sidebar_bg_img = """
<style>
[data-testid="stSidebar"] > div {
    background-image: url("https://static.vecteezy.com/system/resources/previews/021/430/833/non_2x/abstract-colorful-dark-blue-and-purple-gradient-blurred-background-night-sky-gradient-blue-gradation-wallpaper-for-background-themes-abstract-background-in-purple-and-blue-tones-web-design-banner-vector.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
}
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0); /* Make sidebar background transparent */
}
</style>
"""

caption_css = """
<style>
    .caption {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: #FFFFFF;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
</style>
"""
button_animation_css = """
<style>
.stButton > button {
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #4CAF50;
    color: white;
}
</style>
"""
st.markdown(sidebar_bg_img, unsafe_allow_html=True)
st.markdown(caption_css, unsafe_allow_html=True)
st.markdown(button_animation_css, unsafe_allow_html=True)

st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select a Page",
    ["🏠 Home", "📢 Feedback"],
    index=0
)

if page == "🏠 Home":
    page_home()
elif page == "📢 Feedback":
    page_feedback()

st.markdown('<div class="caption">Prometheus</div>', unsafe_allow_html=True)



