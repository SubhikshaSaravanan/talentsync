import streamlit as st

st.set_page_config(page_title="TalentSync Home", page_icon="🦊", layout="wide")

st.markdown(
    """
    <style>
    .big-font {
        font-size: 50px !important;
        text-align: center;
        color: #222222;
        font-weight: bold;
    }
    .small-font {
        font-size: 22px !important;
        text-align: center;
        color: #555555;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="center">', unsafe_allow_html=True)

# --- Main Caption ---
st.markdown('<p class="big-font">Build Your Dream Team, No Cap </p>', unsafe_allow_html=True)

# --- Optional smaller tagline (if you want) ---
st.markdown('<p class="small-font">TalentSync makes recruiting smarter, faster, cooler. </p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="center">', unsafe_allow_html=True)

st.image("assets/welcome_mascot.png", width=400)

st.markdown('<p class="big-font">Welcome to TalentSync!</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Find the best candidates with AI-powered matching and explanations.</p>', unsafe_allow_html=True)

if st.button("Start Recruiting"):
    st.switch_page("pages/Recruit.py")

st.markdown('</div>', unsafe_allow_html=True)
