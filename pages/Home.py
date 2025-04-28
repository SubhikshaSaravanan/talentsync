import streamlit as st

st.set_page_config(page_title="TalentSync Home", page_icon="🦊", layout="wide")

st.markdown(
    """
    <style>
    /* Apply Poppins font and bright white color */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #ffffff !important;
        font-size: 18px;
    }

    /* Big white title */
    h1 {
        font-size: 60px !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }

    /* Subheader */
    h2, h3 {
        font-size: 30px !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }

    /* Center mascot image */
    .stImage {
        display: flex;
        justify-content: center;
    }

    /* Text inside input boxes (if any later) */
    input, textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="center">', unsafe_allow_html=True)

# Main Caption 
st.markdown('<p class="big-font">Build Your Dream Team, No Cap </p>', unsafe_allow_html=True)

# tagline
st.markdown('<p class="small-font">TalentSync makes recruiting smarter, faster, cooler. </p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="center">', unsafe_allow_html=True)

st.image("assets/welcome_mascot.png", width=400)

st.markdown('<p class="big-font">Welcome to TalentSync!</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Find the best candidates with AI-powered matching and explanations.</p>', unsafe_allow_html=True)

if st.button("Start Recruiting"):
    st.switch_page("pages/Recruit.py")

st.markdown('</div>', unsafe_allow_html=True)
