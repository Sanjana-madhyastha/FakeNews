import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Custom Background
def set_bg():
    page_bg = """
    <style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    .stTextArea textarea {
        background: #f5f5f5;
        color: #333;
    }
    .stButton>button {
        background: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background: #ff1e1e;
    }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Set background
set_bg()

# Title & Description
st.title("📰 Fake News Detector")
st.write("🔍 **Check if a news article is real or fake using AI**")

# News Input
news_input = st.text_area("✍️ Paste the news article here:", "", height=150)

# Predict Button
if st.button("🚀 Check Credibility"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text before checking.")
    else:
        # Vectorize input
        text_vectorized = vectorizer.transform([news_input])
        
        # Prediction
        prediction = model.predict(text_vectorized)[0]

        # Display Result
        if prediction == 1:
            st.error("🚨 **Fake News Detected!** ❌")
        else:
            st.success("✅ **This news appears to be real.**")

# Sidebar
st.sidebar.header("📌 About")
st.sidebar.info("🧠 **FakeDetect** is an AI-powered app that detects fake news using Machine Learning and NLP.")

# Footer
st.markdown("---")
st.markdown("💡 **Developed by Your Team | Powered by AI & NLP**")
