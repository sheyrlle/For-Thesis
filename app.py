import streamlit as st
import joblib
import re

# load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('C:\\Users\\Sherylle Rose\\Desktop\\sentiment_app_package\\random_forest_model.pkl')
        vectorizer = joblib.load('C:\\Users\\Sherylle Rose\\Desktop\\sentiment_app_package\\vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, vectorizer = load_model()

# improved function to detect non-English characters
def contains_non_english(text):
    # match any character outside a-z, A-Z, 0-9, space, basic punctuation
    allowed_chars = re.compile(r'^[a-zA-Z0-9\s.,!?\'"\-]*$')
    return not bool(allowed_chars.match(text))

st.title("Sentiment Analysis of CCIT Students' Confidence Level in Programming Languages")
st.write("This app predicts whether a student's comment reflects high or low confidence in programming.")

if model and vectorizer:
    user_input = st.text_area("Enter a student comment:")

    if st.button("Analyze Confidence"):
        if not user_input.strip():
            st.warning("!!!Please enter a comment before predicting.")
        elif contains_non_english(user_input):
            st.warning("Warning: Input contains characters outside English letters and basic punctuation. Please use English for best results.")
        else:
            try:
                def clean_text(text):
                    text = text.lower()
                    text = re.sub(r"http\S+|www.\S+", "", text)
                    text = re.sub(r"\d+", "", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text

                cleaned = clean_text(user_input)

                if cleaned == "":
                    st.warning("Input is empty after cleaning. Please enter more descriptive text.")
                else:
                    st.write(f"**Cleaned input:** {cleaned}")  

                    vectorized_input = vectorizer.transform([cleaned])

                    prediction = model.predict(vectorized_input)[0]
                    confidence_label = "HIGH CONFIDENCE" if prediction == 1 else "LOW CONFIDENCE"

                    st.success(f"Sentiment Classification: **{confidence_label}**")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
