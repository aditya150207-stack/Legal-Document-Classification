
import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Legalised ML",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Legalised ML")
st.subheader("Legal Document Classification System")

st.write(
    "Enter a legal document or paragraph below "
    "to predict its legal category."
)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    document = st.text_area(
        "Enter Legal Document Text",
        height=200,
        placeholder="Type or paste your legal document here..."
    )

    if st.button("Classify Document"):
        if document.strip():
            transformed_text = vectorizer.transform([document])
            prediction = model.predict(transformed_text)

            st.success(f"Predicted Category: {prediction[0]}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_text)[0]
                confidence = max(probabilities) * 100
                st.info(f"Model confidence: {confidence:.2f}%")
        else:
            st.warning("Please enter some document text.")
else:
    st.warning(
        "Model files are not available. "
        "Run train_model.py first to train the model."
    )

st.divider()
st.caption(
    "Academic ML project. Predictions are for educational "
    "purposes and are not legal advice."
)
