from langchain_community.llms import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

# Load the model
model = load_model('mld_new.keras')

# Class labels
diseases = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die back",
            "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]

# Load environment variables
load_dotenv()

# Initialize GPT model
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

# Streamlit app
st.title(" MangoLeafMedic")

uploaded_file = st.file_uploader("Upload a mango leaf image", type=["jpg", "png", "jpeg", "gif"])

bt = st.button("Recommend Medicine")

if bt:
    if uploaded_file:
        # Read and decode image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display uploaded image
        st.image(img_rgb, caption='Uploaded Leaf Image', use_column_width=True)

        # Resize to 224x224 for the model (use RGB)
        resized_img = cv2.resize(img_rgb, (224, 224))  # Resize
        normalized_img = resized_img / 255.0            # Normalize
        test_image = normalized_img.reshape((1, 224, 224, 3))  # Add batch dimension

        # Predict using the CNN model
        prediction = model.predict(test_image)
        index_value = np.argmax(prediction, axis=1)[0]
        diseases_name = diseases[index_value]

        # Show result
        st.markdown('## Leaf Disease Name and Cure')
        st.write("🦠 **Detected Disease**:", diseases_name)

        # Query GPT for cure suggestion
        messages = [
            SystemMessage("I am giving you the Mango leaf disease name. You have to give its cure and name of the medicine. Note: If leaf is healthy, just say the leaf is healthy."),
            HumanMessage(diseases_name)
        ]
        answer = llm.invoke(messages)
        st.write( answer)
