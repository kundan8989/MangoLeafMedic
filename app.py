from langchain_community.llms import OpenAI
from langchain_core.messages import SystemMessage,HumanMessage
import os
from dotenv import load_dotenv
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import tensorflow

# load the model from the  file
model = load_model('mld_new.keras')

diseases=["Anthracnose","Bacterial Canker","Cutting Weevil","Die back",",Gall Midge","Healthy","Powdery Mildew","sooty mould"]

# Load environment variables from .env file
load_dotenv()

#loading the model by default openai uses gpt 3.5 turbo
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)




# Title of the app
st.title("MangoLeafMedic")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "gif"])


bt=st.button("Recommend Medicine")

if bt:
    if uploaded_file:
        # Read the image as bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
        # Decode the image bytes using OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Convert the image from BGR to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Display the image
        #st.image(img_rgb, caption='Uploaded Image', use_column_width=True)
        st.image(img_rgb, caption='Uploaded Image')


        img_grayscale=cv2.resize(img,(128,128))
        image=img_grayscale/255
        test_image = image.reshape((1,128,128,3))
        prediction=model.predict(test_image)

        #to get the index value 
        index_value = np.argmax(prediction, axis=1)[0]

        #to know the diseases name 
        diseases_name=diseases[index_value]

        st.markdown('## Leaf Diseases Name And its cure')
        st.write("Name of this Plant disesses is", diseases_name)


        #passing Diseases name tp the gpt model to know its cure

        messages=[SystemMessage("I am giving you the Mango leaf Diseses name and you have to give its cure and name of the medicine required for that .note:= If Leaf Is healthy ,then just pass a simple meassage ,like your leaf is healthy"),
        HumanMessage(diseases_name)]

        answer=llm.invoke(messages)
        st.write(answer)










        



