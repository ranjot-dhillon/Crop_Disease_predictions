import streamlit as st
import tensorflow as tf
import numpy as np 
import pandas as pd

def model_prediction(test_img):
    model=tf.keras.models.load_model('trained_model')
    image=tf.keras.preprocessing.image.load_img(test_img,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr]) # add batch with image
    prediction=model.predict(input_arr)
    result_ind=np.argmax(prediction)
    return result_ind

##sidebar

# Adding a logo to the sidebar
st.sidebar.image("logo.webp", use_column_width=True)

# Sidebar content
st.sidebar.title("Dashboard")
mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# home Page
if(mode=="Home"):
    # st.header("crop Disease Prediction")
    # img_path="home.jpg"
    # st.image(img_path,use_column_width=True)
    st.markdown("""
    # Crop Disease Prediction Using Image Recognition""")
    img_path="home.jpg"
    st.image(img_path,use_column_width=True)
    st.markdown("""
## Overview

Welcome to the **Crop Disease Prediction** model! This model leverages **Artificial Intelligence** and **Machine Learning** to predict the disease of crops from images. By analyzing plant images, the model identifies signs of diseases, enabling farmers to take action early and protect their crops.

### How It Works:

- The model uses advanced **image recognition** algorithms to detect diseases in crop leaves.
- It analyzes images of healthy and diseased crops to distinguish between different disease types.
- Once you upload a crop image, the system processes it and provides a disease diagnosis with high accuracy.

## Features:
- **Disease Detection**: Detects a wide range of crop diseases based on leaf images.
- **Fast Prediction**: Quickly analyzes and returns disease predictions from uploaded images.
- **User-Friendly Interface**: Simply upload a picture of the crop leaf to get instant results.
- **Accuracy**: Built with deep learning models trained on thousands of crop images, ensuring high accuracy and reliability.

## How to Use:
1. **Upload an Image**: Capture a clear image of the crop leaf showing any signs of disease.
2. **Predict the Disease**: The model will process the image and detect the presence of any diseases.
3. **Get the Result**: The disease prediction will be displayed with details and suggested actions.

## Supported Crops:
- Wheat
- Corn
- Rice
- Sugercane
- And many more...

## Benefits for Farmers:
- **Early Detection**: Identify diseases in their early stages before they spread.
- **Improved Crop Management**: Make data-driven decisions for better crop care.
- **Increase Crop Yield**: Prevent diseases from harming crops, ensuring higher yield.

## Technologies Used:
- **Deep Learning** (Convolutional Neural Networks - CNN)
- **TensorFlow** and **Keras**
- **OpenCV** for image preprocessing
- **Flask** or **Django** for web application deployment

---

## Contact Us

For any queries, suggestions, or collaborations, feel free to reach out to us at [dhillonranjot007@gmail.com].

---

### Disclaimer:

This model provides predictions based on image analysis and is intended for informational purposes. It is always recommended to consult an agricultural expert for confirmation and advice.

""")
    


if(mode=="About"):
    st.markdown("""
## About the Crop Disease Prediction Model

The **Crop Disease Prediction Model** is an innovative tool designed to assist farmers in the early detection and management of crop diseases. By leveraging advanced **Artificial Intelligence (AI)** and **Machine Learning** technologies, this system allows farmers and agricultural professionals to identify diseases in crops simply by analyzing images of crop leaves.

### Our Mission

Our mission is to empower farmers with modern, accessible technology to improve crop health and maximize yield. By using AI-driven image recognition, we aim to provide an easy-to-use solution for the early identification of crop diseases, enabling farmers to take preventative actions before diseases spread, ultimately reducing crop losses and enhancing productivity.

### How It Was Developed

This model was developed using  **Deep Learning** techniques, particularly **Convolutional Neural Networks (CNNs)**, which are well-suited for image recognition tasks. The model was trained on a large dataset of crop images, with both healthy and diseased crops, to learn the subtle differences and signs of various plant diseases.

Key components of the system include:
- **Data Collection**: High-quality images of crop leaves representing different diseases.
- **Image Preprocessing**: The images undergo preprocessing steps such as resizing, normalization, and enhancement to ensure the model's robustness.
- **Model Training**: Using **TensorFlow** and **Keras**, the deep learning model was trained to detect and classify diseases in crops based on their leaf images.
- **Deployment**: The model is deployed via a web interface, allowing users to upload images and receive quick predictions.

### Vision for the Future

As agriculture continues to evolve, we envision this model becoming an integral tool for farmers worldwide. With continuous updates and improvements, initially system will support an five variety  of crops and its diseases, ensuring that more farmers can benefit from early disease detection.

Our goal is to  improve sustainable farming practices, and help farmers make informed, data-driven decisions for better crop management and protection.

### Why It Matters

Crop diseases are one of the leading causes of crop loss globally, leading to significant financial and food security challenges. Early detection is key to preventing the spread of these diseases, and by providing a tool that enables farmers to detect diseases from images, we aim to mitigate these risks and improve agricultural outcomes.

By empowering farmers with technology, we hope to contribute to a healthier, more sustainable agricultural ecosystem that benefits not only farmers but also the communities that depend on them.

---

### Meet the Team

- **Ranjod Pal Singh** - **Model developer**: Develop and train model on diffenrent type of crops for prediction.
- **Rahul Bhatt** - **User-Interface developer**:  Develop User_Friendly interface for model which is easy to work with
- **Pulkit Singh Bora** - **Data Collection **: Collect data from different resources and preprocess the data for model building .

---

### Contact Us

For any inquiries, collaborations, or feedback, please contact us at [dhillonranjot007@gmail.com].




""")
 
#   # Add your Streamlit content
# if (mode == "Disease Recognition"):
       
#         st.header("Disease Recognition")
#         test_image = st.file_uploader("Choose an Image")
        
#         if st.button("Show Image"):
#             st.image(test_image, use_column_width=True)
        
#         if st.button("Prediction"):
#             with st.spinner("Please Wait...."):
#                 st.write("Our prediction")
#                 result_ind = model_prediction(test_image)
#                 class_name = [
#                     'Bacterial Blight',
#                     'Corn___Common_Rust',
#                     'Corn___Gray_Leaf_Spot',
#                     'Corn___Healthy',
#                     'Corn___Northern_Leaf_Blight',
#                     'Healthy',
#                     'Potato___Early_Blight',
#                     'Potato___Healthy',
#                     'Potato___Late_Blight',
#                     'Red Rot',
#                     'Rice___Brown_Spot',
#                     'Rice___Healthy',
#                     'Rice___Leaf_Blast',
#                     'Rice___Neck_Blast',
#                     'Wheat___Brown_Rust',
#                     'Wheat___Healthy',
#                     'Wheat___Yellow_Rust'
#                 ]
#              # Load the dataset from the CSV file
#             df = pd.read_csv("disease_cure_fertilizers.csv")

#            # Set 'Disease Name' as the index to access the data easily
#             df.set_index('Disease Name', inplace=True)

#             # predicted_disease = class_name[result_ind]
            
#             # # Display prediction
#             # st.success(f"Model is predicting: {predicted_disease}")
            
#             # # Fetch the corresponding cure, fertilizer, and fertilizer name from the DataFrame
#             # if predicted_disease in df.index:
#             #     disease_info = df.loc[predicted_disease]
#             #     cure = disease_info['Cure']
#             #     fertilizer = disease_info['Fertilizer']
#             #     fertilizer_name = disease_info['Fertilizer Name']
                
#             #     # Display the cure and fertilizer information
#             #     st.write(f"Cure: {cure}")
#             #     st.write(f"Recommended Fertilizer: {fertilizer}")
#             #     st.write(f"Fertilizer Name: {fertilizer_name}")
#            

# # Load the dataset from the CSV file
df = pd.read_csv("disease_cure_fertilizers.csv")

# Set 'Disease Name' as the index to access the data easily
df.set_index('Disease Name', inplace=True)


# Add custom CSS for styling


st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f9f9f9;
        color: #333;
    }

    .stButton>button {
        background-color: #32cd32;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
        margin-top: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        background-color: #228b22;
    }

    .stHeader {
        color: #4CAF50;          /* Set the header color */
        text-align: center;      /* Center align the header */
        font-size: 36px;         /* Larger font size */
        font-weight: bold;       /* Make the font bold */
        text-transform: uppercase; /* Uppercase text */
        letter-spacing: 2px;     /* Add some letter spacing */
        padding: 20px;           /* Add padding around the header */
        background-color: #e0ffe0; /* Light background color */
        border-radius: 12px;     /* Rounded corners */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow for a 3D effect */
    }

    /* Styling the file uploader button */
    .stFileUploader>label {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
        padding: 12px 0;
        margin-top: 20px;
    }

    .stFileUploader>div {
        background-color: #e0ffe0;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px dashed #4CAF50;  /* Dashed border for the uploader */
        cursor: pointer;
    }

    .stFileUploader input[type="file"] {
        display: none;
    }

    .stFileUploader>div>button {
        background-color: #32cd32;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        width: 100%;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stFileUploader>div>button:hover {
        background-color: #228b22;
    }

    .stFileUploader .stFileUploader__file-name {
        font-size: 14px;
        color: #333;
        margin-top: 10px;
        font-style: italic;
    }

    .prediction-section {
        margin-top: 30px;
        background-color: #fff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ccc;
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.15);
    }

    .prediction-section h4 {
        margin-bottom: 20px;
        font-size: 22px;
        font-weight: bold;
    }

    .prediction-section p {
        font-size: 18px;
        color: #555;
    }

    .prediction-section .cure {
        background-color: #ffe4e1;
        padding: 18px;
        border-radius: 8px;
    }

    .prediction-section .fertilizer {
        background-color: #e0ffe0;
        padding: 18px;
        border-radius: 8px;
        margin-top: 12px;
    }

    .prediction-section .fertilizer-name {
        background-color: #e0f7ff;
        padding: 18px;
        border-radius: 8px;
        margin-top: 12px;
    }

    .disease-name {
        background-color: #ffcc00;
        padding: 25px;
        border-radius: 12px;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        color: white;
    }

    .prediction-line {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;  
        margin-top: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Mockup of model prediction (replace with your model prediction function)
def model_prediction(image):
    # Mock prediction index (replace with actual model prediction logic)
       return 0  # Example: returning index of a disease class

# Function to show the styled prediction output
def display_prediction(predicted_disease, df):
    # Fetch the corresponding cure, fertilizer, and fertilizer name from the DataFrame
    if predicted_disease in df.index:
        disease_info = df.loc[predicted_disease]
        cure = disease_info['Cure']
        fertilizer = disease_info['Fertilizer']
        fertilizer_name = disease_info['Fertilizer Name']
        
        # Styled disease name with increased font size and larger area
        st.markdown(f"""
        <div class="prediction-section">
            <h4 style="color: #ff6347;">Predicted Disease</h4>
            <div class="disease-name">
                <p>{predicted_disease}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Styled prediction info
        st.markdown(f"""
        <div class="prediction-section">
            <h4 style="color: #ff6347;">Cure</h4>
            <div class="cure">
                <p>{cure}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prediction-section">
            <h4 style="color: #32cd32;">Recommended Fertilizer</h4>
            <div class="fertilizer">
                <p>{fertilizer}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prediction-section">
            <h4 style="color: #1e90ff;">Fertilizer Name</h4>
            <div class="fertilizer-name">
                <p>{fertilizer_name}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No information available for the predicted disease.")

# Example usage in the Streamlit layout
if mode == "Disease Recognition":
    st.markdown("<div class='stHeader'>Crop Disease Recognition</div>", unsafe_allow_html=True)  # Custom styled header
    
    # Upload test image (styled)
    test_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    # Show the uploaded image if a file is selected
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Show Prediction"):
        with st.spinner("Please Wait...."):
            st.markdown("<div class='prediction-line'>Our Prediction</div>", unsafe_allow_html=True)  # Styling the "Our Prediction" line
            result_ind = model_prediction(test_image)
            class_name = [
                'Bacterial Blight',
                'Corn___Common_Rust',
                'Corn___Gray_Leaf_Spot',
                'Corn___Healthy',
                'Corn___Northern_Leaf_Blight',
                'Healthy',
                'Potato___Early_Blight',
                'Potato___Healthy',
                'Potato___Late_Blight',
                'Red Rot',
                'Rice___Brown_Spot',
                'Rice___Healthy',
                'Rice___Leaf_Blast',
                'Rice___Neck_Blast',
                'Wheat___Brown_Rust',
                'Wheat___Healthy',
                'Wheat___Yellow_Rust'
            ]
            predicted_disease = class_name[result_ind]
            
            # Call the display prediction function to show the prediction results
            display_prediction(predicted_disease, df)  # Display the prediction result
            st.markdown("<div class='prediction-line'>Rate Our Model</div>", unsafe_allow_html=True)
            rating = st.slider("How would you rate our model?", min_value=1, max_value=5, value=3, step=1)
            if st.button("Submit Rating"):
               st.success(f"Thank you for rating our model a {rating} out of 5!")
