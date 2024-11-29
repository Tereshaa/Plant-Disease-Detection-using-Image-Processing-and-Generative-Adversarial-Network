import streamlit as st
import tensorflow as tf
import numpy as np

# Load models
@st.cache_resource
def load_models():
    gan_segmented_cnn = tf.keras.models.load_model("MODELS/best_gen_segmented_cnn.keras")
    segmented_cnn = tf.keras.models.load_model("MODELS/best_segmented_cnn.keras")
    segmentation_model = tf.keras.models.load_model("MODELS/best_unet_model.keras")
    return gan_segmented_cnn, segmented_cnn, segmentation_model

gan_segmented_cnn, segmented_cnn, segmentation_model = load_models()

# Image Segmentation Function
def segment_image(image):
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    print(f"[DEBUG] Original image array shape: {input_arr.shape}")  # Debug
    input_arr = tf.image.resize(input_arr, (256, 256)) / 255.0  # Resize for segmentation model
    print(f"[DEBUG] Resized and normalized image shape: {input_arr.shape}")  # Debug
    input_arr = np.expand_dims(input_arr, axis=0)  # Batch format
    print(f"[DEBUG] Batched image shape for segmentation model: {input_arr.shape}")  # Debug
    segmented_image = segmentation_model.predict(input_arr)[0]  # Remove batch dimension
    print(f"[DEBUG] Segmentation model output shape: {segmented_image.shape}")  # Debug
    return segmented_image

# Model Prediction Function
def model_prediction(test_image, model_choice):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    print(f"[DEBUG] Loaded image size for prediction: {image.size}")  # Debug
    segmented_image = segment_image(image)  # Segment the image first

    # Choose the CNN model and class names based on selection
    if model_choice == "Segmented CNN":
        cnn_model = gan_segmented_cnn
        class_names = ['Tomato_Spider_mites', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Late_blight', 'Tomato_Early_blight',
                       'Tomato_Mosaic_virus', 'Tomato_Septoria_leafspot', 'Tomato_Bacterial_spot', 'Tomato_Healthy',
                       'Tomato_Target_Spot', 'Tomato_Leaf_Mold']

    elif model_choice == "GAN-Segmented CNN":
        cnn_model = segmented_cnn
        class_names = [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy',
            'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Mosaic_virus',
            'Tomato_Septoria_leafspot', 'Tomato_Spider_mites', 'Tomato_Target_Spot',
            'Tomato_Yellow_Leaf_Curl_Virus'
        ]
    else:
        st.error("Invalid model selection.")
        return None

    # Prepare segmented image for CNN model
    segmented_image = np.expand_dims(segmented_image, axis=0)  # Batch format
    print(f"[DEBUG] Batched segmented image shape for CNN model: {segmented_image.shape}")  # Debug
    predictions = cnn_model.predict(segmented_image)
    print(f"[DEBUG] Prediction output shape: {predictions.shape}")  # Debug
    return np.argmax(predictions), class_names  # Return index of max element and class names

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # Model Selection
    model_choice ="GAN-Segmented CNN"

    test_image = st.file_uploader("Choose an Image:")
    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index, class_names = model_prediction(test_image, model_choice)
            print(f"[DEBUG] Final Prediction Index: {result_index}, Class: {class_names[result_index]}")


            if result_index is not None:
                st.success("{}".format(class_names[result_index]))



# import streamlit as st
# import tensorflow as tf
# import numpy as np

# # Load models
# @st.cache_resource
# def load_models():
#     gan_segmented_cnn = tf.keras.models.load_model("models/best_gen_segmented_cnn.keras")
#     segmented_cnn = tf.keras.models.load_model("models/best_segmented_cnn.keras")
#     segmentation_model = tf.keras.models.load_model("models/best_unet_model.keras")
#     return gan_segmented_cnn, segmented_cnn, segmentation_model

# gan_segmented_cnn, segmented_cnn, segmentation_model = load_models()

# # Image Segmentation Function
# def segment_image(image):
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = tf.image.resize(input_arr, (256, 256)) / 255.0   # Resize for segmentation model
#     input_arr = np.expand_dims(input_arr, axis=0)  # Batch format
#     segmented_image = segmentation_model.predict(input_arr)[0]  # Remove batch dimension
#     return segmented_image

# # Model Prediction Function
# def model_prediction(test_image, model_choice):
#     # Load and preprocess the image
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
#     segmented_image = segment_image(image)  # Segment the image first

#     # Choose the CNN model and class names based on selection
#     if model_choice == "GAN-Segmented CNN":
#         cnn_model = gan_segmented_cnn
#         class_names = ['Tomato_Spider_mites', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Late_blight', 'Tomato_Early_blight', 
#                        'Tomato_Mosaic_virus', 'Tomato_Septoria_leafspot', 'Tomato_Bacterial_spot', 'Tomato_Healthy', 
#                        'Tomato_Target_Spot', 'Tomato_Leaf_Mold']
        
#     elif model_choice == "Segmented CNN":
#         cnn_model = segmented_cnn
#         class_names = [
#             'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy',
#             'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Mosaic_virus',
#             'Tomato_Septoria_leafspot', 'Tomato_Spider_mites', 'Tomato_Target_Spot',
#             'Tomato_Yellow_Leaf_Curl_Virus'
#         ]
#     else:
#         st.error("Invalid model selection.")
#         return None

#     # Prepare segmented image for CNN model
#     segmented_image = np.expand_dims(segmented_image, axis=0)  # Batch format
#     predictions = cnn_model.predict(segmented_image)
#     return np.argmax(predictions), class_names  # Return index of max element and class names

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# # Main Page
# if app_mode == "Home":
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     image_path = "home_page.jpeg"
#     st.image(image_path, use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍

#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
#     """)

# # Prediction Page
# elif app_mode == "Disease Recognition":
#     st.header("Disease Recognition")

#     # Model Selection
#     model_choice = st.selectbox("Choose Model", ["GAN-Segmented CNN", "Segmented CNN"])

#     test_image = st.file_uploader("Choose an Image:")
#     if test_image:
#         if st.button("Show Image"):
#             st.image(test_image, use_column_width=True)

#         # Predict button
#         if st.button("Predict"):
#             st.write("Our Prediction")
#             result_index, class_names = model_prediction(test_image, model_choice)

#             if result_index is not None:
#                 st.success("{}".format(class_names[result_index]))


# import streamlit as st
# import tensorflow as tf
# import numpy as np

# # Load models
# @st.cache_resource
# def load_models():
#     gan_segmented_cnn = tf.keras.models.load_model("models/best_gen_segmented_cnn.keras")
#     segmented_cnn = tf.keras.models.load_model("models/best_segmented_cnn.keras")
#     segmentation_model = tf.keras.models.load_model("models/best_unet_model.keras")
#     return gan_segmented_cnn, segmented_cnn, segmentation_model

# gan_segmented_cnn, segmented_cnn, segmentation_model = load_models()

# # Image Segmentation Function
# def segment_image(image):
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = tf.image.resize(input_arr, (256, 256)) / 255.0   # Resize for segmentation model
#     input_arr = np.expand_dims(input_arr, axis=0)  # Batch format
#     segmented_image = segmentation_model.predict(input_arr)[0]  # Remove batch dimension
#     return segmented_image

# # Model Prediction Function
# def model_prediction(test_image, model_choice):
#     # Load and preprocess the image
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
#     segmented_image = segment_image(image)  # Segment the image first

#     # Choose the CNN model based on selection
#     if model_choice == "GAN-Segmented CNN":
#         cnn_model = gan_segmented_cnn
#     elif model_choice == "Segmented CNN":
#         cnn_model = segmented_cnn
#     else:
#         st.error("Invalid model selection.")
#         return None

#     # Prepare segmented image for CNN model
#     segmented_image = np.expand_dims(segmented_image / 255.0, axis=0) # Batch format
#     predictions = cnn_model.predict(segmented_image)
#     return np.argmax(predictions)  # Return index of max element

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# # Main Page
# if app_mode == "Home":
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     image_path = "home_page.jpeg"
#     st.image(image_path, use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍

#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
#     """)

# # Prediction Page
# elif app_mode == "Disease Recognition":
#     st.header("Disease Recognition")

#     # Model Selection
#     model_choice = st.selectbox("Choose Model", ["GAN-Segmented CNN", "Segmented CNN"])

#     test_image = st.file_uploader("Choose an Image:")
#     if test_image:
#         if st.button("Show Image"):
#             st.image(test_image, use_column_width=True)

#         # Predict button
#         if st.button("Predict"):
#             st.write("Our Prediction")
#             result_index = model_prediction(test_image, model_choice)

#             # Reading Labels
#             class_name = [
#                 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy',
#                 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Mosaic_virus',
#                 'Tomato_Septoria_leafspot', 'Tomato_Spider_mites', 'Tomato_Target_Spot',
#                 'Tomato_Yellow_Leaf_Curl_Virus'
#             ]
#             if result_index is not None:
#                 st.success("{}".format(class_name[result_index]))



# import streamlit as st
# import tensorflow as tf
# import numpy as np


# #Tensorflow Model Prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model("models/best_gen_segmented_cnn.keras")
#     image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) #convert single image to batch
#     predictions = model.predict(input_arr)
#     return np.argmax(predictions) #return index of max element

# #Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","Disease Recognition"])

# #Main Page
# if(app_mode=="Home"):
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     image_path = "home_page.jpeg"
#     st.image(image_path,use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍

#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     """)

# #Prediction Page
# elif(app_mode=="Disease Recognition"):
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image,width=4,use_column_width=True)
#     #Predict button
#     if(st.button("Predict")):
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)
#         #Reading Labels
#         class_name = ['Tomato_Bacterial_spot',
#                'Tomato_Early_blight',
#                'Tomato_Healthy',
#                'Tomato_Late_blight',
#                'Tomato_Leaf_Mold',
#                'Tomato_Mosaic_virus',
#                'Tomato_Septoria_leafspot',
#                'Tomato_Spider_mites',
#                'Tomato_Target_Spot',
#                'Tomato_Yellow_Leaf_Curl_Virus']
#         st.success("{}".format(class_name[result_index]))
