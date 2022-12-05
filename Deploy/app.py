import streamlit as st
import tensorflow as tf
import numpy as np
 
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fresh_stale_banana.h5')
    return model
 
 
if __name__ == '__main__':
 
    model = load_model()
    st.title('Fresh and Stale Banana Differentiation')
 
    uploaded_file = st.file_uploader('Upload an image of banana')
 
    if not uploaded_file:
        st.warning("Please upload an image in JPEG, PNG, GIF, BMP format before proceeding!")
        st.stop()
    else:
        image_as_bytes = uploaded_file.read()
        st.image(image_as_bytes, use_column_width=True)
        img = tf.io.decode_image(image_as_bytes, channels=3)
        img = tf.image.resize(img, (200, 200))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255
        img_array = tf.expand_dims(img_array, 0)
        
        images = np.vstack([img_array])
        classes = model.predict(images)
        
        if classes[0] > 0.5:
            st.write("This image most likely belongs to class.")
            st.warning('stale banana')
        else:
            st.write("This image most likely belongs to class.")
            st.warning('fresh banana')