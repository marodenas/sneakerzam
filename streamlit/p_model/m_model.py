import tensorflow as tf
import streamlit as st


@st.cache
def get_model(whichmodel):
    model = tf.keras.models.load_model('data/complete_model_InceptionV3_v12_classweight_504.h5')
    return model
