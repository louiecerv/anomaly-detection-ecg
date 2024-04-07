#Input the relevant libraries
import streamlit as st

# Define the Streamlit app
def app():

    text = """Anomaly Detection in ECG using CNN Autoencoders"""
    st.header(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('ecg.png', caption="ECG Anomaly Detection")

    text = """This Streamlit app is designed for anomaly detection in electrocardiogram (ECG) data 
    using a convolutional neural network (CNN) autoencoder implemented in TensorFlow. 
    This app provides a user-friendly interface for analyzing ECG signals to detect anomalies, 
    which can be indicative of various cardiac conditions. Leveraging the power of deep learning, 
    our CNN autoencoder model learns the underlying patterns of normal ECG signals during training 
    and can subsequently identify deviations from these patterns as potential anomalies in real-time 
    data. With this tool, healthcare professionals and researchers can efficiently process ECG data, 
    assisting in early detection and diagnosis of cardiac abnormalities. Explore the functionalities 
    of our app to seamlessly visualize, analyze, and interpret ECG signals for enhanced patient 
    care and medical research.
    """
    st.write(text)
    st.image('autoencoder.png', caption="Convolutional Neural Network Autoencoder")
    text = """A CNN autoencoder is a deep learning architecture adept at capturing intricate 
    patterns within data by compressing it into a latent representation and then reconstructing it. 
    In our context, the CNN autoencoder learns to encode normal ECG signals into a compact 
    representation, which is then decoded back to reconstruct the original signal. 
    However, when presented with anomalous ECG patterns, the reconstruction error is typically 
    higher, signifying a deviation from the learned normal patterns. By leveraging TensorFlow, a 
    powerful deep learning framework, our app facilitates the seamless application of CNN 
    autoencoders for real-time anomaly detection in ECG data, providing a valuable tool for 
    healthcare professionals and researchers to identify potential cardiac 
    irregularities efficiently."""
    st.write(text)

#run the app
if __name__ == "__main__":
    app()
