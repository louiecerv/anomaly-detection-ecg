#Input the relevant libraries
import streamlit as st
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams["figure.figsize"] = (6, 4)
plt.style.use("ggplot")
import tensorflow as tf
from tensorflow import data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import mae
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report
import os
import time
import contextlib
import io  # Import the io module

def plot_smoothed_mean(data, class_name = "normal", step_size=5, ax=None):
    df = pd.DataFrame(data)
    roll_df = df.rolling(step_size)
    smoothed_mean = roll_df.mean().dropna().reset_index(drop=True)
    smoothed_std = roll_df.std().dropna().reset_index(drop=True)
    margin = 3*smoothed_std
    lower_bound = (smoothed_mean - margin).values.flatten()
    upper_bound = (smoothed_mean + margin).values.flatten()

    ax.plot(smoothed_mean.index, smoothed_mean)
    ax.fill_between(smoothed_mean.index, lower_bound, y2=upper_bound, alpha=0.3, color="red")
    ax.set_title(class_name, fontsize=9)

def plot_sample(normal, anomaly):
    index = np.random.randint(0, len(normal), 2)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    ax[0].plot(normal.iloc[index[0], :].values, label=f"Case {index[0]}")
    ax[0].plot(normal.iloc[index[1], :].values, label=f"Case {index[1]}")
    ax[0].legend(shadow=True, frameon=True, facecolor="inherit", loc=1, fontsize=9)
    ax[0].set_title("Normal")

    ax[1].plot(anomaly.iloc[index[0], :].values, label=f"Case {index[0]}")
    ax[1].plot(anomaly.iloc[index[1], :].values, label=f"Case {index[1]}")
    ax[1].legend(shadow=True, frameon=True, facecolor="inherit", loc=1, fontsize=9)
    ax[1].set_title("Anomaly")

    plt.tight_layout()
    st.pyplot(fig)

# Define the Streamlit app
def app():

    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []


    if "dataset_ready" not in st.session_state:
        st.session_state.dataset_ready = False 

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
    text = """Number of Samples: 14,552 Number of Categories: 2 Sampling Frequency: 125Hz 
    Data Source: Physionet's PTB Diagnostic Database """

    #normal_df = pd.read_csv("heartbeats/ptbdb_normal.csv").iloc[:, :-1]
    #anomaly_df = pd.read_csv("heartbeats/ptbdb_abnormal.csv").iloc[:, :-1]
    normal_df = pd.read_csv("heartbeats/ptbdb_normal.csv", header=0)
    anomaly_df = pd.read_csv("heartbeats/ptbdb_abnormal.csv", header=0)

    st.subheader('Browse the normal ECG Dataset')
    st.write(normal_df)

    st.write("Shape of Normal data", normal_df.shape)
    st.write("Shape of Abnormal data", anomaly_df.shape)

    plot_sample(normal_df, anomaly_df)

    CLASS_NAMES = ["Normal", "Anomaly"]

    normal_df_copy = normal_df.copy()
    anomaly_df_copy = anomaly_df.copy()

    normal_df_copy = normal_df_copy.set_axis(range(1, 189), axis=1)
    anomaly_df_copy = anomaly_df_copy.set_axis(range(1, 189), axis=1)
    normal_df_copy = normal_df_copy.assign(target = CLASS_NAMES[0])
    anomaly_df_copy = anomaly_df_copy.assign(target = CLASS_NAMES[1])
    df = pd.concat((normal_df_copy, anomaly_df_copy))


    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    axes = axes.flatten()
    for i, label in enumerate(CLASS_NAMES, start=1):
        data_group = df.groupby("target")
        data = data_group.get_group(label).mean(axis=0, numeric_only=True).to_numpy()
        plot_smoothed_mean(data, class_name=label, step_size=20, ax=axes[i-1])
    fig.suptitle("Plot of smoothed mean for each class", y=0.95, weight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    normal_df.drop("target", axis=1, errors="ignore", inplace=True)
    normal = normal_df.to_numpy()
    anomaly_df.drop("target", axis=1, errors="ignore", inplace=True)
    anomaly = anomaly_df.to_numpy()

    X_train, X_test = train_test_split(normal, test_size=0.15, random_state=45, shuffle=True)
    st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, anomaly shape: {anomaly.shape}")

    if st.button("Start"):
        input_dim = X_train.shape[-1]
        latent_dim = 32

        model = AutoEncoder(input_dim, latent_dim)
        model.build((None, input_dim))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae")

        # Capture the summary output
        with contextlib.redirect_stdout(io.StringIO()) as new_stdout:
            model.summary()
            summary_str = new_stdout.getvalue()

        # Display the summary using st.text()
        st.text(summary_str)

tf.keras.utils.set_random_seed(1024)

class AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((input_dim, 1)),  # Reshape to 3D for Conv1D
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(latent_dim, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2, padding="same"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(latent_dim, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv1DTranspose(128, 3, strides=1, activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(input_dim)
        ])

    def build(self, input_shape):  # Define the build method
        # No need to modify anything here as layers are already built during initialization
        super(AutoEncoder, self).build(input_shape)

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
#run the app
if __name__ == "__main__":
    app()
