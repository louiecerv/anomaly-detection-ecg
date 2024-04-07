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

    text = """Number of Samples: 14,552 Number of Categories: 2 Sampling Frequency: 125Hz 
    Data Source: Physionet's PTB Diagnostic Database """
    st.write(text)

    normal_df = pd.read_csv("heartbeats/ptbdb_normal.csv").iloc[:, :-1]
    anomaly_df = pd.read_csv("heartbeats/ptbdb_abnormal.csv").iloc[:, :-1]
    #normal_df = pd.read_csv("heartbeats/ptbdb_normal.csv")
    #anomaly_df = pd.read_csv("heartbeats/ptbdb_abnormal.csv")

    st.subheader('Browse the ECG Dataset')
    st.write(normal_df)

    st.write("Shape of Normal data", normal_df.shape)
    st.write("Shape of Abnormal data", anomaly_df.shape)

    plot_sample(normal_df, anomaly_df)

    CLASS_NAMES = ["Normal", "Anomaly"]

    normal_df_copy = normal_df.copy()
    anomaly_df_copy = anomaly_df.copy()

    normal_df_copy = normal_df_copy.set_axis(range(1, 188), axis=1)
    anomaly_df_copy = anomaly_df_copy.set_axis(range(1, 188), axis=1)
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
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mae", metrics=['mean_squared_error', 'mean_absolute_error'])

        # Capture the summary output
        with contextlib.redirect_stdout(io.StringIO()) as new_stdout:
            model.summary()
            summary_str = new_stdout.getvalue()
        # Display the summary using st.text()
        st.text(summary_str)

        epochs = 100
        batch_size = 128
        early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)


        history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
            validation_split=0.1, callbacks=[early_stopping, CustomCallback()])

        st.write("Best validation loss:", history.history['val_loss'][-1])

         # Extract loss and MAE/MSE values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_mae = history.history['mean_absolute_error']
        val_mae = history.history['val_mean_absolute_error']
        train_mse = history.history['mean_squared_error']
        val_mse = history.history['val_mean_squared_error']

        # Create the figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize for better visualization

        # Plot loss on the first subplot (ax1)
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy on the second subplot (ax2)
        ax2.plot(train_mae, 'g--', label='Training Mean Absolute Error')
        ax2.plot(train_mse, 'g--', label='Training Mean Squared Error')
        ax2.plot(val_mae, 'r--', label='Validation Mean Absolute Error')
        ax2.plot(val_mse, 'r--', label='Validation Mean Squared Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Set the main title (optional)
        fig.suptitle('Training and Validation Performance')

        plt.tight_layout()  # Adjust spacing between subplots
        st.pyplot(fig)   

        train_mae = model.evaluate(X_train, X_train, verbose=0)
        test_mae = model.evaluate(X_test, X_test, verbose=0)
        anomaly_mae = model.evaluate(anomaly_df, anomaly_df, verbose=0)
        st.write("Training dataset error: ", train_mae)
        st.write("Testing dataset error: ", test_mae)
        st.write("Anomaly dataset error: ", anomaly_mae)

        _, train_loss = predict(model, X_train)
        _, test_loss = predict(model, X_test)
        _, anomaly_loss = predict(model, anomaly)
        threshold = np.mean(train_loss) + np.std(train_loss) # Setting threshold for distinguish normal data from anomalous data

        bins = 40

        # Create the figure and axes objects explicitly
        fig, ax = plt.subplots(figsize=(9, 5), dpi=100)

        # Create the histograms using ax
        sns.histplot(np.clip(train_loss, 0, 0.5), bins=bins, kde=True, label="Train Normal", ax=ax)
        sns.histplot(np.clip(test_loss, 0, 0.5), bins=bins, kde=True, label="Test Normal", ax=ax)
        sns.histplot(np.clip(anomaly_loss, 0, 0.5), bins=bins, kde=True, label="Anomaly", ax=ax)

        # Add vertical line and annotation using ax
        ylim = ax.get_ylim()
        ax.vlines(threshold, 0, ylim[-1], color="k", ls="--")
        ax.annotate(f"Threshold: {threshold:.3f}", xy=(threshold, ylim[-1]), xytext=(threshold + 0.009, ylim[-1]),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)

        # Add legend and display plot
        ax.legend(shadow=True, frameon=True, facecolor="inherit", loc="best", fontsize=9)
        st.pyplot(fig)

        fig, axes = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(12, 6))
        random_indexes = np.random.randint(0, len(X_train), size=5)

        for i, idx in enumerate(random_indexes):
            data = X_train[[idx]]
            plot_examples(model, data, ax=axes[0, i], title="Normal")

        for i, idx in enumerate(random_indexes):
            data = anomaly[[idx]]
            plot_examples(model, data, ax=axes[1, i], title="anomaly")

        plt.tight_layout()
        fig.suptitle("Sample plots (Actual vs Reconstructed by the CNN autoencoder)", y=1.04, weight="bold")
        fig.savefig("autoencoder.png")
        st.pyplot(fig)

        st.write("Training", evaluate_model(threshold, anomaly, model, X_train))
        st.write("Testing", evaluate_model(threshold, anomaly, model, X_test))
        st.write("Anomaly", evaluate_model(threshold, anomaly, model, anomaly))

        plot_confusion_matrix(model, X_train, X_test, anomaly, threshold)

        ytrue, ypred = prepare_labels(model, X_train, X_test, anomaly, threshold)
        #use text and not write for correct formatting
        st.text(classification_report(ytrue, ypred, target_names=CLASS_NAMES))

def predict(model, X):
    pred = model.predict(X, verbose=False)
    loss = mae(pred, X)
    return pred, loss

def plot_examples(model, data, ax, title):
    pred, loss = predict(model, data)
    ax.plot(data.flatten(), label="Actual")
    ax.plot(pred[0], label = "Predicted")
    ax.fill_between(range(1, 188), data.flatten(), pred[0], alpha=0.3, color="r")
    ax.legend(shadow=True, frameon=True,
              facecolor="inherit", loc=1, fontsize=7)
#                bbox_to_anchor = (0, 0, 0.8, 0.25))
    ax.set_title(f"{title} (loss: {loss[0]:.3f})", fontsize=9.5)

def evaluate_model(threshold, anomaly, model, data):
    pred, loss = predict(model, data)
    if id(data) == id(anomaly):
        accuracy = np.sum(loss > threshold)/len(data)
    else:
        accuracy = np.sum(loss <= threshold)/len(data)
    return f"Accuracy: {accuracy:.2%}"

def prepare_labels(model, train, test, anomaly, threshold):
    ytrue = np.concatenate((np.ones(len(train)+len(test), dtype=int), np.zeros(len(anomaly), dtype=int)))
    _, train_loss = predict(model, train)
    _, test_loss = predict(model, test)
    _, anomaly_loss = predict(model, anomaly)
    train_pred = (train_loss <= threshold).numpy().astype(int)
    test_pred = (test_loss <= threshold).numpy().astype(int)
    anomaly_pred = (anomaly_loss < threshold).numpy().astype(int)
    ypred = np.concatenate((train_pred, test_pred, anomaly_pred))
    return ytrue, ypred

def plot_confusion_matrix(model, train, test, anomaly, threshold):
    ytrue, ypred = prepare_labels(model, train, test, anomaly, threshold)
    accuracy = accuracy_score(ytrue, ypred)
    precision = precision_score(ytrue, ypred)
    recall = recall_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    st.write(f"""\
        Accuracy: {accuracy:.2%}
        Precision: {precision:.2%}
        Recall: {recall:.2%}
        f1: {f1:.2%}\n
        """)

    cm = confusion_matrix(ytrue, ypred)
    cm_norm = confusion_matrix(ytrue, ypred, normalize="true")
    data = np.array([f"{count}\n({pct:.2%})" for count, pct in zip(cm.ravel(), cm_norm.ravel())]).reshape(cm.shape)
    labels = ["Anomaly", "Normal"]

    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=data, fmt="", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix", weight="bold")
    plt.tight_layout()
    st.pyplot(fig)

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

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        mae = logs['mean_absolute_error']
        
        # Update the Streamlit interface with the current epoch's output
        st.text(f"Epoch {epoch}: loss = {loss:.4f} Mean Absolute Errror = {mae:.4f}")

#run the app
if __name__ == "__main__":
    app()
