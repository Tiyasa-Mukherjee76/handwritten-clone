import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np

@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data = mnist['data'].reshape(-1, 28, 28)
    labels = mnist['target'].astype(int)
    return data, labels

data, labels = load_mnist()

st.title("MNIST Digit Viewer")

digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Show Samples"):
    digit_indices = np.where(labels == digit)[0]
    selected = np.random.choice(digit_indices, 5, replace=False)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for ax, idx in zip(axs, selected):
        ax.imshow(data[idx], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
