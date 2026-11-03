
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib

from src.autoencoder import MusicAutoencoder

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Classification")
st.write(
    "Explore how autoencoder-based latent feature extraction "
    "helps classify musical genres."
)

st.divider()

# --------------------------------------------------
# LOAD MODELS AND DATA
# --------------------------------------------------

@st.cache_resource
def load_models():

    baseline_model = joblib.load(
        "models/random_forest_baseline.pkl"
    )

    latent_model = joblib.load(
        "models/latent_random_forest.pkl"
    )

    scaler = joblib.load(
        "data/processed/scaler.pkl"
    )

    encoder = joblib.load(
        "data/processed/label_encoder.pkl"
    )

    autoencoder = MusicAutoencoder(
        input_dim=57,
        latent_dim=32
    )

    autoencoder.load_state_dict(
        torch.load(
            "models/music_autoencoder.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    autoencoder.eval()

    return (
        baseline_model,
        latent_model,
        scaler,
        encoder,
        autoencoder
    )


@st.cache_data
def load_dataset():

    return pd.read_csv(
        "data/raw/Data/features_3_sec.csv"
    )


try:

    (
        baseline_model,
        latent_model,
        scaler,
        label_encoder,
        autoencoder
    ) = load_models()

    df = load_dataset()

except Exception as error:

    st.error(f"Could not load project files: {error}")
    st.stop()


# --------------------------------------------------
# SAMPLE SELECTION
# --------------------------------------------------

st.subheader("1. Select a Music Sample")

st.write(
    "Choose a sample from the GTZAN dataset "
    "to compare genre predictions."
)

sample_index = st.selectbox(
    "Select sample number",
    options=list(range(len(df))),
    format_func=lambda x: (
        f"Sample {x + 1} — {df.iloc[x]['filename']}"
    )
)

sample = df.iloc[[sample_index]]

# Prepare original features
X = sample.drop(
    columns=["filename", "length", "label"]
)

X_scaled = scaler.transform(X)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

# Original feature prediction
baseline_prediction = baseline_model.predict(X_scaled)

baseline_probabilities = baseline_model.predict_proba(
    X_scaled
)[0]

# Latent feature extraction
X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)

with torch.no_grad():

    latent_features = autoencoder.extract_latent(
        X_tensor
    ).numpy()

# Latent classifier prediction
latent_prediction = latent_model.predict(
    latent_features
)

latent_probabilities = latent_model.predict_proba(
    latent_features
)[0]

# Convert predictions into genre names
baseline_genre = label_encoder.inverse_transform(
    baseline_prediction
)[0]

latent_genre = label_encoder.inverse_transform(
    latent_prediction
)[0]

actual_genre = sample["label"].iloc[0]


# --------------------------------------------------
# DISPLAY SAMPLE INFORMATION
# --------------------------------------------------

st.divider()
st.subheader("2. Sample Information")

col1, col2, col3 = st.columns(3)

col1.metric("Sample Number", sample_index + 1)
col2.metric("Actual Genre", actual_genre.title())
col3.metric("Original Features", X_scaled.shape[1])


# --------------------------------------------------
# DISPLAY PREDICTIONS
# --------------------------------------------------

st.divider()
st.subheader("3. Genre Prediction Results")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Original Feature Model")

    st.metric(
        "Predicted Genre",
        baseline_genre.title()
    )

    st.metric(
        "Prediction Confidence",
        f"{np.max(baseline_probabilities) * 100:.2f}%"
    )

    st.caption("Uses all 57 original features.")


with col2:

    st.markdown("### Latent Feature Model")

    st.metric(
        "Predicted Genre",
        latent_genre.title()
    )

    st.metric(
        "Prediction Confidence",
        f"{np.max(latent_probabilities) * 100:.2f}%"
    )

    st.caption("Uses 32 learned latent features.")


# --------------------------------------------------
# PREDICTION STATUS
# --------------------------------------------------

st.divider()
st.subheader("4. Prediction Evaluation")

col1, col2 = st.columns(2)

with col1:

    if baseline_genre == actual_genre:

        st.success("Original model predicted correctly!")

    else:

        st.warning("Original model predicted incorrectly.")


with col2:

    if latent_genre == actual_genre:

        st.success("Latent model predicted correctly!")

    else:

        st.warning("Latent model predicted incorrectly.")


# --------------------------------------------------
# PROBABILITY VISUALIZATION
# --------------------------------------------------

st.divider()
st.subheader("5. Genre Probability Comparison")

genres = label_encoder.classes_

probability_df = pd.DataFrame({
    "Genre": genres,
    "Original Model": baseline_probabilities,
    "Latent Model": latent_probabilities
})

probability_df = probability_df.set_index("Genre")

st.bar_chart(probability_df)

st.caption(
    "Probabilities show each model's estimated class scores "
    "for the selected sample. They are not guaranteed to be "
    "perfectly calibrated."
)


# --------------------------------------------------
# LATENT FEATURE INFORMATION
# --------------------------------------------------

st.divider()
st.subheader("6. Learned Latent Representation")

st.write(
    "The autoencoder compresses the original 57 features "
    "into a 32-dimensional representation."
)

st.write("Latent feature values:")

st.dataframe(
    pd.DataFrame(
        latent_features,
        columns=[
            f"Latent Feature {i + 1}"
            for i in range(32)
        ]
    ),
    use_container_width=True
)

st.write(
    f"**Feature reduction:** "
    f"{(1 - 32 / 57) * 100:.2f}%"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Project: Latent Feature Extraction for Musical Genres | "
    "Dataset: GTZAN | Models: Random Forest and PyTorch Autoencoder"
)
