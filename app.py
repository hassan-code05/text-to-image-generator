"""
AI Image Generator
-------------------
A simple Streamlit app that turns a text prompt into an image using the
Hugging Face Inference API (via the huggingface_hub InferenceClient).

This file is meant to be deployed on Streamlit Community Cloud.
The Hugging Face token is read from Streamlit Secrets (st.secrets),
never hard-coded here.
"""

import io
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ---------------------------------------------------------------------------
# Basic page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 AI Image Generator")
st.write(
    "Enter a description below and click **Generate Image** to create a picture "
    "using an AI model hosted on Hugging Face."
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# FLUX.1-schnell is a fast, high-quality, free-to-use text-to-image model.
# "schnell" is German for "fast" -- it's the distilled, speed-optimized
# version of the FLUX.1 model family from Black Forest Labs.
MODEL_ID = "black-forest-labs/FLUX.1-schnell"


@st.cache_resource(show_spinner=False)
def get_client():
    """
    Create (and cache) the Hugging Face InferenceClient.

    st.cache_resource makes sure we only build this client once per app
    session instead of on every button click.
    """
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return None
    return InferenceClient(api_key=token)


def generate_image(prompt: str):
    """
    Send the prompt to the Hugging Face API and return a PIL Image.
    Raises exceptions on failure -- the caller is responsible for
    catching and displaying a friendly error message.
    """
    client = get_client()
    if client is None:
        raise RuntimeError(
            "No Hugging Face API token found. Add HF_TOKEN in Streamlit "
            "Secrets (see the README for instructions)."
        )
    image = client.text_to_image(prompt, model=MODEL_ID)
    return image


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
prompt = st.text_area(
    "Describe the image you want to create:",
    placeholder="A futuristic city at night with flying cars",
    height=100,
)

generate_clicked = st.button("Generate Image", type="primary")

if generate_clicked:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating an image.")
    else:
        with st.spinner("Generating your image... this can take 10-30 seconds."):
            try:
                image = generate_image(prompt)
                st.success("Image generated!")
                st.image(image, caption=prompt, use_container_width=True)

                # Offer a download button
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                st.download_button(
                    label="Download image",
                    data=buffer.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png",
                )

            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 401:
                    st.error(
                        "Authentication failed (401). Your Hugging Face token "
                        "is missing or invalid. Check the HF_TOKEN secret."
                    )
                elif status == 429:
                    st.error(
                        "Rate limit reached (429). You've made too many requests "
                        "recently -- wait a bit and try again."
                    )
                elif status == 503:
                    st.error(
                        "The model is still loading on Hugging Face's servers "
                        "(503). Wait 20-30 seconds and click Generate again."
                    )
                else:
                    st.error(f"Hugging Face API error ({status}): {e}")

            except RuntimeError as e:
                st.error(str(e))

            except Exception as e:
                st.error(f"Something went wrong while generating the image: {e}")

st.divider()
st.caption(
    "Powered by the Hugging Face Inference API "
    f"· Model: `{MODEL_ID}`"
)
