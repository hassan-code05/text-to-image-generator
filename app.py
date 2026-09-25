"""
AI Image Generator
-------------------
A polished Streamlit interface for generating images from text prompts
using the Hugging Face Inference API.
"""

import io
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS - UI ONLY
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>

    /* =========================
       MAIN APP
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(168, 85, 247, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f8fafc 0%,
                #eef2ff 50%,
                #faf5ff 100%
            );

        color: #1e293b !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }


    /* =========================
       FORCE TEXT COLORS
       ========================= */

    html,
    body,
    [class*="css"],
    .stApp,
    .stMarkdown,
    p,
    span,
    label,
    div {
        color: #1e293b;
    }

    /* Streamlit labels */

    label {
        color: #334155 !important;
    }

    /* Markdown text */

    .stMarkdown p {
        color: #475569 !important;
    }


    /* =========================
       HERO
       ========================= */

    .hero {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
    }

    .hero-icon {
        width: 78px;
        height: 78px;
        margin: 0 auto 1.2rem auto;
        border-radius: 24px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 38px;

        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );

        box-shadow:
            0 15px 35px rgba(99, 102, 241, 0.25);
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;

        margin-bottom: 0.5rem;

        background: linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        font-size: 1.08rem;
        color: #64748b !important;

        max-width: 650px;
        margin: 0 auto;

        line-height: 1.7;
    }


    /* =========================
       PROMPT CARD
       ========================= */

    .prompt-card {
        background: rgba(255, 255, 255, 0.95);

        border: 1px solid #e2e8f0;

        border-radius: 24px;

        padding: 1.5rem;

        margin-top: 1rem;
        margin-bottom: 1.25rem;

        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.08);

        backdrop-filter: blur(12px);
    }

    .section-label {
        font-size: 0.92rem;
        font-weight: 700;

        color: #334155 !important;

        margin-bottom: 0.5rem;
    }

    .hint {
        font-size: 0.82rem;

        color: #64748b !important;

        margin-top: 0.45rem;
    }


    /* =========================
       TEXT AREA
       ========================= */

    .stTextArea textarea {
        color: #1e293b !important;

        background-color: #ffffff !important;

        border: 1px solid #cbd5e1 !important;

        border-radius: 16px !important;

        padding: 1rem !important;

        font-size: 1rem !important;

        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        color: #1e293b !important;

        background-color: #ffffff !important;

        border-color: #6366f1 !important;

        box-shadow:
            0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    /* Placeholder */

    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }


    /* =========================
       GENERATE BUTTON
       ========================= */

    .stButton > button {
        width: 100%;

        min-height: 52px;

        border: none !important;

        border-radius: 15px !important;

        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            ) !important;

        color: #ffffff !important;

        font-size: 1rem !important;

        font-weight: 700 !important;

        box-shadow:
            0 10px 25px rgba(79, 70, 229, 0.25);

        transition: all 0.2s ease !important;
    }

    .stButton > button p,
    .stButton > button span {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 15px 30px rgba(79, 70, 229, 0.32);
    }


    /* =========================
       RESULT SECTION
       ========================= */

    .result-title {
        text-align: center;

        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .result-title h2 {
        font-size: 1.6rem;

        font-weight: 750;

        color: #1e293b !important;
    }

    .image-card {
        background: #ffffff;

        padding: 0.75rem;

        border-radius: 22px;

        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.12);

        border: 1px solid #e2e8f0;
    }


    /* =========================
       DOWNLOAD BUTTON
       ========================= */

    .stDownloadButton > button {
        width: 100%;

        min-height: 48px;

        border-radius: 14px !important;

        font-weight: 650 !important;

        border: 1px solid #cbd5e1 !important;

        background: #ffffff !important;

        color: #334155 !important;

        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button p,
    .stDownloadButton > button span {
        color: #334155 !important;
    }

    .stDownloadButton > button:hover {
        border-color: #6366f1 !important;

        color: #4f46e5 !important;

        transform: translateY(-1px);
    }


    /* =========================
       ALERTS
       ========================= */

    .stAlert {
        border-radius: 14px !important;
    }


    /* =========================
       SPINNER
       ========================= */

    .stSpinner > div {
        color: #4f46e5 !important;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;

        padding-top: 2rem;

        color: #94a3b8 !important;

        font-size: 0.82rem;

        line-height: 1.7;
    }

    .footer strong {
        color: #64748b !important;
    }

    .footer code {
        color: #6366f1 !important;

        background: #eef2ff !important;

        padding: 2px 6px;

        border-radius: 5px;
    }


    /* =========================
       MOBILE
       ========================= */

    @media (max-width: 640px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.5rem;
        }

        .hero h1 {
            font-size: 2.2rem;
        }

        .hero p {
            font-size: 0.95rem;
        }

        .hero-icon {
            width: 65px;
            height: 65px;
            font-size: 31px;
        }

        .prompt-card {
            padding: 1rem;
            border-radius: 18px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero / Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🎨</div>
        <h1>AI Image Generator</h1>
        <p>
            Turn your imagination into stunning visuals.
            Describe what you want to see and let AI bring your idea to life.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen-Image-2512"


@st.cache_resource(show_spinner=False)
def get_client():
    """
    Create (and cache) the Hugging Face InferenceClient.
    """
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return None
    return InferenceClient(api_key=token)


def generate_image(prompt: str):
    """
    Send the prompt to the Hugging Face API and return a PIL Image.
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
# Prompt Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="prompt-card">
        <div class="section-label">✨ Describe your image</div>
    """,
    unsafe_allow_html=True,
)

prompt = st.text_area(
    "Prompt",
    placeholder=(
        "Example: A futuristic city at night, glowing neon buildings, "
        "flying cars, cinematic lighting, highly detailed..."
    ),
    height=140,
    label_visibility="collapsed",
)

st.markdown(
    """
        <div class="hint">
            💡 Tip: Add details about the subject, environment, lighting,
            colors, style, and mood for more specific results.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Generate Button
# ---------------------------------------------------------------------------
generate_clicked = st.button(
    "✨  Generate Image",
    type="primary",
)

# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------
if generate_clicked:

    if not prompt.strip():

        st.warning("Please enter a prompt before generating an image.")

    else:

        with st.spinner("Creating your image... Please wait."):

            try:

                image = generate_image(prompt)

                st.success("Your image has been generated successfully! 🎉")

                # Result heading
                st.markdown(
                    """
                    <div class="result-title">
                        <h2>🖼️ Your Generated Image</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Image container
                st.markdown(
                    '<div class="image-card">',
                    unsafe_allow_html=True,
                )

                st.image(
                    image,
                    caption=prompt,
                    use_container_width=True,
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True,
                )

                # Download button
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")

                st.download_button(
                    label="⬇️  Download Image",
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
                        "recently — wait a bit and try again."
                    )

                elif status == 503:

                    st.error(
                        "The model is still loading on Hugging Face's servers "
                        "(503). Wait 20–30 seconds and click Generate again."
                    )

                else:

                    st.error(
                        f"Hugging Face API error ({status}): {e}"
                    )

            except RuntimeError as e:

                st.error(str(e))

            except Exception as e:

                st.error(
                    f"Something went wrong while generating the image: {e}"
                )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer">
        <strong>AI Image Generator</strong><br>
        Powered by the Hugging Face Inference API
        · Model: <code>{MODEL_ID}</code>
    </div>
    """,
    unsafe_allow_html=True,
)
