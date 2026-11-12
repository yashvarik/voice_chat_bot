import io
import os
import re
import wave

import numpy as np
import streamlit as st
import whisper
from dotenv import load_dotenv
from gtts import gTTS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Voice AI", page_icon="🎙️", layout="centered")

# ---------- Styling ----------
st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #1b1f3b 0%, #0d0f1f 60%);
}
.hero {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    margin-bottom: 0;
    background: linear-gradient(90deg, #7c83ff, #00e0c6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #9aa0c3; margin-top: 0.2rem; }
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.mic-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(124, 131, 255, 0.35);
    border-radius: 18px;
    padding: 0.6rem 1rem;
    margin: 0.8rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🎙️ Voice AI Assistant</h1>
  <p>Speak or Type to interact with bot</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Models (ek hi baar load) ----------
SYSTEM = SystemMessage(
    content="You are a voice assistant. Answer in 2 to 3 short spoken sentences. "
    "No markdown, no bullet points, no emojis."
)


@st.cache_resource(show_spinner="Model is getting loaded...")
def load_stt(name: str):
    return whisper.load_model(name)


@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
    )


# ---------- Helpers ----------
def wav_to_array(audio_bytes: bytes) -> np.ndarray:
    """st.audio_input ka wav (16 kHz, 16-bit) -> float32 numpy array."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def transcribe(audio_bytes: bytes, model_name: str) -> str:
    model = load_stt(model_name)
    result = model.transcribe(wav_to_array(audio_bytes), fp16=False, language="en")
    return result["text"].strip()


def ask_llm(history: list[dict], user_text: str) -> str:
    msgs = [SYSTEM]
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    msgs.append(HumanMessage(content=user_text))  # naya sawaal sabse last mein
    return load_llm().invoke(msgs).content


def tts_bytes(text: str, tld: str) -> bytes:
    clean = re.sub(r"[*#<`_]", "", text)
    buf = io.BytesIO()
    gTTS(text=clean, lang="en", tld=tld).write_to_fp(buf)
    return buf.getvalue()


# ---------- Session state ----------
ss = st.session_state
ss.setdefault("history", [])       # [{"role": "user"/"assistant", "content": str}]
ss.setdefault("mic_key", 0)        # mic widget reset karne ke liye
ss.setdefault("pending_audio", None)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    stt_name = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1,
                            help="tiny = fastest, small = most accurate")
    accent = st.selectbox("Voice accent", ["Indian", "US", "UK", "Australian"])
    tld = {"Indian": "co.in", "US": "com", "UK": "co.uk", "Australian": "com.au"}[accent]
    speak_on = st.toggle("Want a voice cover", value=True)
    if st.button("🗑️ Chat clear ", use_container_width=True):
        ss.history = []
        ss.pending_audio = None
        st.rerun()
    st.caption(f"Messages: {len(ss.history)}")

# ---------- Input ----------
st.markdown('<div class="mic-card">', unsafe_allow_html=True)
audio = st.audio_input("🎤 Click on Mic to start the conversation",
                       sample_rate=16000, key=f"mic_{ss.mic_key}")
st.markdown("</div>", unsafe_allow_html=True)
typed = st.chat_input("Type here...")

user_text = None
if audio is not None:
    with st.spinner("Listening..."):
        user_text = transcribe(audio.getvalue(), stt_name)
    ss.mic_key += 1                # agle run mein mic fresh ho jayega
elif typed:
    user_text = typed

# ---------- Process ----------
if user_text:
    if len(user_text) < 2:
        st.warning("Didn't hear a thing., Please try again")
    else:
        with st.spinner("Thinking..."):
            reply = ask_llm(ss.history, user_text)
        ss.history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        if speak_on:
            with st.spinner("Converting to audio..."):
                ss.pending_audio = tts_bytes(reply, tld)
        st.rerun()

# ---------- Chat history ----------
if not ss.history:
    st.info("👋 Shuru karne ke liye mic dabao ya kuch type karo.")

for m in ss.history:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.write(m["content"])

# ---------- Latest reply ki awaaz (ek hi baar autoplay) ----------
if ss.pending_audio:
    st.audio(ss.pending_audio, format="audio/mp3", autoplay=True)
    ss.pending_audio = None