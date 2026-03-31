import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer, util
import joblib
import os
import tempfile

# ==========================================
# 1. LOAD AI MODELS (Cached for speed)
# ==========================================
@st.cache_resource
def load_ai_models():
    """Loads heavy models once and keeps them in memory."""
    models = {}
    
    # 1. Load Whisper for Audio Transcription
    models['whisper'] = whisper.load_model("base")
    
    # 2. Load Semantic AI for comparing Title vs Transcript
    models['semantic'] = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Load your trained Scikit-Learn model (Fallback to dummy if missing)
    try:
        models['clickbait'] = joblib.load("models/clickbait_model.pkl")
    except FileNotFoundError:
        st.warning("⚠️ 'models/clickbait_model.pkl' not found. Please run train.py first. (Waiting for model...)")
        st.stop()
        
    return models

# ==========================================
# 2. NLP & SCORING LOGIC
# ==========================================
def calculate_clickbait_gap(title: str, transcript: str, semantic_model) -> float:
    """Checks if the video actually talks about the title."""
    title_embedding = semantic_model.encode(title, convert_to_tensor=True)
    intro_embedding = semantic_model.encode(transcript[:1000], convert_to_tensor=True)
    
    similarity = util.cos_sim(title_embedding, intro_embedding).item()
    similarity_score = round(similarity * 100, 2)
    
    return similarity_score

def calculate_hook_score(transcript: str) -> dict:
    """Analyzes the first 100 words for engagement markers."""
    intro_text = " ".join(transcript.split()[:100]).lower()
    score = 0
    feedback = []
    
    if "?" in intro_text or any(w in intro_text for w in ["how", "why", "what", "secret"]):
        score += 35
        feedback.append("✅ Great use of curiosity/questions.")
    else: feedback.append("❌ Lacks a strong curiosity hook.")
        
    if any(w in intro_text for w in ["you", "your", "imagine"]):
        score += 35
        feedback.append("✅ Speaks directly to the viewer.")
    else: feedback.append("❌ Does not directly address the viewer.")
        
    if "!" in intro_text or any(w in intro_text for w in ["now", "today", "instantly", "always"]):
        score += 30
        feedback.append("✅ High energy/urgency language detected.")
    else: feedback.append("❌ Lacks urgent/high-energy language.")
        
    return {"score": score, "feedback": feedback}

def predict_retention(hook_score: float, deception_score: float) -> dict:
    """Predicts audience drop-off based on AI scores."""
    minute_1 = 25.0 + (hook_score * 0.60) 
    honesty_multiplier = (100 - deception_score) / 100
    avg_retention = minute_1 * (0.40 + (honesty_multiplier * 0.50))
    completion = avg_retention * (0.30 + (honesty_multiplier * 0.60))
    
    return {
        "minute_1": int(max(0, min(100, minute_1))),
        "average": int(max(0, min(100, avg_retention))),
        "completion": int(max(0, min(100, completion)))
    }

# ==========================================
# 3. STREAMLIT UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Video AI Analyzer", page_icon="📊", layout="centered")

st.title("📊 Video Content AI Analyzer")
st.markdown("Upload a video/audio file. AI will transcribe it, analyze the hook, and check if it delivers on the title's promise.")

# Load models (shows a spinner automatically while loading)
with st.spinner("Loading AI Models into memory..."):
    ai = load_ai_models()

# UI Inputs
video_title = st.text_input("Enter the Planned Video Title / Topic:")
uploaded_file = st.file_uploader("Upload Video/Audio File", type=['mp4', 'mp3', 'wav', 'm4a'])

if st.button("Run Full AI Analysis", type="primary"):
    if not video_title or not uploaded_file:
        st.error("Please provide both a title and a media file.")
    else:
        # Step 1: Save the uploaded file temporarily so Whisper can read it
        with st.spinner("🎧 Transcribing Audio with Whisper AI (This may take a minute)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_filepath = tmp_file.name
            
            # Run Whisper Transcription
            transcript_result = ai['whisper'].transcribe(tmp_filepath)
            transcript_text = transcript_result["text"]
            
            # Clean up the heavy video file
            os.remove(tmp_filepath)

        # Step 2: Run NLP Analysis
        with st.spinner("🧠 Running Deep NLP Analysis..."):
            # A. Title Sensationalism
            title_prob = ai['clickbait'].predict_proba([video_title])[0][1] * 100
            
            # B. Topic vs Content Gap
            similarity = calculate_clickbait_gap(video_title, transcript_text, ai['semantic'])
            delivery_penalty = 100 - similarity
            deception_score = round((title_prob + delivery_penalty) / 2, 2)
            
            # C. Hook Analysis
            hook_data = calculate_hook_score(transcript_text)
            
            # D. Retention Math
            retention = predict_retention(hook_data["score"], deception_score)

        # Step 3: Display Results
        st.success("Analysis Complete!")
        
        st.subheader("Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Deception / Clickbait Penalty", f"{deception_score}%", delta="Lower is better", delta_color="inverse")
            st.caption("Measures if the title is sensational and if the video actually talks about it.")
            
        with col2:
            st.metric("Intro Hook Score", f"{hook_data['score']}%", delta="Higher is better")
            st.caption("Measures engagement, questions, and energy in the first 30 seconds.")

        st.divider()

        st.subheader("📈 Predicted Audience Retention")
        st.write(f"**Still watching at 1:00:** {retention['minute_1']}%")
        st.progress(retention['minute_1'])
        
        st.write(f"**Average View Duration:** {retention['average']}%")
        st.progress(retention['average'])
        
        st.write(f"**Finished the Video:** {retention['completion']}%")
        st.progress(retention['completion'])

        st.divider()
        
        st.subheader("📝 NLP Notes & Feedback")
        for feedback in hook_data['feedback']:
            st.write(feedback)
            
        with st.expander("View Full AI Transcript"):
            st.write(transcript_text)