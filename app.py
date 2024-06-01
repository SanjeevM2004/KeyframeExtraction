import streamlit as st
from keyframe_extraction import main as extract_keyframes

st.title("Keyframe Extraction Tool")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = f"temp_{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)
    
    if st.button("Extract Keyframes"):
        keyframes = extract_keyframes(video_path)
        st.success(f"Extracted {len(keyframes)} keyframes.")
        
        for i, frame in enumerate(keyframes):
            st.image(frame, caption=f"Keyframe {i+1}")
