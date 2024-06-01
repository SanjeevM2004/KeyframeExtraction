import streamlit as st
from keyframe_extraction import extract_keyframes

st.title("Keyframe Extraction Tool")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = f"temp_{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)
    
    num_clusters = st.slider("Number of frames to extract", min_value=1, max_value=20, value=5)
    
    if st.button("Extract Keyframes"):
        keyframes = extract_keyframes(video_path, num_clusters=num_clusters)
        st.success(f"Extracted {len(keyframes)} keyframes.")
        
        for i, frame in enumerate(keyframes):
            st.image(frame, caption=f"Keyframe {i+1}")
