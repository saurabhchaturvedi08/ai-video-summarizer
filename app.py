import streamlit as st
import time
from summarizer import summarize_youtube_video

st.set_page_config(layout="wide")
MODEL_PATH = "llama-2-7b-32k-instruct.Q4_K_S.gguf"


@st.cache_resource
def get_summary(url: str):
    return summarize_youtube_video(url, MODEL_PATH)


def main():
    st.title("🎥 YouTube Video Summarizer")
    st.markdown("Built with **Llama 2**, **Haystack**, **Whisper**, and **Streamlit**")

    with st.expander("ℹ️ About"):
        st.write("Paste a YouTube video URL below to get a summary. Works entirely offline using open-source models.")

    url = st.text_input("📺 Enter YouTube URL")

    if st.button("Summarize") and url:
        with st.spinner("⏳ Processing video... this may take a minute"):
            try:
                start = time.time()
                summary = get_summary(url)
                duration = time.time() - start

                col1, col2 = st.columns(2)
                with col1:
                    st.video(url)
                with col2:
                    st.header("📝 Summary")
                    st.success(summary)
                    st.write(f"⏱️ Time Taken: {duration:.2f} seconds")
                    st.download_button("📄 Download Summary", summary, file_name="summary.txt")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
