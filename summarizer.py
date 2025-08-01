import tempfile
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer


def download_youtube_audio(url: str) -> str:
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True, abr="160kbps").first()
    temp_dir = tempfile.mkdtemp()
    return stream.download(output_path=temp_dir)


def load_llama_model(model_path: str) -> PromptNode:
    model = PromptModel(
        model_name_or_path=model_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512,
    )
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)


def summarize_youtube_video(url: str, model_path: str) -> str:
    file_path = download_youtube_audio(url)

    whisper = WhisperTranscriber()
    prompt_node = load_llama_model(model_path)

    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="Whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="Summarizer", inputs=["Whisper"])

    output = pipeline.run(file_paths=[file_path])
    raw_output = output.get("results", [""])[0]

    # Clean output
    if "[/INST]" in raw_output:
        return raw_output.split("[/INST]")[-1].strip()
    return raw_output.strip()
