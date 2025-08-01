from haystack.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
from llama_cpp import Llama

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: str, use_gpu=False, max_length=512, **kwargs):
        super().__init__(model_name_or_path)
        self.model_path = model_name_or_path
        self.max_length = max_length
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=32768,
            n_threads=8,
            use_mlock=False,
            verbose=False
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        output = self.model(prompt, max_tokens=self.max_length, stop=["</s>"])
        return output["choices"][0]["text"].strip()

    def is_available(self) -> bool:
        return True
