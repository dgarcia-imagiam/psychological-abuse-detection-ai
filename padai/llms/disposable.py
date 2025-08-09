from typing import Protocol
from langchain_huggingface import ChatHuggingFace
import gc, torch


def _dispose_hf_chat_llm(chat_llm) -> None:
    """
    Best-effort cleanup for ChatHuggingFace wrapping a HuggingFacePipeline.
    Frees GPU VRAM by moving model to CPU, dropping refs, and clearing caches.
    """
    try:
        # chat_llm.llm is a HuggingFacePipeline wrapper
        pipe_wrapper = getattr(chat_llm, "llm", None)
        # the actual transformers pipeline
        t_pipe = getattr(pipe_wrapper, "pipeline", None) or pipe_wrapper
        model = getattr(t_pipe, "model", None)
        tok = getattr(t_pipe, "tokenizer", None)

        if model is not None:
            try:
                model.to("cpu")  # optional but helps deterministic release
            except Exception:
                pass

        # Drop references
        del model, tok, t_pipe, pipe_wrapper
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Disposable(Protocol):
    def dispose(self) -> None:
        ...


class NullDisposable:
    def dispose(self) -> None:
        pass


class HuggingFaceDisposable:
    def __init__(self, chat_llm):
        self._llm = chat_llm

    def dispose(self) -> None:
        if self._llm is None:
            return
        _dispose_hf_chat_llm(self._llm)
        self._llm = None


def make_disposable(llm) -> Disposable:
    return HuggingFaceDisposable(llm) if isinstance(llm, ChatHuggingFace) else NullDisposable()
