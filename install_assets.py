"""Pinned, public assets used by the standalone Windows installer."""
from pathlib import Path

MODEL_NAME = "Qwen3.5-2B-Heretic-Q4_K_M.gguf"
MODEL_SIZE = 1312164832
MODEL_SHA256 = "ff31f688573119093db583a75eafc84b15c9cc2a20ab5a31b981e106efd8c9f1"
MODEL_URL = ("https://huggingface.co/darrellbest/Qwen3.5-2B-Heretic-GGUF/resolve/"
             "85bf5e6e623d71d712f978c8a8cab220fa6f31c2/" + MODEL_NAME)
LLAMA_VERSION = "b11819"
RUNTIME_ASSETS = {
    "vulkan": ("llama-b11819-bin-win-vulkan-x64.zip",
               "c3b2484d18258425fe3f69d0b6d07dce01501d6e010578dad3e53150a30df557", 36875213),
    "cpu": ("llama-b11819-bin-win-cpu-x64.zip",
            "312d9925881131d4c9caa7282b3c738cc5084cfc8286e04eecd8a8d19eebb2dc", 24074743),
}


def model_path(root: Path) -> Path:
    return root / ".models" / MODEL_NAME


def runtime_path(root: Path, backend: str = "vulkan") -> Path:
    return root / ".runtimes" / f"llama-{LLAMA_VERSION}-{backend}"


def resolve_setting(value: str, root: Path) -> str:
    path = Path(value.strip().strip('"')).expanduser()
    return str(path if path.is_absolute() else root / path)
