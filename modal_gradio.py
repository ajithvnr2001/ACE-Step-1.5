import modal
import os

# Define the Modal App
app = modal.App("acestep-gradio")

# Define the Image with dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    # Layer 1: Build tools
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    # Layer 2: Core ML stack
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "torchvision",
        "torchao==0.11.0",
    )
    # Layer 3: Application dependencies
    .pip_install(
        "transformers>=4.51.0,<4.58.0",
        "diffusers",
        "gradio>=6.5.1",
        "matplotlib>=3.7.5",
        "scipy>=1.10.1",
        "soundfile>=0.13.1",
        "loguru>=0.7.3",
        "einops>=0.8.1",
        "accelerate>=1.12.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "numba>=0.63.1",
        "vector-quantize-pytorch>=1.27.15",
        "toml",
        "modelscope",
        "peft>=0.7.0",
        "lightning>=2.0.0",
        "tensorboard>=2.0.0",
        "huggingface_hub",
        "diskcache",
    )
    # Layer 4: Flash Attention (No Build Isolation is CRITICAL here)
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    # Copy the core application and local dependencies
    .add_local_dir("acestep", remote_path="/root/acestep", copy=True)
    .add_local_file("cli.py", remote_path="/root/cli.py", copy=True)
    .add_local_dir("assets", remote_path="/root/assets", copy=True)
    .add_local_dir("examples", remote_path="/root/examples", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    # Install nano-vllm from the copied local path
    .run_commands("pip install -e /root/acestep/third_parts/nano-vllm")
)

# Persistent volume for model checkpoints
volume = modal.Volume.from_name("acestep-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",  # Using A100 for top performance as requested
    volumes={"/root/checkpoints": volume},
    timeout=3600,
)
@modal.asgi_app()
def gradio_app():
    import sys
    sys.path.append("/root")
    
    # Set environment variables for top model configuration
    os.environ["ACESTEP_INIT_LLM"] = "true"
    os.environ["ACESTEP_CONFIG_PATH"] = "acestep-v15-turbo"
    os.environ["ACESTEP_LM_MODEL_PATH"] = "acestep-5Hz-lm-4B"
    os.environ["ACESTEP_DOWNLOAD_SOURCE"] = "huggingface"
    
    from acestep.acestep_v15_pipeline import create_demo
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    
    # Initialize handlers
    handler = AceStepHandler()
    llm_handler = LLMHandler()
    
    # Create Gradio interface
    demo = create_demo(init_params=None)
    
    # Wrap in FastAPI for Modal compatibility
    import gradio as gr
    from fastapi import FastAPI
    web_app = FastAPI()
    return gr.mount_gradio_app(web_app, demo, path="/")
