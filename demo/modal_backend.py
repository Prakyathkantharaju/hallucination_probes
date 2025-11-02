"""
Modal backend for serving hallucination detection probes with vLLM.
Provides a fast inference API for the Streamlit dashboard.
"""

import modal
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import json
from transformers import AutoTokenizer
import numpy as np
import os
from pathlib import Path
import shutil
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PROBE_REPO = "andyrdt/hallucination-probes"
N_GPU = 2
GPU_CONFIG = f"H100:{N_GPU}"
SCALEDOWN_WINDOW = 2 * 60 # 2 minutes
TIMEOUT = 15 * 60 # 15 minutes

VOLUME = modal.Volume.from_name("hallucination-models", create_if_missing=True)
VOLUME_PATH = "/models"
PROBES_DIR = Path(VOLUME_PATH) / "probes"

if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent)
    assert os.getenv("HF_TOKEN"), "HF_TOKEN must be set to be able to load Llama models from HuggingFace"
    LOCAL_HF_TOKEN_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
else:
    LOCAL_HF_TOKEN_SECRET = modal.Secret.from_dict({})

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0", 
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "vllm>=0.5.0",
        "jaxtyping>=0.2.0",
        "huggingface_hub>=0.20.0",
    )
)

app = modal.App("hallucination-probe-backend", image=image)

def download_probe_from_hf(
    repo_id: str,
    repo_subfolder: str,
    local_folder: Path,
    token: Optional[str] = None
) -> None:
    """Simplified probe download function for Modal."""
    api = HfApi()
    
    # Create local folder
    local_folder.mkdir(parents=True, exist_ok=True)
    
    # List files in the repository subfolder
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision="main")
    
    # Filter files by subfolder
    if repo_subfolder:
        subfolder_files = [f for f in repo_files if f.startswith(f"{repo_subfolder}/")]
    else:
        subfolder_files = repo_files
    
    # Download each file
    for file_path in subfolder_files:
        # Get relative path within subfolder
        if repo_subfolder:
            relative_path = file_path[len(repo_subfolder):].lstrip('/')
        else:
            relative_path = file_path
        
        # Create subdirectory if needed
        local_file_path = local_folder / relative_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            token=token
        )
        
        # Copy to destination
        shutil.copy(downloaded_file, local_file_path)
    
    print(f"Downloaded probe to {local_folder}")


def load_probe_head(
    probe_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda'
) -> Tuple[nn.Module, int]:
    """Load probe head from disk."""
    # Load probe config
    with open(probe_dir / "probe_config.json") as f:
        probe_config = json.load(f)
    
    hidden_size = probe_config['hidden_size']
    probe_layer_idx = probe_config['layer_idx']
    
    # Create probe head
    probe_head = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
    
    # Load weights
    state_dict = torch.load(
        probe_dir / "probe_head.bin",
        map_location="cpu",
        weights_only=True
    )
    probe_head.load_state_dict(state_dict)
    probe_head.eval()
    
    return probe_head, probe_layer_idx

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={VOLUME_PATH: VOLUME},
    timeout=TIMEOUT,
    allow_concurrent_inputs=10,
    secrets=[LOCAL_HF_TOKEN_SECRET],
)
class ProbeInferenceService:
    """Modal service for running hallucination probe inference with vLLM."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_MODEL
        self.llm = None
        self.tokenizer = None
        self.loaded_probes = {}  # Cache loaded probes
    
    @modal.enter()
    def load_model(self):
        """Load the vLLM model on container startup."""
        # Lazy imports to avoid issues with modal image building
        from vllm import LLM
        
        print(f"Loading vLLM model: {self.model_name}")
        
        # Set environment variable for vLLM
        os.environ["VLLM_USE_V1"] = "0"
        
        # Initialize vLLM with LoRA support and tensor parallelism
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            enable_lora=True,
            enforce_eager=True,
            download_dir=VOLUME_PATH,
            tensor_parallel_size=N_GPU,  # Use tensor parallelism across available GPUs
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN")
        )
        
        print(f"Model loaded successfully!")
    
    def _ensure_probe_downloaded(self, probe_id: str, repo_id: Optional[str] = None) -> Path:
        """Ensure probe is downloaded and return its path."""
        if not probe_id:
            raise ValueError("probe_id cannot be None or empty")
        
        # Use default repo if not provided (explicit env var no longer used)
        if not repo_id:
            repo_id = DEFAULT_PROBE_REPO
        
        # Create a unique directory name that includes repo info
        safe_repo_id = repo_id.replace("/", "_")
        probe_dir = PROBES_DIR / f"{safe_repo_id}_{probe_id}"
        
        if not probe_dir.exists():
            print(f"Downloading probe {probe_id} from {repo_id}...")

            download_probe_from_hf(
                repo_id=repo_id,
                repo_subfolder=probe_id,
                local_folder=probe_dir,
                token=os.environ.get("HF_TOKEN")
            )
        
        return probe_dir
    
    def _load_probe_if_needed(self, probe_id: str, repo_id: Optional[str] = None) -> Tuple[nn.Module, int, bool]:
        """Load probe head if not already loaded."""
        # Create a unique cache key that includes repo_id
        cache_key = f"{repo_id or 'default'}:{probe_id}"
        
        if cache_key not in self.loaded_probes:
            probe_dir = self._ensure_probe_downloaded(probe_id, repo_id)
            probe_head, probe_layer_idx = load_probe_head(probe_dir)
            
            # Check if LoRA adapters exist
            has_lora = (probe_dir / "adapter_model.safetensors").exists()
            
            self.loaded_probes[cache_key] = {
                "probe_head": probe_head,
                "probe_layer_idx": probe_layer_idx,
                "probe_dir": probe_dir,
                "has_lora": has_lora
            }
        
        probe_info = self.loaded_probes[cache_key]
        return probe_info["probe_head"], probe_info["probe_layer_idx"], probe_info["has_lora"]
    
    @modal.method()
    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different model."""
        if model_name != self.model_name:
            from vllm import LLM
            
            print(f"Switching model to: {model_name}")
            self.model_name = model_name
            
            # Reload vLLM with new model and tensor parallelism
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                trust_remote_code=True,
                enable_lora=True,
                enforce_eager=True,
                download_dir=VOLUME_PATH,
                tensor_parallel_size=N_GPU,  # Use tensor parallelism across available GPUs
            )
            
            # Reload tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.environ.get("HF_TOKEN")
            )
            
            # Clear loaded probes as they may not be compatible
            self.loaded_probes.clear()
            
            return {"status": "success", "message": f"Switched to {model_name}"}
        
        return {"status": "no_change", "message": "Model already loaded"}
    
    @modal.method()
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current model configuration."""
        return {
            "model_name": self.model_name,
            "probe_id": None,  # No default probe
        }
    
    @modal.method()
    def generate_with_probe(
        self, 
        messages: List[Dict[str, str]], 
        probe_id: str,
        repo_id: Optional[str] = None,
        threshold: float = 0.5,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a response with probe probabilities using vLLM.
        
        Args:
            messages: Conversation history
            probe_id: ID of the probe to use
            repo_id: HuggingFace repository ID (e.g., "username/repo-name"). If None, uses default
            threshold: Probability threshold for hallucination detection
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with generated token IDs, tokens, text, and probe probabilities
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        
        try:
            # Load probe if needed
            probe_head, probe_layer_idx, has_lora = self._load_probe_if_needed(probe_id, repo_id)
            cache_key = f"{repo_id or 'default'}:{probe_id}"
            probe_dir = self.loaded_probes[cache_key]["probe_dir"]
            
            # Format the conversation
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9 if temperature > 0 else 1.0
            )
            
            # Prepare generation kwargs with LoRA if available
            generate_kwargs = {'use_tqdm': False}
            if has_lora:
                generate_kwargs['lora_request'] = LoRARequest(probe_id, 1, str(probe_dir))
            
            # Get the model from vLLM
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            target_layer = model.model.layers[probe_layer_idx]
            
            # Storage for probe probabilities
            probe_probs = []
            input_probe_probs = []
            first_fwd_pass = True

            def activation_hook(module, input, output):
                nonlocal first_fwd_pass, probe_probs, input_probe_probs

                # Extract hidden states and compute probe probability
                assert len(output) == 2
                hidden_states, residual = output
                resid_post = hidden_states + residual

                with torch.no_grad():
                    probe_logits = probe_head(resid_post)
                    probs = torch.sigmoid(probe_logits).squeeze(-1)

                    if first_fwd_pass:
                        # First forward pass: capture all input token scores
                        # Handle both batch and single token cases
                        if probs.dim() == 0:
                            # Single token (shouldn't happen in prefill, but handle it)
                            input_probe_probs.append(probs.item())
                        else:
                            # Batch of tokens (typical prefill case)
                            input_probe_probs.extend(probs.cpu().tolist())
                        first_fwd_pass = False
                    else:
                        # Subsequent forward passes: single token per pass (decode)
                        probe_probs.append(probs.item())
            
            # Register hook
            hook_handle = target_layer.register_forward_hook(activation_hook)
            
            try:
                # Generate with vLLM
                outputs = self.llm.generate(
                    prompt_token_ids=[prompt_token_ids],
                    sampling_params=sampling_params,
                    **generate_kwargs
                )
                
                # Extract generated tokens
                generated_ids = list(outputs[0].outputs[0].token_ids)
                
            finally:
                # Remove hook
                hook_handle.remove()

            # Fix alignment issues if needed
            if len(probe_probs) != len(generated_ids):
                # EOS token might cause mismatch
                if len(probe_probs) + 1 == len(generated_ids):
                    probe_probs.append(0.0)  # Assign 0.0 to EOS token
            
            # Decode tokens
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Decode input tokens
            input_tokens = self.tokenizer.convert_ids_to_tokens(prompt_token_ids)

            return {
                "input_token_ids": prompt_token_ids,
                "input_tokens": input_tokens,
                "input_probe_probs": input_probe_probs,
                "generated_token_ids": generated_ids,
                "generated_tokens": generated_tokens,
                "generated_text": generated_text,
                "probe_probs": probe_probs
            }
            
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
    

@app.function(image=image)
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "hallucination-probe-backend"}


if __name__ == "__main__":
    with app.run():
        service = ProbeInferenceService()  # Uses DEFAULT_MODEL
        # Or with custom model: service = ProbeInferenceService(model_name=custom_model)
        
        result = service.generate_with_probe.remote(
            [{"role": "user", "content": "What is the capital of France?"}],
            probe_id="llama3_1_8b_lora_lambda_kl=0.5",
            repo_id="andyrdt/hallucination-probes",  # Optional custom repo
            max_tokens=100
        )
        print(result)