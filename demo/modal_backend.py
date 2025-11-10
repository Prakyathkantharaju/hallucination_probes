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

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_PROBE_REPO = "Prakyathkantharaju/llama3-hallucination-probe"
N_GPU = 1
GPU_CONFIG = f"H100:{N_GPU}"
SCALEDOWN_WINDOW = 2 * 60  # 15 minutes (increased from 2)
TIMEOUT = 10 * 60  # 10 minutes

# Updated Volume configuration
VOLUME = modal.Volume.from_name("hallucination-models", create_if_missing=True)
VOLUME_PATH = "/models"
PROBES_DIR = Path(VOLUME_PATH) / "probes"

if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv("/Users/prakyath/developments/hallucination_probes/.env")
    assert os.getenv("HF_TOKEN"), "HF_TOKEN must be set to be able to load Llama models from HuggingFace"
    LOCAL_HF_TOKEN_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
else:
    LOCAL_HF_TOKEN_SECRET = modal.Secret.from_dict({})

# Updated Modal image with newer CUDA and vLLM
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "vllm==0.6.3.post1",  # More stable version
        "torch==2.4.0",
        "transformers>=4.40.0", 
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "jaxtyping>=0.2.0",
        "huggingface_hub>=0.20.0",
    )
)

app = modal.App("hallucination-probe-backend")

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
    secrets=[LOCAL_HF_TOKEN_SECRET],  # Use environment HF token
)
@modal.concurrent(max_inputs=10)
class ProbeInferenceService:
    """Modal service for running hallucination probe inference with vLLM."""
    
    @modal.enter()
    def load_model(self):
        """Load the vLLM model on container startup."""
        from vllm import LLM
        
        # Initialize instance variables
        self.model_name = DEFAULT_MODEL
        self.llm = None
        self.tokenizer = None
        self.loaded_probes = {}
        
        print(f"Loading vLLM model: {self.model_name}")
        
        # Initialize vLLM with LoRA support
        # NOTE: enforce_eager=True disables CUDA graphs to allow PyTorch hooks to work
        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=64,
            download_dir=VOLUME_PATH,
            tensor_parallel_size=N_GPU,
            dtype="bfloat16",
            enforce_eager=True,  # Required for hooks to work with probes
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
            
            # Reload vLLM with new model
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.85,
                max_model_len=4096,
                enable_lora=True,
                max_loras=4,
                max_lora_rank=64,
                download_dir=VOLUME_PATH,
                tensor_parallel_size=N_GPU,
                dtype="bfloat16",
                enforce_eager=True,  # Required for hooks to work with probes
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
        temperature: float = 0.7,
        only_prefill: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response with probe probabilities using vLLM.
        
        Args:
            messages: Conversation history
            probe_id: ID of the probe to use (e.g., "clean_code_llama3_1_8b_lora")
            repo_id: HuggingFace repository ID. Defaults to "Prakyathkantharaju/llama3-hallucination-probe"
            threshold: Probability threshold for hallucination detection
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            only_prefill: If True, only compute prefill probabilities without generation
            
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
            
            # Tokenize just the last message to know its length
            # (for extracting only last message prefill probabilities)
            if messages:
                last_message = messages[-1]
                last_message_tokens = self.tokenizer.apply_chat_template(
                    [last_message],
                    tokenize=True,
                    add_generation_prompt=True
                )
                last_message_length = len(last_message_tokens)
            else:
                last_message_length = 0
            
            # Handle prefill-only mode
            if only_prefill:
                return self._compute_prefill_only(
                    prompt_token_ids,
                    last_message_length,
                    probe_head,
                    probe_layer_idx,
                    has_lora,
                    probe_dir,
                    probe_id
                )
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9 if temperature > 0 else 1.0,
                skip_special_tokens=False,  # Important for probe alignment
            )
            
            # Prepare generation kwargs with LoRA if available
            generate_kwargs = {}
            if has_lora:
                generate_kwargs['lora_request'] = LoRARequest(
                    lora_name=probe_id,
                    lora_int_id=1,
                    lora_path=str(probe_dir)
                )
            
            # Get the model from vLLM (API may vary by version)
            try:
                model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            except AttributeError:
                # Fallback for different vLLM API structure
                model = self.llm.llm_engine.workers[0].model_runner.model
            
            target_layer = model.model.layers[probe_layer_idx]
            
            # Storage for probe probabilities
            prefill_probs = []
            decode_probs = []
            prompt_length = len(prompt_token_ids)
            is_prefill_pass = True
            
            def activation_hook(module, input, output):
                nonlocal is_prefill_pass, prefill_probs, decode_probs
                
                # Extract hidden states and compute probe probability
                if isinstance(output, tuple) and len(output) == 2:
                    hidden_states, residual = output
                    resid_post = hidden_states + residual
                else:
                    resid_post = output
                
                with torch.no_grad():
                    if is_prefill_pass:
                        # Prefill: compute probs for all input tokens
                        print(f"Prefill pass: processing {resid_post.shape[0]} tokens")
                        for token_idx in range(resid_post.shape[0]):
                            probe_logits = probe_head(resid_post[token_idx:token_idx+1])
                            prob = torch.sigmoid(probe_logits).squeeze(-1).cpu().item()
                            prefill_probs.append(prob)
                        is_prefill_pass = False
                        print(f"Prefill complete: collected {len(prefill_probs)} probabilities")
                    else:
                        # Decode: compute prob for last token only
                        probe_logits = probe_head(resid_post[-1:])
                        prob = torch.sigmoid(probe_logits).squeeze(-1).cpu().item()
                        decode_probs.append(prob)
                        print(f"Decode step {len(decode_probs)}: prob = {prob:.4f}")
            
            # Register hook
            hook_handle = target_layer.register_forward_hook(activation_hook)
            
            try:
                print(f"Starting generation with prompt length: {prompt_length}")
                print(f"Sampling params: max_tokens={max_tokens}, temperature={temperature}")
                
                # Generate with vLLM
                outputs = self.llm.generate(
                    prompt_token_ids=[prompt_token_ids],
                    sampling_params=sampling_params,
                    **generate_kwargs
                )
                
                print(f"Generation complete. Number of outputs: {len(outputs)}")
                print(f"Collected {len(decode_probs)} decode probabilities")
                
                # Extract generated tokens
                generated_ids = list(outputs[0].outputs[0].token_ids)
                print(f"Generated {len(generated_ids)} token IDs")
                
            finally:
                # Remove hook
                hook_handle.remove()

            # Fix alignment issues if needed
            print(f"Before alignment: {len(decode_probs)} probs, {len(generated_ids)} tokens")
            if len(decode_probs) != len(generated_ids):
                # Handle EOS token or other alignment issues
                if len(decode_probs) + 1 == len(generated_ids):
                    print("Adding 0.0 for EOS token")
                    decode_probs.append(0.0)  # Assign 0.0 to EOS token
                elif len(decode_probs) > len(generated_ids):
                    print(f"Truncating probs from {len(decode_probs)} to {len(generated_ids)}")
                    decode_probs = decode_probs[:len(generated_ids)]
                else:
                    print(f"WARNING: Mismatch - {len(decode_probs)} probs vs {len(generated_ids)} tokens")
                    # Pad with zeros if we have fewer probs than tokens
                    while len(decode_probs) < len(generated_ids):
                        decode_probs.append(0.0)
            
            # Decode tokens
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract only the last message tokens and probabilities
            # Slice from the end to get the last message_length tokens
            if last_message_length > 0 and last_message_length <= len(prompt_token_ids):
                last_msg_token_ids = prompt_token_ids[-last_message_length:]
                last_msg_probs = prefill_probs[-last_message_length:] if len(prefill_probs) >= last_message_length else prefill_probs
                prefill_tokens = self.tokenizer.convert_ids_to_tokens(last_msg_token_ids)
                prefill_probs = last_msg_probs
            else:
                # Fallback: use all tokens if we can't determine last message length
                prefill_tokens = self.tokenizer.convert_ids_to_tokens(prompt_token_ids)
            
            print(f"Generated text: {generated_text}")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Decode probs ({len(decode_probs)}): {decode_probs}")
            print(f"Last message length: {last_message_length}")
            print(f"Prefill tokens count (last message only): {len(prefill_tokens)}")
            print(f"Prefill probs count (last message only): {len(prefill_probs)}")
            
            return {
                "generated_token_ids": generated_ids,
                "generated_tokens": generated_tokens,
                "generated_text": generated_text,
                "probe_probs": decode_probs,
                "prefill_tokens": prefill_tokens,
                "prefill_probs": prefill_probs,
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            return {
                "error": f"Generation failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "status": "error"
            }
    
    def _compute_prefill_only(
        self,
        prompt_token_ids: List[int],
        last_message_length: int,
        probe_head: nn.Module,
        probe_layer_idx: int,
        has_lora: bool,
        probe_dir: Path,
        probe_id: str
    ) -> Dict[str, Any]:
        """Compute only prefill probabilities without generation."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        
        try:
            # Set up sampling parameters to generate 0 tokens (prefill only)
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                max_tokens=1,  # Generate just 1 token to trigger prefill
                top_p=1.0,
                skip_special_tokens=False,
            )
            
            # Prepare generation kwargs with LoRA if available
            generate_kwargs = {}
            if has_lora:
                generate_kwargs['lora_request'] = LoRARequest(
                    lora_name=probe_id,
                    lora_int_id=1,
                    lora_path=str(probe_dir)
                )
            
            # Get the model from vLLM
            try:
                model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            except AttributeError:
                model = self.llm.llm_engine.workers[0].model_runner.model
            
            target_layer = model.model.layers[probe_layer_idx]
            
            # Storage for probe probabilities
            prefill_probs = []
            is_prefill_pass = True
            
            def activation_hook(module, input, output):
                nonlocal is_prefill_pass, prefill_probs
                
                # Extract hidden states and compute probe probability
                if isinstance(output, tuple) and len(output) == 2:
                    hidden_states, residual = output
                    resid_post = hidden_states + residual
                else:
                    resid_post = output
                
                with torch.no_grad():
                    if is_prefill_pass:
                        # Prefill: compute probs for all input tokens
                        print(f"Prefill pass (prefill-only mode): processing {resid_post.shape[0]} tokens")
                        for token_idx in range(resid_post.shape[0]):
                            probe_logits = probe_head(resid_post[token_idx:token_idx+1])
                            prob = torch.sigmoid(probe_logits).squeeze(-1).cpu().item()
                            prefill_probs.append(prob)
                        is_prefill_pass = False
                        print(f"Prefill complete: collected {len(prefill_probs)} probabilities")
            
            # Register hook
            hook_handle = target_layer.register_forward_hook(activation_hook)
            
            try:
                print(f"Starting prefill-only computation with prompt length: {len(prompt_token_ids)}")
                
                # Generate with vLLM (will only do prefill + 1 token)
                outputs = self.llm.generate(
                    prompt_token_ids=[prompt_token_ids],
                    sampling_params=sampling_params,
                    **generate_kwargs
                )
                
                print(f"Prefill computation complete.")
                
            finally:
                # Remove hook
                hook_handle.remove()
            
            # Extract only the last message tokens and probabilities
            if last_message_length > 0 and last_message_length <= len(prompt_token_ids):
                last_msg_token_ids = prompt_token_ids[-last_message_length:]
                last_msg_probs = prefill_probs[-last_message_length:] if len(prefill_probs) >= last_message_length else prefill_probs
                prefill_tokens = self.tokenizer.convert_ids_to_tokens(last_msg_token_ids)
                prefill_probs = last_msg_probs
            else:
                # Fallback: use all tokens if we can't determine last message length
                prefill_tokens = self.tokenizer.convert_ids_to_tokens(prompt_token_ids)
            
            print(f"Last message length: {last_message_length}")
            print(f"Prefill tokens count (last message only): {len(prefill_tokens)}")
            print(f"Prefill probs count (last message only): {len(prefill_probs)}")
            
            return {
                "prefill_tokens": prefill_tokens,
                "prefill_probs": prefill_probs,
                "generated_token_ids": [],
                "generated_tokens": [],
                "generated_text": "",
                "probe_probs": [],
                "only_prefill": True,
                "status": "success"
            }
            
        except Exception as e:
            import traceback
            return {
                "error": f"Prefill computation failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "status": "error"
            }

@app.function(image=image)
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "hallucination-probe-backend"}


if __name__ == "__main__":
    with app.run():
        service = ProbeInferenceService()
        
        result = service.generate_with_probe.remote(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            probe_id="clean_code_llama3_1_8b_lora",
            repo_id="Prakyathkantharaju/llama3-hallucination-probe",
            max_tokens=100
        )
        print(result)