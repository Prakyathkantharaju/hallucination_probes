"""
Streamlit interface for interactive hallucination detection using probes.
Connects to Modal backend for fast inference.
"""

import streamlit as st
import modal
import json
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from conversation_renderer import conversation_builder_ui

# Page configuration
st.set_page_config(
    page_title="Hallucination Detection Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modal app connection
@st.cache_resource
def get_modal_app():
    """Get the Modal app connection."""
    try:
        return modal.App.lookup("hallucination-probe-backend", create_if_missing=False)
    except Exception as e:
        st.error(f"Failed to connect to Modal backend: {e}")
        return None

def get_probe_service():
    """Get the probe inference service."""
    app = get_modal_app()
    if app is None:
        return None
    cls = modal.Cls.from_name("hallucination-probe-backend", "ProbeInferenceService")
    probe_service = cls()
    return probe_service

def colorize_tokens(tokens: List[str], probabilities: List[float], predictions: List[int], threshold: float) -> str:
    """Create HTML with color-coded tokens based on hallucination probabilities."""
    html_parts = []
    
    for token, prob, pred in zip(tokens, probabilities, predictions):
        # Clean up token display (remove special characters)
        display_token = token.replace('▁', ' ').replace('Ġ', ' ')
        if display_token.startswith('##'):
            display_token = display_token[2:]
        
        # Color intensity based on probability
        if pred == 1:  # Flagged as hallucination
            red_intensity = min(255, int(150 + (prob - threshold) / (1 - threshold) * 105))
            color = f"rgb({red_intensity}, 100, 100)"
            style = "font-weight: bold;"
        else:
            # Subtle green for non-hallucination
            green_intensity = min(255, int(200 + (1 - prob) * 55))
            color = f"rgb(100, {green_intensity}, 100)"
            style = ""
        
        html_parts.append(
            f'<span style="background-color: {color}; padding: 1px 2px; '
            f'border-radius: 3px; margin: 1px; {style}" '
            f'title="Probability: {prob:.3f}">{"" if display_token.startswith(" ") else ""}{display_token}</span>'
        )
    
    return "".join(html_parts)


def main():
    load_dotenv()
    st.markdown("#### 🔍 Token-level Hallucination Probes")
    
    # Probe service connection
    probe_service = get_probe_service()
    if probe_service is None:
        st.error("❌ Cannot connect to Modal backend")
        st.stop()
    
    # Get current configuration
    try:
        current_config = probe_service.get_current_config.remote()
        st.success("✅ Connected to backend")
    except Exception as e:
        st.error(f"❌ Failed to get current config: {e}")
        st.stop()
    
    # Probe Configuration - simplified without box
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        model_name = st.selectbox(
            "Model",
            ["meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"],
            index=0 if current_config.get('model_name') == "meta-llama/Meta-Llama-3.1-8B-Instruct" else 1
        )
    
    with col2:
        probe_id = st.text_input("Probe ID", value=current_config.get('probe_id', 'clean_code_llama3_1_8b_lora'))
    
    with col3:
        repo_id = st.text_input("Repository ID (optional)", value="", placeholder="Prakyathkantharaju/llama3-hallucination-probe", 
                               help="HuggingFace repository ID. Leave empty to use Prakyathkantharaju/llama3-hallucination-probe")
    
    # Initialize parameters for chat modes
    threshold = 0.6  # Fixed threshold
    max_tokens = 1024
    temperature = 0.7
    
    # Use the conversation builder with parameters
    input_method, messages = conversation_builder_ui(
        probe_service=probe_service,
        probe_id=probe_id,
        repo_id=repo_id if repo_id else None,
        threshold=threshold,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Analysis button (only for non-chat modes)
    if input_method != "Chat with Model":
        if st.button("🔍 Analyze Conversation", type="primary"):
            if not messages:
                st.warning("Please create a conversation to analyze.")
                return
        
            with st.spinner("Analyzing conversation..."):
                try:
                    # Get prediction from Modal backend
                    result = probe_service.predict_conversation.remote(messages, threshold)
                    
                    if "error" in result:
                        st.error(f"Analysis failed: {result['error']}")
                        return
                    
                    # Display results
                    st.header("Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tokens", result["total_tokens"])
                    with col2:
                        st.metric("Flagged Tokens", result["num_flagged"])
                    with col3:
                        flagged_percentage = (result["num_flagged"] / result["total_tokens"]) * 100 if result["total_tokens"] > 0 else 0
                        st.metric("Flagged %", f"{flagged_percentage:.1f}%")
                    with col4:
                        avg_prob = np.mean(result["probabilities"]) if result["probabilities"] else 0
                        st.metric("Avg Probability", f"{avg_prob:.3f}")
                    
                    # Color-coded conversation
                    st.subheader("Token-level Analysis")
                    st.markdown("**Legend:** 🟢 Low hallucination probability | 🔴 High hallucination probability")
                    
                    colored_html = colorize_tokens(
                        result["tokens"], 
                        result["probabilities"], 
                        result["predictions"],
                        threshold
                    )
                    st.components.v1.html(
                        f'<div style="font-family: monospace; font-size: 14px; line-height: 1.6; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">{colored_html}</div>',
                        height=300,
                        scrolling=True
                    )
                
                    
                    # Detailed token analysis
                    with st.expander("Detailed Token Analysis"):
                        df = pd.DataFrame({
                            'Token': result["tokens"],
                            'Probability': result["probabilities"],
                            'Flagged': result["predictions"],
                            'Position': range(len(result["tokens"]))
                        })
                        
                        # Filter options
                        show_all = st.checkbox("Show all tokens", value=True)
                        if not show_all:
                            df = df[df['Flagged'] == 1]
                        
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "Probability": st.column_config.ProgressColumn(
                                    "Probability",
                                    help="Hallucination probability",
                                    min_value=0,
                                    max_value=1,
                                    format="%.3f"
                                ),
                                "Flagged": st.column_config.CheckboxColumn(
                                    "Flagged",
                                    help="Flagged as hallucination"
                                )
                            }
                        )
                    
                    # Download results
                    st.subheader("Export Results")
                    results_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=results_json,
                        file_name="hallucination_analysis.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()
