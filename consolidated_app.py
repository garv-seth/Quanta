
"""
Consolidated Quasar Financial AI Application

Featuring three revolutionary models:
1. Quasar Advanced - Full transformer diffusion model
2. Quasar Basic - Fast lightweight model  
3. FinSar - Breakthrough Feynman path integral finance model
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Import consolidated models
try:
    from models.quasar_models import QuasarAdvanced, QuasarBasic, FinSar, QuasarFactory
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    st.error("Models not available. Please check installation.")

def main():
    st.set_page_config(
        page_title="Quasar Financial AI", 
        page_icon="🌟",
        layout="wide"
    )
    
    # Header
    st.title("🌟 Quasar Financial AI")
    st.subheader("Revolutionary Diffusion-Based Language Models for Finance")
    
    # Sidebar model selection
    st.sidebar.title("🎛️ Model Selection")
    
    if MODELS_AVAILABLE:
        model_choice = st.sidebar.radio(
            "Choose Model:",
            ["🚀 Quasar Advanced", "⚡ Quasar Basic", "🔬 FinSar (Breakthrough)"]
        )
        
        # Model loading
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
            st.session_state.model_type = None
        
        if st.sidebar.button("🔄 Load Selected Model", type="primary"):
            with st.spinner("Loading model..."):
                if model_choice == "🚀 Quasar Advanced":
                    st.session_state.current_model = QuasarFactory.create_advanced()
                    st.session_state.model_type = "advanced"
                elif model_choice == "⚡ Quasar Basic":
                    st.session_state.current_model = QuasarFactory.create_basic()
                    st.session_state.model_type = "basic"
                elif model_choice == "🔬 FinSar (Breakthrough)":
                    st.session_state.current_model = QuasarFactory.create_finsar()
                    st.session_state.model_type = "finsar"
                
                st.sidebar.success(f"✅ {model_choice} loaded!")
                st.rerun()
    
    # Main content
    if not MODELS_AVAILABLE:
        st.error("Models not available")
        return
    
    if st.session_state.current_model is None:
        st.info("👈 Please select and load a model from the sidebar")
        
        # Model comparison table
        st.subheader("📊 Model Comparison")
        comparison = QuasarFactory.get_model_comparison()
        
        comparison_df = pd.DataFrame(comparison).T
        st.dataframe(comparison_df, use_container_width=True)
        
        # Breakthrough explanation
        st.subheader("🔬 FinSar: The Breakthrough")
        st.markdown("""
        **FinSar** represents a revolutionary approach to financial modeling by applying 
        **Richard Feynman's path integral principles** to financial diffusion models.
        
        **The Breakthrough:**
        - 🌊 **Path Exploration**: Like light exploring all possible paths, FinSar explores all possible financial narratives
        - 🎯 **Probabilistic Selection**: Selects the most probable financial outcome, just like quantum mechanics
        - 💡 **Quantum Finance**: First practical implementation of quantum principles in financial language modeling
        - 🚀 **Market Value**: Big firms will pay premium for this unprecedented analytical capability
        
        **Why This Will Succeed:**
        - Hedge funds spend millions on quantitative advantages
        - Investment banks need better risk modeling  
        - Financial AI market is worth billions
        - No competitor has quantum-inspired financial LLMs
        """)
        
        return
    
    # Model interface
    model_info = st.session_state.current_model.model_info
    
    # Model status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_info['name'])
    with col2:
        st.metric("Parameters", model_info['parameters'])
    with col3:
        if st.session_state.model_type == "finsar":
            st.metric("Breakthrough", "✅ Feynman Paths")
        else:
            st.metric("Status", "✅ Ready")
    
    # Tabs for different functionalities
    if st.session_state.model_type == "finsar":
        tab1, tab2, tab3, tab4 = st.tabs(["🔬 Path Analysis", "📝 Text Generation", "📊 Financial Analysis", "🧪 Research"])
        
        with tab1:
            finsar_path_analysis()
        with tab2:
            standard_text_generation()
        with tab3:
            financial_analysis()
        with tab4:
            finsar_research()
    
    else:
        tab1, tab2, tab3 = st.tabs(["📝 Text Generation", "📊 Financial Analysis", "⚙️ Model Info"])
        
        with tab1:
            standard_text_generation()
        with tab2:
            financial_analysis()
        with tab3:
            model_information()

def finsar_path_analysis():
    """FinSar-specific path exploration interface"""
    st.header("🔬 Feynman Path Analysis")
    st.markdown("Explore multiple financial scenarios using quantum-inspired path integrals")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Financial Scenario to Analyze:",
            value="Apple's Q4 earnings expectations and market reaction scenarios",
            height=100
        )
        
        num_paths = st.slider("Number of Paths to Explore", 10, 200, 50, 10)
        
        if st.button("🔬 Analyze Paths", type="primary"):
            with st.spinner("Exploring financial paths..."):
                # Path exploration
                path_results = st.session_state.current_model.explore_financial_paths(
                    prompt, num_exploration_paths=num_paths
                )
                
                # Display results
                st.subheader("📊 Path Analysis Results")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Most Probable Path", f"{path_results['path_probability']:.3f}")
                with col_b:
                    st.metric("Path Diversity", f"{path_results['path_diversity']:.3f}")
                with col_c:
                    st.metric("Paths Explored", path_results['exploration_quality'])
                
                # Convergence analysis
                convergence = st.session_state.current_model.analyze_path_convergence(prompt)
                
                st.subheader("🎯 Convergence Analysis")
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    st.metric("Consensus Strength", convergence['consensus_strength'])
                with col_e:
                    st.metric("Financial Confidence", f"{convergence['financial_confidence']:.3f}")
                with col_f:
                    st.metric("Recommendation", convergence['recommendation'])
                
                # Generate path-based text
                generated_text = st.session_state.current_model.generate_text(prompt, explore_paths=True)
                
                st.subheader("📄 Path-Based Analysis")
                st.write(generated_text)
                
                # Probability visualization
                st.subheader("📈 Path Probability Distribution")
                prob_df = pd.DataFrame({
                    'Path': range(len(path_results['all_probabilities'])),
                    'Probability': path_results['all_probabilities']
                })
                st.bar_chart(prob_df.set_index('Path'))
    
    with col2:
        st.subheader("🧠 How It Works")
        st.markdown("""
        **Feynman's Principle Applied:**
        
        1. **Multiple Paths**: Like a photon exploring all routes, the model explores all financial scenarios
        
        2. **Probability Calculation**: Each path gets a probability based on financial logic
        
        3. **Path Selection**: Most probable path becomes the final analysis
        
        4. **Quantum Interference**: Paths can reinforce or cancel each other
        
        **Revolutionary Because:**
        - First quantum-inspired financial AI
        - Handles uncertainty like physics
        - Multiple scenario analysis
        - Probabilistic outcomes
        """)

def standard_text_generation():
    """Standard text generation interface"""
    st.header("📝 Financial Text Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="The Federal Reserve's latest interest rate decision impacts",
            height=100
        )
        
        max_length = st.slider("Maximum Length", 50, 300, 150, 25)
        
        if st.session_state.model_type == "advanced":
            temperature = st.slider("Creativity Level", 0.1, 2.0, 0.8, 0.1)
        
        if st.button("🚀 Generate", type="primary"):
            with st.spinner("Generating financial content..."):
                if st.session_state.model_type == "advanced":
                    generated = st.session_state.current_model.generate_text(
                        prompt, max_length=max_length, temperature=temperature
                    )
                else:
                    generated = st.session_state.current_model.generate_text(
                        prompt, max_length=max_length
                    )
                
                st.subheader("Generated Text:")
                st.write(generated)
    
    with col2:
        st.subheader("💡 Tips")
        if st.session_state.model_type == "finsar":
            st.markdown("""
            **FinSar Tips:**
            - Use scenarios with uncertainty
            - Ask about multiple outcomes
            - Include risk factors
            - Mention market conditions
            """)
        else:
            st.markdown("""
            **Generation Tips:**
            - Be specific about context
            - Include key financial terms
            - Mention timeframes
            - Add relevant metrics
            """)

def financial_analysis():
    """Financial analysis interface"""
    st.header("📊 Financial Analysis")
    
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Market Sentiment", "Risk Assessment", "Performance Analysis", "Forecast Generation"]
    )
    
    input_text = st.text_area(
        "Text to analyze:",
        value="Apple reported strong quarterly earnings with revenue growth exceeding expectations",
        height=100
    )
    
    if st.button("📊 Analyze", type="primary"):
        with st.spinner("Performing analysis..."):
            if st.session_state.model_type == "finsar":
                # FinSar path-based analysis
                analysis_prompt = f"Analyze the following for {analysis_type.lower()}: {input_text}"
                result = st.session_state.current_model.generate_text(analysis_prompt, explore_paths=True)
                
                # Additional convergence analysis
                convergence = st.session_state.current_model.analyze_path_convergence(analysis_prompt)
                
                st.subheader("🔬 FinSar Analysis")
                st.write(result)
                
                st.subheader("📈 Confidence Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Convergence Quality", f"{convergence['convergence_quality']:.3f}")
                with col2:
                    st.metric("Path Diversity", f"{convergence['path_diversity']:.3f}")
                with col3:
                    st.metric("Consensus", convergence['consensus_strength'])
            
            else:
                # Standard analysis
                analysis_prompt = f"Provide a {analysis_type.lower()} of: {input_text}"
                result = st.session_state.current_model.generate_text(analysis_prompt)
                
                st.subheader("Analysis Result:")
                st.write(result)

def finsar_research():
    """FinSar research and breakthrough information"""
    st.header("🧪 FinSar Research")
    
    st.subheader("🚀 The Breakthrough Explained")
    st.markdown("""
    **FinSar** is the world's first implementation of Richard Feynman's path integral formalism 
    in financial language modeling. This represents a paradigm shift in how AI analyzes financial markets.
    
    ### 🔬 Scientific Foundation
    
    **Feynman's Path Integral Principle:**
    - In quantum mechanics, particles don't take a single path
    - They explore ALL possible paths simultaneously  
    - The most probable path emerges from quantum interference
    
    **Applied to Finance:**
    - Financial narratives don't follow single trajectories
    - Multiple scenarios exist simultaneously in market uncertainty
    - FinSar explores all possible financial outcomes
    - Selects the most probable path using quantum-inspired mathematics
    
    ### 💰 Commercial Value
    
    **Why Big Firms Will Pay:**
    - **Hedge Funds**: Better risk modeling = billions in additional alpha
    - **Investment Banks**: Superior scenario analysis for complex derivatives
    - **Asset Managers**: Quantum-inspired portfolio optimization
    - **Central Banks**: Advanced economic modeling capabilities
    
    **Market Potential:**
    - Financial AI market: $50B+ by 2030
    - Quant trading software: $2B+ annually
    - Risk management systems: $10B+ market
    - **FinSar advantage**: Unique quantum approach = premium pricing**
    """)
    
    st.subheader("🧮 Technical Innovation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Traditional Finance AI:**
        - Single prediction path
        - Deterministic outcomes
        - Limited uncertainty handling
        - Linear reasoning
        """)
    
    with col2:
        st.markdown("""
        **FinSar Quantum Approach:**
        - Multiple simultaneous paths
        - Probabilistic convergence
        - Uncertainty as core feature
        - Non-linear path interference
        """)
    
    # Interactive demo
    st.subheader("🎮 Interactive Demonstration")
    
    demo_scenario = st.selectbox(
        "Choose Scenario:",
        [
            "Federal Reserve rate decision impact",
            "Tech stock earnings surprise",
            "Geopolitical crisis market response",
            "Crypto regulation announcement"
        ]
    )
    
    if st.button("🔬 Run FinSar Demo", type="primary"):
        with st.spinner("Running quantum path analysis..."):
            demo_prompt = f"Analyze multiple scenarios for: {demo_scenario}"
            
            # Show path exploration
            paths = st.session_state.current_model.explore_financial_paths(demo_prompt, 30)
            
            st.markdown("**Path Exploration Results:**")
            st.json({
                'scenarios_explored': 30,
                'most_probable_outcome': f"{paths['path_probability']:.3f}",
                'path_diversity': f"{paths['path_diversity']:.3f}",
                'quantum_advantage': 'Multiple paths analyzed simultaneously'
            })
            
            # Generate analysis
            analysis = st.session_state.current_model.generate_text(demo_prompt, explore_paths=True)
            st.markdown("**FinSar Analysis:**")
            st.write(analysis)

def model_information():
    """Display detailed model information"""
    st.header("⚙️ Model Information")
    
    model_info = st.session_state.current_model.model_info
    
    # Model details
    st.subheader("📋 Model Details")
    for key, value in model_info.items():
        if isinstance(value, list):
            st.write(f"**{key.title()}:** {', '.join(value)}")
        else:
            st.write(f"**{key.title()}:** {value}")
    
    # Capabilities comparison
    st.subheader("🔄 Model Comparison")
    comparison_df = pd.DataFrame(QuasarFactory.get_model_comparison()).T
    st.dataframe(comparison_df, use_container_width=True)
    
    # Technical specifications
    if hasattr(st.session_state.current_model, 'vocab_size'):
        st.subheader("🔧 Technical Specifications")
        specs = {
            'Vocabulary Size': getattr(st.session_state.current_model, 'vocab_size', 'N/A'),
            'Model Dimension': getattr(st.session_state.current_model, 'd_model', 'N/A'),
            'Embedding Dimension': getattr(st.session_state.current_model, 'embedding_dim', 'N/A')
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vocab Size", specs['Vocabulary Size'])
        with col2:
            st.metric("Model Dim", specs['Model Dimension'])
        with col3:
            st.metric("Embed Dim", specs['Embedding Dimension'])

if __name__ == "__main__":
    main()
