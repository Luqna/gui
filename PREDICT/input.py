import streamlit as st
import numpy as np
from utils.constants import METRICS

def create_input_section(container, prefix):
    with container:
        st.markdown(f'<div class="section-title">{prefix}</div>', unsafe_allow_html=True)
        
        questions = []
        metrics = METRICS[prefix]

        for metric_name, key in metrics:
            st.markdown(f'<div class="metric-label">{metric_name}</div>', unsafe_allow_html=True)
            
            value = st.slider(
                label="",
                min_value=0,
                max_value=10,
                value=5,
                key=key
            )
            
            st.markdown(f"""
                <div class="score-box">
                    {value}/10
                </div>
            """, unsafe_allow_html=True)
            
            questions.append(value)
            st.markdown("<br>", unsafe_allow_html=True)

        avg_score = np.mean(questions)
        st.markdown(f"""
            <div class="input-card">
                <div style="text-align: center;">
                    <div style="color: #666;">Average Score</div>
                    <div style="color: #1E88E5; font-size: 1.2em; font-weight: bold;">{avg_score:.1f}/10</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        return avg_score