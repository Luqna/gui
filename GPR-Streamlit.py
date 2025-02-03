import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
import time

def create_features(input_data):
    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])
    else:
        data = input_data.copy()
    
    X = pd.DataFrame({
        'ServiceQuality': [data['ServiceQuality']],
        'Price': [data['Price']],
        'Innovation': [data['Innovation']]
    })
    
    return X

# Only showing the modified get_detailed_insights function - rest of code remains the same

def get_detailed_insights(sq_scores, p_scores, i_scores, category_scores):
    insights = []
    
    # Get individual scores
    sq_details = {
        'Driver Quality': sq_scores[0],
        'UI Experience': sq_scores[1],
        'Vehicle Quality': sq_scores[2]
    }
    
    p_details = {
        'Affordability': p_scores[0],
        'Price Transparency': p_scores[1],
        'Overall Satisfaction': p_scores[2]
    }
    
    i_details = {
        'Service Development': i_scores[0],
        'Ease of Use': i_scores[1],
        'Response Time': i_scores[2]
    }
    
    # Service Quality Insights
    sq_sorted = sorted(sq_details.items(), key=lambda x: x[1])
    insights.append("🔍 **Service Quality Analysis:**")
    
    # Weakest areas
    if sq_sorted[0][1] < 6:
        insights.append(f"• Area for improvement: {sq_sorted[0][0]} (Score: {sq_sorted[0][1]:.1f}/10)")
        if sq_sorted[0][0] == 'Driver Quality':
            insights.append("  - Consider implementing enhanced driver training programs")
            insights.append("  - Develop regular performance evaluation systems")
            insights.append("  - Introduce driver incentive programs for high ratings")
        elif sq_sorted[0][0] == 'UI Experience':
            insights.append("  - Consider user interface redesign based on user feedback")
            insights.append("  - Implement A/B testing for new features")
            insights.append("  - Enhance app responsiveness and intuitiveness")
        else:  # Vehicle Quality
            insights.append("  - Review and upgrade vehicle maintenance protocols")
            insights.append("  - Consider fleet modernization where necessary")
            insights.append("  - Implement stricter vehicle quality checks")
    
    # Strong areas
    if sq_sorted[2][1] >= 7:
        insights.append(f"• Strong performance: {sq_sorted[2][0]} (Score: {sq_sorted[2][1]:.1f}/10)")
        insights.append("  - Document and share best practices from this area")
        insights.append("  - Consider using this strength in marketing materials")
    
    # Price Insights
    p_sorted = sorted(p_details.items(), key=lambda x: x[1])
    insights.append("\n💰 **Price Strategy Analysis:**")
    
    # Weakest areas
    if p_sorted[0][1] < 6:
        insights.append(f"• Area for improvement: {p_sorted[0][0]} (Score: {p_sorted[0][1]:.1f}/10)")
        if p_sorted[0][0] == 'Affordability':
            insights.append("  - Consider introducing flexible pricing tiers")
            insights.append("  - Analyze competitor pricing strategies")
            insights.append("  - Explore loyalty discount programs")
        elif p_sorted[0][0] == 'Price Transparency':
            insights.append("  - Enhance fare breakdown visibility")
            insights.append("  - Provide clearer surge pricing notifications")
            insights.append("  - Implement predictive fare estimates")
        else:  # Overall Satisfaction
            insights.append("  - Conduct customer surveys for pricing feedback")
            insights.append("  - Consider value-added services")
            insights.append("  - Review overall pricing strategy")
    
    # Strong areas
    if p_sorted[2][1] >= 7:
        insights.append(f"• Strong performance: {p_sorted[2][0]} (Score: {p_sorted[2][1]:.1f}/10)")
        insights.append("  - Continue successful pricing practices")
        insights.append("  - Share pricing strategy success with stakeholders")
    
    # Innovation Insights
    i_sorted = sorted(i_details.items(), key=lambda x: x[1])
    insights.append("\n💡 **Innovation Strategy Analysis:**")
    
    # Weakest areas
    if i_sorted[0][1] < 6:
        insights.append(f"• Area for improvement: {i_sorted[0][0]} (Score: {i_sorted[0][1]:.1f}/10)")
        if i_sorted[0][0] == 'Service Development':
            insights.append("  - Accelerate new feature development cycle")
            insights.append("  - Increase focus on user feedback implementation")
            insights.append("  - Consider beta testing programs")
        elif i_sorted[0][0] == 'Ease of Use':
            insights.append("  - Conduct usability testing")
            insights.append("  - Simplify user workflows")
            insights.append("  - Implement user behavior analytics")
        else:  # Response Time
            insights.append("  - Optimize system response times")
            insights.append("  - Enhance server infrastructure")
            insights.append("  - Implement performance monitoring")
    
    # Strong areas
    if i_sorted[2][1] >= 7:
        insights.append(f"• Strong performance: {i_sorted[2][0]} (Score: {i_sorted[2][1]:.1f}/10)")
        insights.append("  - Document innovative successes")
        insights.append("  - Build on current technological advantages")
    
    # Overall Excellence Check
    all_high = all(score >= 7 for score in category_scores.values())
    if all_high:
        insights.append("\n🌟 **Overall Excellence Strategy:**")
        insights.append("• Maintain current high performance levels:")
        insights.append("  - Document and share successful practices across all departments")
        insights.append("  - Develop case studies of successful implementations")
        insights.append("  - Consider implementing a knowledge sharing system")
        insights.append("  - Set new benchmark targets for continuous improvement")
    
    return insights

def create_radar_plot(sq_scores, p_scores, i_scores):
    categories = ['Driver Quality', 'UI Experience', 'Vehicle Quality',
                 'Affordability', 'Price Transparency', 'Price Satisfaction',
                 'Service Development', 'Ease of Use', 'Response Time']
    
    values = sq_scores + p_scores + i_scores + [sq_scores[0]]  # Add first value to close the loop
    categories = categories + [categories[0]]  # Add first category to close the loop
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Metrics',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False,
        title="Detailed Metrics Analysis"
    )
    
    return fig

def create_input_section(container, prefix):
    with container:
        st.markdown(f'### {prefix}')
        
        questions = []
        
        if prefix == 'Service Quality':
            metrics = [
                ('Driver Quality', 'sq_1'),
                ('UI Experience', 'sq_2'),
                ('Vehicle Quality', 'sq_3')
            ]
        elif prefix == 'Price':
            metrics = [
                ('Affordability', 'p_1'),
                ('Price Transparency', 'p_2'),
                ('Overall Satisfaction', 'p_3')
            ]
        else:  # Innovation
            metrics = [
                ('Service Development', 'i_1'),
                ('Ease of Use', 'i_2'),
                ('Response Time', 'i_3')
            ]

        for metric_name, key in metrics:
            value = st.slider(
                label=metric_name,
                min_value=0,
                max_value=10,
                value=5,
                key=key
            )
            questions.append(value)

        return np.mean(questions)

def create_trend_comparison(prediction, service_quality, price, innovation):
    categories = ['Overall Loyalty', 'Service Quality', 'Price', 'Innovation']
    values = [prediction, service_quality, price, innovation]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    fig = go.Figure()
    
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            name=cat,
            y=[val],
            x=[cat],
            marker_color=color,
            text=f'{val:.1f}',
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Metrics Comparison",
        yaxis_range=[0, 10],
        yaxis_title="Score",
        showlegend=False,
        height=400
    )
    
    return fig

def get_loyalty_level(score):
    if score <= 4:
        return {
            'level': "Low",
            'icon': "🔴",
            'color': "red",
            'description': "Significant improvement needed. High risk of customer churn.",
            'background': "#ffebee"
        }
    elif score <= 6:
        return {
            'level': "Medium",
            'icon': "🟡",
            'color': "orange",
            'description': "Room for improvement. Moderate customer satisfaction.",
            'background': "#fff3e0"
        }
    elif score <= 8:
        return {
            'level': "High",
            'icon': "🟢",
            'color': "green",
            'description': "Strong performance. Good customer satisfaction.",
            'background': "#e8f5e9"
        }
    else:
        return {
            'level': "Very High",
            'icon': "💫",
            'color': "blue",
            'description': "Exceptional performance. Strong customer loyalty.",
            'background': "#e3f2fd"
        }

def show_prediction_results(prediction, service_quality, price, innovation, sq_scores, p_scores, i_scores):
    st.markdown("## 🎯 Prediction Results")
    
    # Get loyalty level information
    loyalty_info = get_loyalty_level(prediction)
    
    # Display loyalty level prominently
    st.markdown(f"""
    <div style='padding: 20px; border-radius: 10px; background-color: {loyalty_info['background']}; 
         border: 2px solid {loyalty_info['color']}; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: {loyalty_info['color']}'>
            {loyalty_info['icon']} Loyalty Level: {loyalty_info['level']}
            <span style='float: right'>Score: {prediction:.1f}/10</span>
        </h3>
        <p style='margin: 10px 0 0 0; color: {loyalty_info['color']}'>{loyalty_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display score ranges reference
    st.markdown("""
    <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
        <div style='text-align: center; padding: 10px; background-color: #ffebee; border-radius: 5px; flex: 1; margin: 0 5px;'>
            <strong style='color: red'>🔴 Low</strong><br>0-4
        </div>
        <div style='text-align: center; padding: 10px; background-color: #fff3e0; border-radius: 5px; flex: 1; margin: 0 5px;'>
            <strong style='color: orange'>🟡 Medium</strong><br>4.1-6
        </div>
        <div style='text-align: center; padding: 10px; background-color: #e8f5e9; border-radius: 5px; flex: 1; margin: 0 5px;'>
            <strong style='color: green'>🟢 High</strong><br>6.1-8
        </div>
        <div style='text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 5px; flex: 1; margin: 0 5px;'>
            <strong style='color: blue'>💫 Very High</strong><br>8.1-10
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for the main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Overall Loyalty", prediction, col1),
        ("Service Quality", service_quality, col2),
        ("Price", price, col3),
        ("Innovation", innovation, col4)
    ]
    
    for title, value, col in metrics:
        with col:
            st.metric(title, f"{value:.1f}/10")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🕸 Detailed Analysis", "📈 Trends"])
    
    with tab1:
        # Gauge chart with 4 levels
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Customer Loyalty Score"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': loyalty_info['color']},
                'steps': [
                    {'range': [0, 4], 'color': "#ffebee"},
                    {'range': [4, 6], 'color': "#fff3e0"},
                    {'range': [6, 8], 'color': "#e8f5e9"},
                    {'range': [8, 10], 'color': "#e3f2fd"}
                ],
                'threshold': {
                    'line': {'color': loyalty_info['color'], 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with tab2:
        # Create 4-dimension radar plot
        categories = ['Customer Loyalty', 'Service Quality', 'Price', 'Innovation']
        values = [prediction, service_quality, price, innovation]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Metrics',
            line_color=loyalty_info['color'],
            fillcolor=f"rgba({','.join(['31', '119', '180', '0.3'])})"
        ))
        
        # Add a reference line for comparison
        # reference_values = [7.5, 7.5, 7.5, 7.5]  # Example reference values
        # fig_radar.add_trace(go.Scatterpolar(
        #     r=reference_values,
        #     theta=categories,
        #     name='Target',
        #     line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash'),
        #     fill='none'
        # ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with tab3:
        # Trend comparison
        fig_trend = go.Figure()
        
        categories = ['Customer Loyalty', 'Service Quality', 'Price', 'Innovation']
        values = [prediction, service_quality, price, innovation]
        
        fig_trend.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=[loyalty_info['color'], '#ff7f0e', '#2ca02c', '#d62728'],
            text=[f'{v:.1f}' for v in values],
            textposition='auto',
        ))
        
        fig_trend.update_layout(
            title="Category Performance Comparison",
            yaxis=dict(
                title="Score",
                range=[0, 10]
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Show insights
    st.markdown("## 💡 Key Insights")
    scores = {
        'Service Quality': service_quality,
        'Price': price,
        'Innovation': innovation
    }
    insights = get_detailed_insights(sq_scores, p_scores, i_scores, scores)
    
    # Create an expandable section for detailed insights
    with st.expander("Click to view detailed recommendations", expanded=False):
        for insight in insights:
            st.markdown(insight)

def show_home_page():
    # Header Section with Title and Description
    st.set_page_config(page_title="Pre", layout="wide")
    
    # Custom CSS for better styling and centering
    st.markdown("""
        <style>
        .main-header {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1em;
            color: #1E88E5;
        }
        .sub-header {
            font-size: 1.5em;
            font-weight: bold;
            color: #424242;
            text-align: center;
            margin-top: 2em;
        }
        .card {
            padding: 1.5em;
            border-radius: 8px;
            margin: 1.5em auto;
            background-color: #f8f9fa;
            border-left: 5px solid #1E88E5;
            max-width: 900px;
        }
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2em;
        }
        .section-content {
            text-align: center;
            padding: 1em;
        }
        .feature-card {
            background-color: #ffffff;
            padding: 1.5em;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 100%;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Content Container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Main Header
    st.markdown('<p class="main-header">🎯 LOYALITICS </p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
        <div class="card">
        <p style='font-size: 1.2em; text-align: center;'>
        Welcome to our advanced Customer Loyalty Prediction tool! Make data-driven decisions 
        by analyzing key metrics that influence customer satisfaction and loyalty.
        </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown('<p class="sub-header">🔑 Key Features</p>', unsafe_allow_html=True)
    
    # Create columns with padding
    col1, padding1, col2, padding2, col3 = st.columns([1, 0.2, 1, 0.2, 1])
    
    with col1:
        st.markdown("""
            <div class="feature-card">
            <h3>📊 Service Quality</h3>
            <p>Evaluate core service metrics:</p>
            <ul style="text-align: left;">
                <li>Driver Performance</li>
                <li>User Interface Experience</li>
                <li>Vehicle Standards</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown("""
            <div class="feature-card">
            <h3>💰 Price Analysis</h3>
            <p>Assess pricing strategy through:</p>
            <ul style="text-align: left;">
                <li>Cost Effectiveness</li>
                <li>Price Transparency</li>
                <li>Value Satisfaction</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
    with col3:
        st.markdown("""
            <div class="feature-card">
            <h3>💡 Innovation Impact</h3>
            <p>Measure innovation success via:</p>
            <ul style="text-align: left;">
                <li>Service Development</li>
                <li>User Experience</li>
                <li>System Performance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown('<p class="sub-header">🔄 How It Works</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
        <div style="text-align: center;">
            <ol style="display: inline-block; text-align: left;">
                <li><strong>Input Your Metrics:</strong> Rate each component on a scale of 0-10</li>
                <li><strong>Get Instant Analysis:</strong> Receive real-time loyalty predictions</li>
                <li><strong>View Detailed Insights:</strong> Access comprehensive recommendations</li>
                <li><strong>Track Progress:</strong> Monitor improvements over time</li>
            </ol>
        </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Benefits Section
    st.markdown('<p class="sub-header">✨ Benefits</p>', unsafe_allow_html=True)
    
    col1, padding, col2 = st.columns([1, 0.2, 1])
    
    with col1:
        st.markdown("""
            <div class="feature-card">
            <h3>For Business</h3>
            <ul style="text-align: left;">
                <li>Data-driven decision making</li>
                <li>Clear improvement areas</li>
                <li>Competitive advantage insights</li>
                <li>Customer retention strategies</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
            <h3>For Customers</h3>
            <ul style="text-align: left;">
                <li>Enhanced service quality</li>
                <li>Better value proposition</li>
                <li>Improved user experience</li>
                <li>Innovative solutions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Call to Action
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button('🚀 Start Prediction', use_container_width=True):
            st.session_state.page = 'prediction'
       
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Made with ❤️ for better customer experiences</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Close content container
    st.markdown('</div>', unsafe_allow_html=True)

def prediction_page():
    st.title('🪄 LOYALITY CUSTOMER PREDICTIOR')
    
    if st.button('← Back to Home'):
        st.session_state.page = 'home'
    
    st.markdown("### Enter your metrics (0-10)")
    st.markdown(
        "Users are expected to provide ratings on a **scale of 0 to 10**, where:\n"
        "- **0** represents the **lowest level of satisfaction**.\n"
        "- **10** represents the **highest level of satisfaction**.\n"
        "\n"
        "A **higher score** means greater satisfaction with the given factor, while a **lower score** suggests dissatisfaction. "
        "These ratings help analyze customer loyalty based on factors like **service quality, price, and innovation**.\n"
        "\n"
        "Please provide your ratings carefully, as they will be used to generate insights into customer preferences and loyalty trends."
    )

    # Create three columns for input
    col1, col2, col3 = st.columns(3)
    
    service_quality = create_input_section(col1, 'Service Quality')
    price = create_input_section(col2, 'Price')
    innovation = create_input_section(col3, 'Innovation')
    
    # Pilih kernel
    st.markdown("### Choose Kernel")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        rbf = st.checkbox("RBF")
    with col2:
        rational_quadratic = st.checkbox("RationalQuadratic")
    with col3:
        white_kernel = st.checkbox("WhiteKernel")
    with col4:
        constant_kernel = st.checkbox("ConstantKernel")
    with col5:
        semua = st.checkbox("All")

    if semua and (rbf or rational_quadratic or white_kernel or constant_kernel):
        st.error("Semua telah mencakup seluruh kernel yang ada. Silakan hapus ceklis lainnya.")

    if semua:
        rbf = False
        rational_quadratic = False
        white_kernel = False
        constant_kernel = False

    if st.button('🔍 Predict', use_container_width=True):
        try:
            if constant_kernel and not (rbf or rational_quadratic or white_kernel or semua):
                st.error("Constant Kernel tidak bisa dipilih sendiri. Silakan pilih 1 kernel lagi.")
            else:
                input_data = {
                    'ServiceQuality': service_quality,
                    'Price': price,
                    'Innovation': innovation
                }
                
                X = create_features(input_data)
                
                if semua:
                    kernel = (C(constant_value=1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) +
                            RBF(length_scale=1.0) +
                            WhiteKernel(noise_level=1.0))
                elif rbf and rational_quadratic and white_kernel:
                    kernel = (RationalQuadratic(length_scale=1.0, alpha=1.0) +
                            RBF(length_scale=1.0) +
                            WhiteKernel(noise_level=1.0))
                elif rbf and rational_quadratic:
                    kernel = (RationalQuadratic(length_scale=1.0, alpha=1.0) +
                            RBF(length_scale=1.0))
                elif rbf and white_kernel:
                    kernel = (RBF(length_scale=1.0) +
                            WhiteKernel(noise_level=1.0))
                elif rational_quadratic and white_kernel:
                    kernel = (RationalQuadratic(length_scale=1.0, alpha=1.0) +
                            WhiteKernel(noise_level=1.0))
                elif rbf:
                    kernel = RBF(length_scale=1.0)
                elif rational_quadratic:
                    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
                elif white_kernel:
                    kernel = WhiteKernel(noise_level=1.0)
                elif constant_kernel:
                    kernel = C(constant_value=1.0)
                
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.2,
                    n_restarts_optimizer=20,
                    normalize_y=True,
                    random_state=42
                )
                
                # Get individual scores
                sq_scores = [float(st.session_state.sq_1),
                            float(st.session_state.sq_2),
                            float(st.session_state.sq_3)]
                
                p_scores = [float(st.session_state.p_1),
                        float(st.session_state.p_2),
                        float(st.session_state.p_3)]
                
                i_scores = [float(st.session_state.i_1),
                        float(st.session_state.i_2),
                        float(st.session_state.i_3)]
                
                # Load training data and make prediction
                training_data = pd.read_csv('data_noise.csv')
                X_train = pd.DataFrame({
                    'ServiceQuality': training_data[['ServiceQuality_Driver', 'ServiceQuality_UI', 'ServiceQuality_Vehicle']].mean(axis=1),
                    'Price': training_data[['Price_Affordable', 'Price_Transparency', 'Price_Satisfaction']].mean(axis=1),
                    'Innovation': training_data[['Innovation_ServiceDevelopment', 'Innovation_Ease', 'Innovation_Response']].mean(axis=1)
                })
                y_train = training_data['Loyalty_Satisfaction'].values
                
                model.fit(X_train, y_train)
                prediction = float(model.predict(X))
                
                # Show prediction results with enhanced visualizations
                show_prediction_results(prediction, service_quality, price, innovation, 
                                    sq_scores, p_scores, i_scores)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")    

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'processing':
        show_processing_page()
    else:
        prediction_page()

if __name__ == '__main__':
    main()