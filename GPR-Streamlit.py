import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C

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

def create_input_section(container, prefix):
    with container:
        st.subheader(prefix)
        questions = []
        if prefix == 'Service Quality':
            questions.append(st.slider('Question - Driver Quality', 0, 10, 5, key='sq_1'))
            questions.append(st.slider('Question - UI Experience', 0, 10, 5, key='sq_2'))
            questions.append(st.slider('Question - Vehicle Quality', 0, 10, 5, key='sq_3'))
        elif prefix == 'Price':
            questions.append(st.slider('Question - Affordability', 0, 10, 5, key='p_1'))
            questions.append(st.slider('Question - Price Transparency', 0, 10, 5, key='p_2'))
            questions.append(st.slider('Question - Overall Satisfaction', 0, 10, 5, key='p_3'))
        else:  # Innovation
            questions.append(st.slider('Question - Service Development', 0, 10, 5, key='i_1'))
            questions.append(st.slider('Question - Ease of Use', 0, 10, 5, key='i_2'))
            questions.append(st.slider('Question - Response Time', 0, 10, 5, key='i_3'))
        
        return np.mean(questions)

def show_home_page():
    # Header Section with Title and Description
    st.set_page_config(page_title="Customer Loyalty Predictor", layout="wide")
    
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
    st.markdown('<p class="main-header">🎯 Customer Loyalty Predictor</p>', unsafe_allow_html=True)
    
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
    st.title('Customer Loyalty Prediction')

    if st.button('← Back to Home'):
        st.session_state.page = 'home'

    st.write('Enter values for each parameter (scale 0-10):')

    col1, col2, col3 = st.columns(3)

    service_quality = create_input_section(col1, 'Service Quality')
    price = create_input_section(col2, 'Price')
    innovation = create_input_section(col3, 'Innovation')

    input_data = {
        'ServiceQuality': service_quality,
        'Price': price,
        'Innovation': innovation
    }

    if st.button('🔍 Predict'):
        try:
            X = create_features(input_data)

            kernel = (C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) +
                     RBF(length_scale=1.0) +
                     WhiteKernel(noise_level=1.0))

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.2,
                n_restarts_optimizer=20,
                normalize_y=True,
                random_state=42
            )

            try:
                training_data = pd.read_csv('data_noise.csv')
                X_train = pd.DataFrame({
                    'ServiceQuality': training_data[['ServiceQuality_Driver', 'ServiceQuality_UI', 'ServiceQuality_Vehicle']].mean(axis=1),
                    'Price': training_data[['Price_Affordable', 'Price_Transparency', 'Price_Satisfaction']].mean(axis=1),
                    'Innovation': training_data[['Innovation_ServiceDevelopment', 'Innovation_Ease', 'Innovation_Response']].mean(axis=1)
                })
                y_train = training_data['Loyalty_Satisfaction'].values
                model.fit(X_train, y_train)
                prediction = float(model.predict(X))
                
                st.subheader('🔮 Prediction Results')

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Customer Loyalty Score"},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 4], 'color': "red"},
                            {'range': [4, 6], 'color': "yellow"},
                            {'range': [6, 8], 'color': "lightgreen"},
                            {'range': [8, 10], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                st.plotly_chart(fig_gauge)

                st.write(f'Predicted Loyalty Level: {prediction:.2f}/10')

                if prediction <= 4:
                    st.error('🔴 Low Loyalty Level')
                elif prediction <= 6:
                    st.warning('🟡 Medium Loyalty Level')
                elif prediction <= 8:
                    st.success('🟢 High Loyalty Level')
                else:
                    st.success('🟢 Very High Loyalty Level')

                # Bar chart for main categories
                fig_bars = go.Figure(data=[
                    go.Bar(
                        x=['Service Quality', 'Price', 'Innovation'],
                        y=[service_quality, price, innovation],
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                    )
                ])
                fig_bars.update_layout(
                    title="Category Scores",
                    xaxis_title="Category",
                    yaxis_title="Score",
                    yaxis_range=[0, 10]
                )

                st.plotly_chart(fig_bars)

                st.subheader("📊 Component Analysis")
                
                scores = {
                    'Service Quality': service_quality,
                    'Price': price,
                    'Innovation': innovation
                }
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("💪 Strongest Category:")
                    component, score = sorted_scores[0]
                    st.write(f"{component}: {score:.1f}/10")
                with col2:
                    st.write("🎯 Area for Improvement:")
                    component, score = sorted_scores[-1]
                    st.write(f"{component}: {score:.1f}/10")

                st.subheader("💡 Key Insights & Recommendations")
                
                sq_scores = [
                    float(st.session_state.sq_1),
                    float(st.session_state.sq_2),
                    float(st.session_state.sq_3)
                ]
                
                p_scores = [
                    float(st.session_state.p_1),
                    float(st.session_state.p_2),
                    float(st.session_state.p_3)
                ]
                
                i_scores = [
                    float(st.session_state.i_1),
                    float(st.session_state.i_2),
                    float(st.session_state.i_3)
                ]
                
                insights = get_detailed_insights(sq_scores, p_scores, i_scores, scores)
                
                for insight in insights:
                    st.markdown(insight)

            except Exception as e:
                st.error(f"Error loading training data: {str(e)}")
                return

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Error details for debugging:", e)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_home_page()
    else:
        prediction_page()

if __name__ == '__main__':
    main()