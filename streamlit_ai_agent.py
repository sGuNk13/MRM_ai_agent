"""
AI Model Assessment Agent - Streamlit Version
==============================================
Complete AI-powered chatbot with:
- Conversation memory via Streamlit session state
- Natural language understanding via Llama
- Fuzzy model matching
- Risk assessment with detailed reports

Author: Financial Engineering Team
Version: 1.0 (Streamlit Native)
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import re
from groq import Groq

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Model Assessment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ModelPerformance:
    model_id: str
    metric: str
    baseline_performance: float
    current_performance: float
    deviation_percentage: float
    risk_rating: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            'model_id': self.model_id,
            'metric': self.metric,
            'baseline': self.baseline_performance,
            'current': self.current_performance,
            'deviation': self.deviation_percentage,
            'risk_rating': self.risk_rating,
            'timestamp': self.timestamp
        }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_state' not in st.session_state:
        st.session_state.current_state = "greeting"
    
    if 'model_id' not in st.session_state:
        st.session_state.model_id = None
    
    if 'current_performance' not in st.session_state:
        st.session_state.current_performance = None
    
    if 'assessment_result' not in st.session_state:
        st.session_state.assessment_result = None
    
    if 'model_database' not in st.session_state:
        st.session_state.model_database = None
    
    if 'criteria_database' not in st.session_state:
        st.session_state.criteria_database = None
    
    if 'groq_client' not in st.session_state:
        if 'GROQ_API_KEY' in st.secrets:
            st.session_state.groq_client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        else:
            st.session_state.groq_client = None

# ============================================================================
# AI AGENT CORE FUNCTIONS
# ============================================================================

def find_model_info(model_id: str, database: pd.DataFrame) -> Optional[Dict]:
    """Find model with intelligent fuzzy matching"""
    model_id_clean = model_id.strip().upper()
    
    # Exact match
    model_row = database[database['model_id'].str.strip().str.upper() == model_id_clean]
    if not model_row.empty:
        return model_row.iloc[0].to_dict()
    
    # Contains match
    similar = database[database['model_id'].str.upper().str.contains(model_id_clean, na=False)]
    if not similar.empty:
        return similar.iloc[0].to_dict()
    
    # Partial match (remove special characters)
    clean_search = re.sub(r'[^A-Z0-9]', '', model_id_clean)
    for _, row in database.iterrows():
        clean_row = re.sub(r'[^A-Z0-9]', '', row['model_id'].upper())
        if clean_search in clean_row or clean_row in clean_search:
            return row.to_dict()
    
    return None

def calculate_risk_rating(deviation_percentage: float, criteria: Dict) -> str:
    """Calculate risk rating based on performance deviation"""
    thresholds = {
        'low': criteria.get('low_threshold', 5.0),
        'medium': criteria.get('medium_threshold', 15.0),
        'high': criteria.get('high_threshold', 30.0)
    }
    
    # Performance improvement = low risk
    if deviation_percentage > 0:
        return "Low"
    
    # Performance degradation = risk based on thresholds
    abs_degradation = abs(deviation_percentage)
    
    if abs_degradation <= thresholds['low']:
        return "Low"
    elif abs_degradation <= thresholds['medium']:
        return "Medium"
    elif abs_degradation <= thresholds['high']:
        return "High"
    else:
        return "Critical"

def process_model_assessment(model_id: str, current_performance: float, 
                            model_database: pd.DataFrame, 
                            criteria_database: pd.DataFrame) -> ModelPerformance:
    """Process complete model assessment"""
    model_info = find_model_info(model_id, model_database)
    if not model_info:
        raise ValueError(f"Model '{model_id}' not found in database")
    
    baseline_performance = model_info.get('baseline_performance', model_info.get('baseline'))
    metric = model_info.get('metric')
    
    if baseline_performance is None or metric is None:
        raise ValueError(f"Incomplete model info for '{model_id}'")
    
    deviation_percentage = ((current_performance - baseline_performance) / baseline_performance) * 100
    
    criteria_row = criteria_database[
        criteria_database['metric'].str.strip().str.lower() == metric.strip().lower()
    ]
    
    if criteria_row.empty:
        raise ValueError(f"Risk criteria not found for metric '{metric}'")
    
    criteria = criteria_row.iloc[0].to_dict()
    risk_rating = calculate_risk_rating(deviation_percentage, criteria)
    
    return ModelPerformance(
        model_id=model_id,
        metric=metric,
        baseline_performance=baseline_performance,
        current_performance=current_performance,
        deviation_percentage=deviation_percentage,
        risk_rating=risk_rating
    )

def extract_model_id(message: str) -> Optional[str]:
    """Extract model ID from message"""
    patterns = [
        r'MODEL[_\s-]?\d+',
        r'model[_\s-]?\d+',
        r'\bM\d+\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(0).upper().replace(' ', '_')
    
    words = message.upper().split()
    for word in words:
        clean_word = word.strip('.,!?()[]{}')
        if st.session_state.model_database is not None:
            if find_model_info(clean_word, st.session_state.model_database):
                return clean_word
    
    return None

def extract_number(message: str) -> Optional[float]:
    """Extract performance number from message"""
    numbers = re.findall(r'\b\d+\.?\d*\b', message)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass
    return None

def get_llama_response(user_message: str, conversation_context: str) -> str:
    """Get response from Llama via Groq"""
    if st.session_state.groq_client is None:
        return "AI unavailable. Please configure GROQ_API_KEY in Streamlit secrets."
    
    try:
        system_prompt = f"""You are an AI Model Assessment Assistant. 
{conversation_context}

Respond naturally and helpfully. Keep responses concise but informative.
If the user is asking about assessing models, guide them through the process.
If they provide model IDs or performance values, acknowledge them clearly."""

        completion = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Llama: {str(e)}"

def understand_intent(message: str) -> str:
    """Understand user intent from natural language"""
    message_lower = message.lower().strip()
    
    if any(word in message_lower for word in ['assess', 'evaluate', 'check', 'test', 'analyze']):
        return 'assess'
    if any(word in message_lower for word in ['list', 'show', 'view', 'display', 'all models']):
        return 'list'
    if any(word in message_lower for word in ['help', 'how', 'what can', 'guide']):
        return 'help'
    if any(word in message_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return 'greeting'
    if any(word in message_lower for word in ['criteria', 'threshold', 'risk', 'rating']):
        return 'criteria'
    if any(word in message_lower for word in ['reset', 'start over', 'new', 'clear']):
        return 'reset'
    
    return 'unknown'

def generate_ai_response(user_message: str) -> str:
    """Generate AI response based on conversation state"""
    
    # Check for reset
    if understand_intent(user_message) == 'reset':
        st.session_state.current_state = "greeting"
        st.session_state.model_id = None
        st.session_state.current_performance = None
        st.session_state.assessment_result = None
        return "Chat reset. What would you like to do?"
    
    # State machine logic
    if st.session_state.current_state == "greeting":
        intent = understand_intent(user_message)
        
        if intent == 'assess':
            model_id = extract_model_id(user_message)
            if model_id and st.session_state.model_database is not None:
                model_info = find_model_info(model_id, st.session_state.model_database)
                if model_info:
                    st.session_state.model_id = model_id
                    st.session_state.current_state = "performance_input"
                    baseline = model_info.get('baseline_performance', model_info.get('baseline'))
                    metric = model_info.get('metric')
                    return f"Great! Found {model_id}.\nMetric: {metric}\nBaseline: {baseline}\n\nWhat's the current performance value?"
                else:
                    st.session_state.current_state = "model_input"
                    return "Let's assess a model. Please enter the Model ID."
            else:
                st.session_state.current_state = "model_input"
                return "Let's assess a model. Please enter the Model ID."
        
        elif intent == 'list':
            if st.session_state.model_database is not None:
                models = st.session_state.model_database.head(10)
                models_text = "\n".join([f"• {row['model_id']} ({row['metric']})" 
                                        for _, row in models.iterrows()])
                return f"Available Models:\n\n{models_text}\n\nSay 'assess MODEL_ID' to evaluate one!"
            return "Database not loaded yet."
        
        elif intent == 'criteria':
            if st.session_state.criteria_database is not None:
                criteria_text = "Risk Assessment Criteria:\n\n"
                for _, row in st.session_state.criteria_database.iterrows():
                    criteria_text += f"• {row['metric']}: Low ≤{row['low_threshold']}%, Medium ≤{row['medium_threshold']}%, High ≤{row['high_threshold']}%\n"
                return criteria_text
            return "Criteria database not loaded yet."
        
        else:
            context = "User is in greeting state. Guide them to assess models, list models, or ask for help."
            return get_llama_response(user_message, context)
    
    elif st.session_state.current_state == "model_input":
        model_id = extract_model_id(user_message)
        
        if not model_id:
            model_id = user_message.upper().replace(' ', '_')
        
        if st.session_state.model_database is not None:
            model_info = find_model_info(model_id, st.session_state.model_database)
            
            if model_info:
                st.session_state.model_id = model_id
                st.session_state.current_state = "performance_input"
                baseline = model_info.get('baseline_performance', model_info.get('baseline'))
                metric = model_info.get('metric')
                return f"Perfect! Found {model_id}.\nMetric: {metric}\nBaseline: {baseline}\n\nWhat's the current performance value?"
            else:
                available = list(st.session_state.model_database['model_id'].head(5))
                return f"Couldn't find '{model_id}'.\n\nTry one of these:\n" + "\n".join([f"• {m}" for m in available])
        
        return "Database not loaded yet."
    
    elif st.session_state.current_state == "performance_input":
        performance = extract_number(user_message)
        
        if performance is None:
            try:
                performance = float(user_message)
            except:
                return "I need a number for the performance value.\n\nExample: 95.5 or 0.89"
        
        try:
            assessment = process_model_assessment(
                st.session_state.model_id, 
                performance,
                st.session_state.model_database,
                st.session_state.criteria_database
            )
            st.session_state.assessment_result = assessment.to_dict()
            st.session_state.current_state = "assessment_complete"
            return "Assessment complete! See results below.\n\nWould you like to:\n• Assess another model?\n• Get a detailed report?\n• Start over?"
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif st.session_state.current_state == "assessment_complete":
        intent = understand_intent(user_message)
        
        if intent == 'assess' or 'another' in user_message.lower():
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            return "Ready for a new assessment. Which model?"
        
        elif 'report' in user_message.lower() or 'detail' in user_message.lower():
            return "Detailed report is displayed below. Would you like to assess another model?"
        
        else:
            context = f"User completed assessment. Result: {st.session_state.assessment_result}"
            return get_llama_response(user_message, context)
    
    return "I'm here to help with model assessments. Try 'assess model' to start."

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_assessment_card(assessment_dict: Dict):
    """Display assessment result card"""
    risk_colors = {
        "Low": "#27AE60",
        "Medium": "#F39C12", 
        "High": "#E67E22",
        "Critical": "#C0392B"
    }
    
    risk_color = risk_colors.get(assessment_dict['risk_rating'], "#95a5a6")
    
    trend = "improved" if assessment_dict['deviation'] > 0 else "degraded" if assessment_dict['deviation'] < 0 else "unchanged"
    
    st.markdown(f"""
    <div style="background: white; padding: 25px; border-radius: 15px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0;">
        <h3 style="color: {risk_color}; margin-top: 0;">Assessment Results</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Model ID:</td>
                <td style="padding: 10px 0;">{assessment_dict['model_id']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Metric:</td>
                <td style="padding: 10px 0;">{assessment_dict['metric']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Baseline:</td>
                <td style="padding: 10px 0;">{assessment_dict['baseline']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Current:</td>
                <td style="padding: 10px 0;">{assessment_dict['current']}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Deviation:</td>
                <td style="padding: 10px 0;">{assessment_dict['deviation']:.2f}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Status:</td>
                <td style="padding: 10px 0;">Performance {trend}</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; font-weight: bold;">Risk Rating:</td>
                <td style="padding: 10px 0;">
                    <span style="background: {risk_color}; color: white; padding: 5px 12px; 
                                 border-radius: 12px; font-weight: bold;">{assessment_dict['risk_rating']}</span>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

def generate_detailed_report(assessment_dict: Dict) -> str:
    """Generate comprehensive business report"""
    result = assessment_dict
    
    if result['deviation'] > 0:
        direction = "increase"
        status = "improved"
    elif result['deviation'] < 0:
        direction = "decrease"
        status = "degraded"
    else:
        direction = "no change"
        status = "remained stable"
    
    abs_deviation = abs(result['deviation'])
    
    risk_actions = {
        "Low": "Continue standard monitoring procedures with periodic performance reviews.",
        "Medium": "Implement enhanced monitoring and conduct root cause analysis within the next review cycle.",
        "High": "Immediate investigation required. Initiate model retraining process and validate data quality.",
        "Critical": "Emergency response required. Consider model rollback, immediate retraining, and stakeholder notification."
    }
    
    action = risk_actions[result['risk_rating']]
    
    report = f"""
# Model Performance Assessment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

The model's performance has **{status}**, with a **{abs_deviation:.2f}% {direction}** in {result['metric']}. 
This results in a **{result['risk_rating']}** risk rating.

---

## Model Information

- **Model ID:** {result['model_id']}
- **Metric:** {result['metric']}
- **Baseline Performance:** {result['baseline']}
- **Current Performance:** {result['current']}
- **Deviation:** {result['deviation']:.2f}%
- **Risk Rating:** {result['risk_rating']}

---

## Performance Analysis

The {direction} in {result['metric']} from {result['baseline']} to {result['current']} 
represents a {abs_deviation:.2f}% change in model effectiveness.

---

## Recommendations

**Primary Action:** {action}

**Next Steps:**
- Review model performance metrics
- Analyze data quality and pipeline health
- {'Document improvement factors' if result['deviation'] > 0 else 'Investigate root causes'}
- Brief stakeholders on findings

---

**Report ID:** {result['model_id']}-{datetime.now().strftime('%Y%m%d%H%M%S')}
"""
    
    return report

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    initialize_session_state()
    
    # Header
    st.title("AI Model Assessment Agent")
    st.caption("Powered by Llama 3.1 via Groq")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File uploaders
        st.subheader("Upload Databases")
        
        model_file = st.file_uploader("Model Database (Excel)", type=['xlsx'], key='model_upload')
        if model_file:
            st.session_state.model_database = pd.read_excel(model_file)
            st.success(f"Loaded {len(st.session_state.model_database)} models")
        
        criteria_file = st.file_uploader("Criteria Database (Excel)", type=['xlsx'], key='criteria_upload')
        if criteria_file:
            st.session_state.criteria_database = pd.read_excel(criteria_file)
            st.success(f"Loaded {len(st.session_state.criteria_database)} criteria")
        
        st.divider()
        
        # Actions
        st.subheader("Actions")
        
        if st.button("Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            st.session_state.current_performance = None
            st.session_state.assessment_result = None
            st.rerun()
        
        if st.button("Show Databases", use_container_width=True):
            if st.session_state.model_database is not None:
                with st.expander("Model Database"):
                    st.dataframe(st.session_state.model_database)
            if st.session_state.criteria_database is not None:
                with st.expander("Criteria Database"):
                    st.dataframe(st.session_state.criteria_database)
        
        st.divider()
        
        st.subheader("Status")
        st.write(f"**State:** {st.session_state.current_state}")
        if st.session_state.model_id:
            st.write(f"**Model:** {st.session_state.model_id}")
    
    # Main chat interface
    if st.session_state.model_database is None or st.session_state.criteria_database is None:
        st.warning("Please upload both Model Database and Criteria Database in the sidebar to begin.")
        return
    
    if st.session_state.groq_client is None:
        st.error("GROQ_API_KEY not configured. Add it to Streamlit secrets to enable AI responses.")
        st.info("Set up: .streamlit/secrets.toml with GROQ_API_KEY = 'your-key-here'")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Display assessment if available and in right state
    if (st.session_state.assessment_result and 
        st.session_state.current_state == "assessment_complete"):
        display_assessment_card(st.session_state.assessment_result)
        
        if st.button("Generate Detailed Report"):
            report = generate_detailed_report(st.session_state.assessment_result)
            st.markdown(report)
            st.download_button(
                "Download Report",
                report,
                file_name=f"report_{st.session_state.assessment_result['model_id']}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Generate AI response
        response = generate_ai_response(prompt)
        
        # Add assistant message
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        
        # Rerun to update UI
        st.rerun()

if __name__ == "__main__":
    main()
