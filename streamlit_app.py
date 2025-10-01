"""
AI Model Assessment Agent - Streamlit Version
==============================================
Natural conversational AI with Llama integration
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
import re
from groq import Groq
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Model Assessment Agent",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_DATABASE_FILE = "mockup_database.xlsx"
CRITERIA_DATABASE_FILE = "mockup_criteria.xlsx"

# ============================================================================
# DATA MODELS
# ============================================================================

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
# DATABASE LOADING
# ============================================================================

@st.cache_data
def load_databases():
    """Load required Excel databases with error handling"""
    errors = []
    
    if not os.path.exists(MODEL_DATABASE_FILE):
        errors.append(f"Missing required file: {MODEL_DATABASE_FILE}")
    
    if not os.path.exists(CRITERIA_DATABASE_FILE):
        errors.append(f"Missing required file: {CRITERIA_DATABASE_FILE}")
    
    if errors:
        raise FileNotFoundError("\n".join(errors))
    
    try:
        model_db = pd.read_excel(MODEL_DATABASE_FILE)
        criteria_db = pd.read_excel(CRITERIA_DATABASE_FILE)
        
        required_model_cols = ['model_id', 'baseline_performance', 'metric']
        required_criteria_cols = ['metric', 'low_threshold', 'medium_threshold', 'high_threshold']
        
        missing_model_cols = [col for col in required_model_cols if col not in model_db.columns]
        missing_criteria_cols = [col for col in required_criteria_cols if col not in criteria_db.columns]
        
        if missing_model_cols:
            errors.append(f"Model database missing columns: {missing_model_cols}")
        
        if missing_criteria_cols:
            errors.append(f"Criteria database missing columns: {missing_criteria_cols}")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return model_db, criteria_db
        
    except Exception as e:
        raise Exception(f"Error loading databases: {str(e)}")

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
    
    if 'assessment_result' not in st.session_state:
        st.session_state.assessment_result = None
    
    if 'groq_client' not in st.session_state:
        if 'GROQ_API_KEY' in st.secrets:
            st.session_state.groq_client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        else:
            st.session_state.groq_client = None
    
    if 'gsheet_client' not in st.session_state:
        st.session_state.gsheet_client = None
        if 'gcp_service_account' in st.secrets and 'GOOGLE_SHEET_ID' in st.secrets:
            try:
                scope = ['https://spreadsheets.google.com/feeds',
                        'https://www.googleapis.com/auth/drive']
                creds = ServiceAccountCredentials.from_json_keyfile_dict(
                    st.secrets["gcp_service_account"], scope)
                st.session_state.gsheet_client = gspread.authorize(creds)
            except Exception as e:
                st.session_state.gsheet_error = str(e)

# ============================================================================
# GOOGLE SHEETS LOGGING
# ============================================================================

def log_assessment_to_gsheet(assessment_dict: Dict) -> bool:
    """Log assessment to Google Sheets"""
    if st.session_state.gsheet_client is None:
        return False
    
    try:
        sheet = st.session_state.gsheet_client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        worksheet = sheet.sheet1
        
        row_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            assessment_dict['model_id'],
            assessment_dict['metric'],
            assessment_dict['baseline'],
            assessment_dict['current'],
            f"{assessment_dict['deviation']:.2f}%",
            assessment_dict['risk_rating']
        ]
        
        worksheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"Failed to log to Google Sheets: {str(e)}")
        return False

# ============================================================================
# CORE ASSESSMENT FUNCTIONS
# ============================================================================

def find_model_info(model_id: str, database: pd.DataFrame) -> Optional[Dict]:
    """Find model with intelligent fuzzy matching"""
    model_id_clean = model_id.strip().upper()
    
    model_row = database[database['model_id'].str.strip().str.upper() == model_id_clean]
    if not model_row.empty:
        return model_row.iloc[0].to_dict()
    
    similar = database[database['model_id'].str.upper().str.contains(model_id_clean, na=False)]
    if not similar.empty:
        return similar.iloc[0].to_dict()
    
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
    
    if deviation_percentage > 0:
        return "Low"
    
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

def extract_model_id(message: str, model_database: pd.DataFrame) -> Optional[str]:
    """Extract model ID from message"""
    patterns = [
        r'MODEL[_\s-]?\d+',
        r'model[_\s-]?\d+',
        r'\bM\d+\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            candidate = match.group(0).upper().replace(' ', '_')
            if find_model_info(candidate, model_database):
                return candidate
    
    words = message.upper().split()
    for word in words:
        clean_word = word.strip('.,!?()[]{}')
        if find_model_info(clean_word, model_database):
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

# ============================================================================
# AI CONVERSATION ENGINE
# ============================================================================

def build_context(model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Build rich context for Llama"""
    
    models_list = "\n".join([f"- {row['model_id']} (Metric: {row['metric']}, Baseline: {row['baseline_performance']})" 
                         for _, row in model_database.iterrows()])
    
    criteria_list = "\n".join([f"- {row['metric']}: Low ≤{row['low_threshold']}%, Medium ≤{row['medium_threshold']}%, High ≤{row['high_threshold']}%" 
                               for _, row in criteria_database.iterrows()])
    
    state = st.session_state.current_state
    model_id = st.session_state.model_id
    
    context = f"""You are a helpful AI assistant for model performance assessment.

CURRENT STATE: {state}

AVAILABLE MODELS (total: {len(model_database)}):
{models_list}

CRITICAL: Only use models from this exact list. Do not invent or suggest models that are not listed above.

RISK CRITERIA:
{criteria_list}

CONVERSATION FLOW:
1. greeting -> User wants to assess a model or ask questions
2. model_input -> Waiting for user to provide a model ID
3. performance_input -> Model {model_id if model_id else '[pending]'} selected, waiting for current performance value
4. assessment_complete -> Assessment done, user can request report or assess another model

YOUR ROLE:
- Have natural, helpful conversations
- Guide users through the assessment process
- Answer questions about models, metrics, and risk ratings
- Be conversational but professional
- When user provides model ID or performance value, acknowledge it clearly

IMPORTANT:
- Keep responses concise (2-3 sentences unless explaining something complex)
- Don't repeat the entire state/context back to the user
- Be helpful and friendly without being overly verbose"""
    
    return context

def get_llama_response(user_message: str, model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Get natural response from Llama"""
    if st.session_state.groq_client is None:
        return "I need the GROQ_API_KEY to respond. Please configure it in Streamlit secrets."
    
    try:
        context = build_context(model_database, criteria_database)
        
        # Build conversation history for Llama
        messages = [{"role": "system", "content": context}]
        
        # Add recent conversation history (last 6 messages for context)
        for msg in st.session_state.messages[-6:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Llama: {str(e)}"

# ============================================================================
# STATE MACHINE
# ============================================================================

def process_user_input(user_message: str, model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Process user input and manage state transitions"""
    
    # Check for reset request
    if any(word in user_message.lower() for word in ['reset', 'start over', 'restart']):
        st.session_state.current_state = "greeting"
        st.session_state.model_id = None
        st.session_state.assessment_result = None
        return get_llama_response("User wants to reset/start over. Acknowledge the reset and ask what they'd like to do.", model_database, criteria_database)
    
    state = st.session_state.current_state
    
    # State: greeting - open conversation
    if state == "greeting":
        # Check if user mentions wanting to assess a model
        model_id = extract_model_id(user_message, model_database)
        
        if model_id:
            model_info = find_model_info(model_id, model_database)
            st.session_state.model_id = model_id
            st.session_state.current_state = "performance_input"
            st.session_state.assessment_result = None  # Clear old assessment
            st.session_state.logged_to_gsheet = False  # Clear log status
            
            context_msg = f"User mentioned model {model_id}. Confirm you found it (metric: {model_info['metric']}, baseline: {model_info['baseline_performance']}) and ask for the current performance value."
            return get_llama_response(context_msg, model_database, criteria_database)
        else:
            return get_llama_response(user_message, model_database, criteria_database)
    
    # State: model_input - waiting for model ID
    elif state == "model_input":
        model_id = extract_model_id(user_message, model_database)
        
        if not model_id:
            model_id = user_message.upper().replace(' ', '_')
        
        model_info = find_model_info(model_id, model_database)
        
        if model_info:
            st.session_state.model_id = model_id
            st.session_state.current_state = "performance_input"
            
            context_msg = f"User provided model {model_id}. Confirm you found it (metric: {model_info['metric']}, baseline: {model_info['baseline_performance']}) and ask for the current performance value."
            return get_llama_response(context_msg, model_database, criteria_database)
        else:
            available = list(model_database['model_id'].head(5))
            context_msg = f"Model {model_id} not found. Available models include: {', '.join(available)}. Ask user to try one of these or provide a valid model ID."
            return get_llama_response(context_msg, model_database, criteria_database)
    
    # State: performance_input - waiting for performance value
    elif state == "performance_input":
        performance = extract_number(user_message)
        
        if performance is None:
            try:
                performance = float(user_message.strip())
            except:
                context_msg = f"User's input '{user_message}' doesn't contain a valid number. Ask them to provide the current performance value as a number."
                return get_llama_response(context_msg, model_database, criteria_database)
        
        try:
            assessment = process_model_assessment(
                st.session_state.model_id,
                performance,
                model_database,
                criteria_database
            )
            st.session_state.assessment_result = assessment.to_dict()
            st.session_state.current_state = "assessment_complete"
            
            # Log to Google Sheets
            if log_assessment_to_gsheet(st.session_state.assessment_result):
                st.session_state.logged_to_gsheet = True
            
            result = st.session_state.assessment_result
            context_msg = f"Assessment complete! Model {result['model_id']} - Deviation: {result['deviation']:.2f}%, Risk: {result['risk_rating']}. Tell user assessment is ready (they'll see the detailed card below). Ask if they want a detailed report or to assess another model."
            return get_llama_response(context_msg, model_database, criteria_database)
        except Exception as e:
            return f"Error during assessment: {str(e)}"
    
    # State: assessment_complete - assessment done
    elif state == "assessment_complete":
        # Check if user wants to assess another model
        if any(word in user_message.lower() for word in ['assess', 'another', 'new model', 'next']):
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            st.session_state.assessment_result = None  # Clear old assessment
            st.session_state.logged_to_gsheet = False  # Clear log status
            return get_llama_response("User wants to assess another model. Acknowledge and ask which model they'd like to assess next.", model_database, criteria_database)
        else:
            return get_llama_response(user_message, model_database, criteria_database)
    
    return get_llama_response(user_message, model_database, criteria_database)

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

    # Custom CSS to flip chat positions
    st.markdown("""
        <style>
        /* Move user messages to right */
        .stChatMessage[data-testid="user-message"] {
            flex-direction: row-reverse;
        }
        .stChatMessage[data-testid="user-message"] .stMarkdown {
            text-align: right;
        }
        
        /* Move assistant messages to left (default, but explicit) */
        .stChatMessage[data-testid="assistant-message"] {
            flex-direction: row;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load databases
    try:
        model_database, criteria_database = load_databases()
    except FileNotFoundError as e:
        st.error("CRITICAL ERROR: Required database files not found")
        st.code(str(e))
        st.info(f"""
Please ensure these files are in the same directory as streamlit_app.py:
- {MODEL_DATABASE_FILE}
- {CRITERIA_DATABASE_FILE}

Upload them to your GitHub repository and redeploy.
        """)
        return
    except Exception as e:
        st.error("CRITICAL ERROR: Failed to load databases")
        st.code(str(e))
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        st.success(f"Loaded {len(model_database)} models")
        st.success(f"Loaded {len(criteria_database)} criteria")
        
        if st.session_state.gsheet_client:
            st.success("Google Sheets connected")
        else:
            st.warning("Google Sheets not configured")
        
        st.divider()
        
        st.subheader("Actions")
        
        if st.button("Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            st.session_state.assessment_result = None
            st.rerun()
        
        if st.button("Show Databases", use_container_width=True):
            with st.expander("Model Database"):
                st.dataframe(model_database)
            with st.expander("Criteria Database"):
                st.dataframe(criteria_database)
        
        st.divider()
        
        st.subheader("Status")
        st.write(f"**State:** {st.session_state.current_state}")
        if st.session_state.model_id:
            st.write(f"**Model:** {st.session_state.model_id}")
    
    # Display chat messages with cat avatars
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message(msg['role'], avatar="👤"):
                st.markdown(msg['content'])
        else:
            with st.chat_message(msg['role'], avatar="🐱"):
                st.markdown(msg['content'])
    
    # Display assessment card if available
    if (st.session_state.assessment_result and 
        st.session_state.current_state == "assessment_complete"):
        display_assessment_card(st.session_state.assessment_result)
        
        if st.session_state.get('logged_to_gsheet'):
            st.success("Assessment logged to Google Sheets")
        elif st.session_state.gsheet_client is None:
            st.info("Google Sheets not configured - assessment not logged")
        
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
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        response = process_user_input(prompt, model_database, criteria_database)
        
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        
        st.rerun()

if __name__ == "__main__":
    main()
