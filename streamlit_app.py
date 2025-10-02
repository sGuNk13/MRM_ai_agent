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

    if 'degradation_reason' not in st.session_state:
        st.session_state.degradation_reason = None
    
    if 'mitigation_plan' not in st.session_state:
        st.session_state.mitigation_plan = None

# ============================================================================
# GOOGLE SHEETS LOGGING
# ============================================================================

def log_assessment_to_gsheet(assessment_dict: Dict) -> bool:
    """Log assessment to Google Sheets with monthly sheet selection"""
    if st.session_state.gsheet_client is None:
        return False
    
    try:
        sheet = st.session_state.gsheet_client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        
        # Get current month name
        current_month = datetime.now().strftime('%B')  # e.g., "October", "November"
        
        # Try to get the worksheet for current month
        try:
            worksheet = sheet.worksheet(current_month)
        except gspread.exceptions.WorksheetNotFound:
            # Sheet doesn't exist, create it
            worksheet = sheet.add_worksheet(title=current_month, rows=1000, cols=10)
            
            # Add headers to new sheet
            headers = ['timestamp', 'model_id', 'metric', 'baseline', 
                      'current_performance', 'deviation', 'risk_rating']
            worksheet.append_row(headers)
        
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

def log_assessment_to_gsheet_with_details(assessment_dict: Dict, reason: str, mitigation: str) -> bool:
    """Log assessment with degradation reason and mitigation plan"""
    if st.session_state.gsheet_client is None:
        return False
    
    try:
        sheet = st.session_state.gsheet_client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        
        current_month = datetime.now().strftime('%B')
        
        try:
            worksheet = sheet.worksheet(current_month)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=current_month, rows=1000, cols=12)
            headers = ['timestamp', 'model_id', 'metric', 'baseline', 
                      'current_performance', 'deviation', 'risk_rating',
                      'degradation_reason', 'mitigation_plan']
            worksheet.append_row(headers)
        
        row_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            assessment_dict['model_id'],
            assessment_dict['metric'],
            assessment_dict['baseline'],
            assessment_dict['current'],
            f"{assessment_dict['deviation']:.2f}%",
            assessment_dict['risk_rating'],
            reason if reason else 'N/A',
            mitigation if mitigation else 'N/A'
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
    """Extract model ID from message - handles multiple formats"""
    
    # Pattern for real production format: CRR_XXX_###_##_##
    # Pattern for test format: model_id_####
    patterns = [
        r'\b[A-Z]{3}_[A-Z]{3}_\d+(?:_\d+)*\b',  # CRR_OTH_233 or CRR_OTH_444_02_01
        r'\bmodel_id_\d+\b',                      # model_id_1234 (test format)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, message.lower())  # Use lower() for case-insensitive
        for match in matches:
            # Verify it exists in database (case-insensitive check)
            if not model_database[model_database['model_id'].str.lower() == match.lower()].empty:
                # Return the actual model_id from database (preserves original case)
                actual_id = model_database[model_database['model_id'].str.lower() == match.lower()].iloc[0]['model_id']
                return actual_id
    
    # Exact word match as fallback
    words = message.split()
    for word in words:
        clean_word = word.strip('.,!?()[]{}"\' ')
        if not model_database[model_database['model_id'].str.lower() == clean_word.lower()].empty:
            actual_id = model_database[model_database['model_id'].str.lower() == clean_word.lower()].iloc[0]['model_id']
            return actual_id
    
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
    
    state = st.session_state.current_state
    model_id = st.session_state.model_id
    
    # Only include model list when actually needed
    if state in ["model_input", "greeting"]:
        models_context = f"""
There are {len(model_database)} models available in the database.
DO NOT mention specific model IDs unless the user explicitly asks for examples or a list.
"""
    else:
        models_list = "\n".join([f"- {row['model_id']} (Metric: {row['metric']}, Baseline: {row['baseline_performance']})" 
                                 for _, row in model_database.iterrows()])
        models_context = f"""
AVAILABLE MODELS:
{models_list}
"""
    
    criteria_list = "\n".join([f"- {row['metric']}: Low ≤{row['low_threshold']}%, Medium ≤{row['medium_threshold']}%, High ≤{row['high_threshold']}%" 
                               for _, row in criteria_database.iterrows()])
    
    # Only show current assessment details if we're in assessment_complete state
    if state == "assessment_complete" and st.session_state.assessment_result:
        assessment_info = f"""
COMPLETED ASSESSMENT:
- Model: {st.session_state.assessment_result['model_id']}
- Metric: {st.session_state.assessment_result['metric']}
- Deviation: {st.session_state.assessment_result['deviation']:.2f}%
- Risk: {st.session_state.assessment_result['risk_rating']}
"""
    else:
        assessment_info = ""
    
    context = f"""You are a helpful AI assistant for model performance assessment.

CURRENT STATE: {state}
{f"SELECTED MODEL: {model_id}" if model_id and state != "greeting" else ""}

{models_context}

RISK CRITERIA:
{criteria_list}

{assessment_info}

CONVERSATION FLOW:
1. greeting -> User wants to assess a model or ask questions
2. model_input -> Waiting for user to provide a model ID
3. performance_input -> Model {model_id if model_id else '[pending]'} selected, waiting for current performance value
4. reason_required -> For High/Critical risk only - asking for degradation reason
5. mitigation_required -> For High/Critical risk only - asking for mitigation plan
6. assessment_complete -> Assessment done

YOUR ROLE:
- Have natural, helpful conversations
- Guide users through the assessment process
- When in model_input state and user hasn't provided a model ID yet, simply ask "Which model ID would you like to assess?" - DO NOT mention any specific model IDs
- When asking for mitigation plans, keep it simple - don't provide lengthy checklists or sub-questions
- Trust users to provide appropriate detail without excessive prompting
- Be conversational and concise (2-3 sentences)
- NEVER reference or mention metrics, model IDs, or assessment results from previous assessments
- Each assessment is independent - clear your memory of past assessments when starting a new one

CRITICAL INSTRUCTIONS:
- Only use models from the exact list provided above
- Do not invent or suggest models that are not listed
- When asking user which model to assess, do NOT mention specific model IDs unless they ask for examples
- Focus only on the CURRENT assessment - ignore any data from previous assessments"""
    
    return context

def refine_text_with_llama(text: str, field_type: str) -> str:
    """Use Llama to refine user's answer into formal professional English"""
    if st.session_state.groq_client is None:
        return text
    
    try:
        if field_type == "reason":
            prompt = f"""Refine this explanation into 1-2 clear, professional sentences. Keep the original meaning, just improve grammar and formality. Do not add extra details.

Original: {text}

Refined:"""
        else:  # mitigation
            prompt = f"""Refine this mitigation plan into 2-3 clear, professional sentences. Keep it concise - just improve grammar and formality. Do not expand or add steps that weren't mentioned.

Original: {text}

Refined:"""
        
        completion = st.session_state.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You refine text to be professional and grammatically correct. Keep responses BRIEF - only improve the original, don't expand it."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Very low temperature for minimal creativity
            max_tokens=150     # Limit length
        )
        
        refined = completion.choices[0].message.content.strip()
        
        # Safety check - if refined is more than 2x original length, use original
        if len(refined) > len(text) * 2:
            return text
        
        return refined if refined else text
        
    except Exception as e:
        return text

def get_llama_response(user_message: str, model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Get natural response from Llama"""
    if st.session_state.groq_client is None:
        return "I need the GROQ_API_KEY to respond. Please configure it in Streamlit secrets."
    
    try:
        context = build_context(model_database, criteria_database)
        
        messages = [{"role": "system", "content": context}]
        
        # Only include conversation history if NOT in model_input state
        # This prevents Llama from "remembering" model IDs mentioned earlier
        if st.session_state.current_state != "model_input":
            for msg in st.session_state.messages[-6:]:
                messages.append({"role": msg['role'], "content": msg['content']})
        
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
        st.session_state.degradation_reason = None
        st.session_state.mitigation_plan = None
        return get_llama_response("User wants to reset/start over. Acknowledge the reset and ask what they'd like to do.", model_database, criteria_database)
    
    state = st.session_state.current_state
    
    # State: greeting - open conversation
    if state == "greeting":
        user_lower = user_message.lower()
        wants_assessment = any(word in user_lower for word in ['assess', 'check', 'evaluate', 'test', 'analyze'])
        
        if wants_assessment:
            # Try to extract model ID using the pattern
            found_model = extract_model_id(user_message, model_database)
            
            if found_model:
                model_info = find_model_info(found_model, model_database)
                
                # Clear all previous assessment data
                st.session_state.model_id = found_model
                st.session_state.current_state = "performance_input"
                st.session_state.assessment_result = None
                st.session_state.logged_to_gsheet = False
                st.session_state.degradation_reason = None
                st.session_state.mitigation_plan = None
                
                context_msg = f"""User wants to assess model {found_model}.
Respond EXACTLY in this format:
"You've selected {found_model}, which has a {model_info['metric']} metric and a baseline performance of {model_info['baseline_performance']}.

To proceed with the assessment, could you please provide the current {model_info['metric']} performance value?"

Follow this exact structure to inform the user."""
                
                return get_llama_response(context_msg, model_database, criteria_database)
            else:
                # No model ID found - ask for it
                st.session_state.current_state = "model_input"
                st.session_state.model_id = None
                st.session_state.assessment_result = None
                
                context_msg = "User wants to assess a model but didn't specify which one. Ask them which model ID they want to assess."
                return get_llama_response(context_msg, model_database, criteria_database)
        else:
            return get_llama_response(user_message, model_database, criteria_database)
    
    # State: model_input - waiting for model ID
    elif state == "model_input":
        # Try to extract model ID
        found_model = extract_model_id(user_message, model_database)
        
        if not found_model:
            # Try exact match with cleaned input
            clean_input = user_message.strip().upper()
            if not model_database[model_database['model_id'].str.upper() == clean_input].empty:
                found_model = clean_input
        
        if found_model:
            model_info = find_model_info(found_model, model_database)
            
            # Clear all previous data
            st.session_state.model_id = found_model
            st.session_state.current_state = "performance_input"
            st.session_state.assessment_result = None
            st.session_state.logged_to_gsheet = False
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
            
            context_msg = f"""User selected model {found_model}.
Respond EXACTLY in this format:
"You've selected {found_model}, which has a {model_info['metric']} metric and a baseline performance of {model_info['baseline_performance']}.

To proceed with the assessment, could you please provide the current {model_info['metric']} performance value?"

Follow this exact structure to inform the user."""
            
            return get_llama_response(context_msg, model_database, criteria_database)
        else:
            context_msg = f"'{user_message}' is not a valid model ID. Ask user to provide a valid model ID from the database."
            return get_llama_response(context_msg, model_database, criteria_database)
    
    # State: performance_input - waiting for performance value
    elif state == "performance_input":
        performance = extract_number(user_message)
        
        if performance is None:
            try:
                performance = float(user_message.strip())
            except:
                model_info = find_model_info(st.session_state.model_id, model_database)
                metric = model_info['metric'] if model_info else 'performance'
                
                context_msg = f"User's input '{user_message}' is not a number. We need the numeric {metric} value. Ask them to provide just the number."
                return get_llama_response(context_msg, model_database, criteria_database)
        
        # Log for debugging
        st.write(f"DEBUG: Extracted performance value: {performance}")
        st.write(f"DEBUG: Current model_id: {st.session_state.model_id}")
        st.write(f"DEBUG: Current state: {st.session_state.current_state}")
        
        try:
            assessment = process_model_assessment(
                st.session_state.model_id,
                performance,
                model_database,
                criteria_database
            )
            
            st.write(f"DEBUG: Assessment result: {assessment.to_dict()}")
            
            st.session_state.assessment_result = assessment.to_dict()
            risk_rating = st.session_state.assessment_result['risk_rating']
            
            st.write(f"DEBUG: Risk rating: {risk_rating}")
            
            if risk_rating in ['High', 'Critical']:
                st.session_state.current_state = "reason_required"
                context_msg = f"Assessment shows {risk_rating} risk with {assessment.deviation_percentage:.2f}% degradation. Ask user to explain the REASON for this performance degradation. Vague answers are not acceptable."
                return get_llama_response(context_msg, model_database, criteria_database)
            else:
                st.session_state.current_state = "assessment_complete"
                
                if log_assessment_to_gsheet(st.session_state.assessment_result):
                    st.session_state.logged_to_gsheet = True
                
                result = st.session_state.assessment_result
                context_msg = f"Assessment complete! Model {result['model_id']}: Current {result['current']} vs Baseline {result['baseline']}. Deviation: {result['deviation']:.2f}%, Risk: {result['risk_rating']}. Tell user assessment is complete (results shown below). Ask if they want to assess another model."
                return get_llama_response(context_msg, model_database, criteria_database)
                
        except Exception as e:
            st.write(f"DEBUG: Error in assessment: {str(e)}")
            return f"Error during assessment: {str(e)}"

    # State: reason_required - waiting for degradation reason (High/Critical only)
    elif state == "reason_required":
        uninformative_phrases = [
            'no idea', 'don\'t know', 'don\'t care', 'not sure', 
            'dunno', 'idk', 'whatever', 'none', 'n/a', 'na'
        ]
        
        user_lower = user_message.lower().strip()
        
        if len(user_message.strip()) < 20 or any(phrase in user_lower for phrase in uninformative_phrases):
            context_msg = f"User provided uninformative response: '{user_message}'. This is a {st.session_state.assessment_result['risk_rating']} risk situation. Firmly explain that vague answers are not acceptable for high-risk situations. Ask them to provide a specific, detailed explanation of what caused the performance degradation."
            return get_llama_response(context_msg, model_database, criteria_database)
        
        # Refine the user's answer
        refined_reason = refine_text_with_llama(user_message, "reason")
        st.session_state.degradation_reason = refined_reason
        st.session_state.current_state = "mitigation_required"
        
        context_msg = f"User explained the degradation reason. Acknowledge it briefly, then simply ask: 'What's your mitigation plan to address this issue?' Keep it short - don't provide a long list of sub-questions."
        return get_llama_response(context_msg, model_database, criteria_database)
    
    # State: mitigation_required - waiting for mitigation plan (High/Critical only)
    elif state == "mitigation_required":
        uninformative_phrases = [
            'no idea', 'don\'t know', 'don\'t care', 'not sure', 
            'dunno', 'idk', 'whatever', 'none', 'n/a', 'na', 
            'will check', 'look into it', 'investigate'
        ]
        
        user_lower = user_message.lower().strip()
        
        if len(user_message.strip()) < 30 or any(phrase in user_lower for phrase in uninformative_phrases):
            context_msg = f"User provided uninformative mitigation plan: '{user_message}'. This is unacceptable for {st.session_state.assessment_result['risk_rating']} risk. Firmly explain they must provide a specific, actionable mitigation plan with concrete steps."
            return get_llama_response(context_msg, model_database, criteria_database)
        
        # Refine the user's answer
        refined_mitigation = refine_text_with_llama(user_message, "mitigation")
        st.session_state.mitigation_plan = refined_mitigation
        st.session_state.current_state = "assessment_complete"
        
        # Log to Google Sheets with refined reason and mitigation
        if log_assessment_to_gsheet_with_details(st.session_state.assessment_result, 
                                                   st.session_state.degradation_reason,
                                                   st.session_state.mitigation_plan):
            st.session_state.logged_to_gsheet = True
        
        result = st.session_state.assessment_result
        context_msg = f"Excellent. Assessment complete. You refined their mitigation plan to: '{refined_mitigation}'. Tell user everything is documented professionally and logged. Ask if they want to assess another model."
        return get_llama_response(context_msg, model_database, criteria_database)
    
    # State: assessment_complete - assessment done
    elif state == "assessment_complete":
        # Check if user wants to assess another model
        if any(word in user_message.lower() for word in ['assess', 'another', 'new model', 'next']):
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            st.session_state.assessment_result = None  # Clear old assessment
            st.session_state.logged_to_gsheet = False  # Clear log status
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
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
        /* Target user messages - move to right */
        [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessageAvatar"]:contains("👤")) {
            margin-left: auto;
            margin-right: 0;
        }
        
        /* Alternative approach - use nth-child if avatars alternate consistently */
        div[data-testid="stChatMessage"]:nth-child(even) {
            flex-direction: row-reverse;
            justify-content: flex-start;
        }
        
        div[data-testid="stChatMessage"]:nth-child(odd) {
            flex-direction: row;
            justify-content: flex-start;
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
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
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

    # Display chat messages with manual layout control
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            col1, col2 = st.columns([3, 1])
            with col2:
                st.markdown(f"**👤 You**")
                st.markdown(msg['content'])
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**🐱 Assistant**")
                st.markdown(msg['content'])
    
    # Display assessment card if available
    if (st.session_state.assessment_result and 
        st.session_state.current_state == "assessment_complete"):
        display_assessment_card(st.session_state.assessment_result)
        
        if st.session_state.get('logged_to_gsheet'):
            st.success("Assessment logged to Google Sheets")
        elif st.session_state.gsheet_client is None:
            st.info("Google Sheets not configured - assessment not logged")
        
        # Automatically show detailed report
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
