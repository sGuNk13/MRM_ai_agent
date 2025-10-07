"""
Model Assessment Module
AI-powered conversational model performance assessment
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from groq import Groq
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.google_sheets import (
    get_gsheet_client, 
    log_assessment_to_gsheet, 
    log_assessment_to_gsheet_with_details
)
from utils.assessment_logic import (
    find_model_info,
    process_model_assessment,
    extract_model_id,
    extract_number,
    refine_text_with_llama
)
from utils.visualization import display_assessment_card, generate_detailed_report

st.set_page_config(
    page_title="Model Assessment",
    page_icon="📊",
    layout="wide"
)

# Constants
MODEL_DATABASE_FILE = "mockup_database.xlsx"
CRITERIA_DATABASE_FILE = "mockup_criteria.xlsx"

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
    
    if 'degradation_reason' not in st.session_state:
        st.session_state.degradation_reason = None
    
    if 'mitigation_plan' not in st.session_state:
        st.session_state.mitigation_plan = None
    
    if 'logged_to_gsheet' not in st.session_state:
        st.session_state.logged_to_gsheet = False
    
    if 'groq_client' not in st.session_state:
        if 'GROQ_API_KEY' in st.secrets:
            st.session_state.groq_client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        else:
            st.session_state.groq_client = None
    
    if 'gsheet_client' not in st.session_state:
        st.session_state.gsheet_client = get_gsheet_client()

# ============================================================================
# AI CONVERSATION
# ============================================================================

def build_context(model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Build rich context for Llama"""
    state = st.session_state.current_state
    model_id = st.session_state.model_id
    
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
3. performance_input -> Model selected, waiting for current performance value
4. reason_required -> For High/Critical risk only - asking for degradation reason
5. mitigation_required -> For High/Critical risk only - asking for mitigation plan
6. assessment_complete -> Assessment done

YOUR ROLE:
- Have natural, helpful conversations
- Guide users through the assessment process
- When in model_input state, simply ask "Which model ID would you like to assess?" - DO NOT mention specific model IDs
- When asking for mitigation plans, keep it simple - don't provide lengthy checklists
- Be conversational and concise (2-3 sentences)
- NEVER reference previous assessments
- Each assessment is independent

CRITICAL INSTRUCTIONS:
- Only use models from the exact list provided above
- Do not invent or suggest models that are not listed
- Focus only on the CURRENT assessment"""
    
    return context

def get_llama_response(user_message: str, model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Get natural response from Llama"""
    if st.session_state.groq_client is None:
        return "I need the GROQ_API_KEY to respond. Please configure it in Streamlit secrets."
    
    try:
        context = build_context(model_database, criteria_database)
        messages = [{"role": "system", "content": context}]
        
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

def process_user_input(user_message: str, model_database: pd.DataFrame, criteria_database: pd.DataFrame) -> str:
    """Process user input and manage state transitions"""
    
    if any(word in user_message.lower() for word in ['reset', 'start over', 'restart']):
        st.session_state.current_state = "greeting"
        st.session_state.model_id = None
        st.session_state.assessment_result = None
        st.session_state.degradation_reason = None
        st.session_state.mitigation_plan = None
        return get_llama_response("User wants to reset/start over. Acknowledge the reset and ask what they'd like to do.", model_database, criteria_database)
    
    state = st.session_state.current_state
    
    if state == "greeting":
        user_lower = user_message.lower()
        wants_assessment = any(word in user_lower for word in ['assess', 'check', 'evaluate', 'test', 'analyze'])
        
        if wants_assessment:
            # CRITICAL: Clear ALL state before starting new assessment
            st.session_state.assessment_result = None
            st.session_state.logged_to_gsheet = False
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
            st.session_state.model_id = None
            
            found_model = extract_model_id(user_message, model_database)
            
            if found_model:
                model_info = find_model_info(found_model, model_database)
                st.session_state.model_id = found_model
                st.session_state.current_state = "performance_input"
                
                # CRITICAL: Clear conversation history to prevent metric confusion
                st.session_state.messages = []
                
                # Return HARDCODED response - don't let Llama mess it up
                return f"You've selected {found_model}, which has a {model_info['metric']} metric and a baseline performance of {model_info['baseline_performance']}.\n\nTo proceed with the assessment, could you please provide the current {model_info['metric']} performance value?"
            else:
                st.session_state.current_state = "model_input"
                st.session_state.model_id = None
                st.session_state.assessment_result = None
                
                context_msg = "User wants to assess a model but didn't specify which one. Ask them which model ID they want to assess."
                return get_llama_response(context_msg, model_database, criteria_database)
        else:
            return get_llama_response(user_message, model_database, criteria_database)
    
    elif state == "model_input":
        # CRITICAL: Clear old assessment immediately when entering model_input
        st.session_state.assessment_result = None
        st.session_state.logged_to_gsheet = False
        
        found_model = extract_model_id(user_message, model_database)
        
        if not found_model:
            clean_input = user_message.strip().upper()
            if not model_database[model_database['model_id'].str.upper() == clean_input].empty:
                found_model = clean_input
        
        if found_model:
            model_info = find_model_info(found_model, model_database)
            st.session_state.model_id = found_model
            st.session_state.current_state = "performance_input"
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
            
            # CRITICAL: Clear conversation history to prevent metric confusion
            st.session_state.messages = []
            
            # Return HARDCODED response - don't let Llama paraphrase it
            return f"You've selected {found_model}, which has a {model_info['metric']} metric and a baseline performance of {model_info['baseline_performance']}.\n\nTo proceed with the assessment, could you please provide the current {model_info['metric']} performance value?"
        else:
            context_msg = f"'{user_message}' is not a valid model ID. Ask user to provide a valid model ID from the database."
            return get_llama_response(context_msg, model_database, criteria_database)
    
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
        
        try:
            assessment = process_model_assessment(
                st.session_state.model_id,
                performance,
                model_database,
                criteria_database
            )
            
            st.session_state.assessment_result = assessment.to_dict()
            risk_rating = st.session_state.assessment_result['risk_rating']
            
            if risk_rating in ['High', 'Critical']:
                st.session_state.current_state = "reason_required"
                context_msg = f"Assessment shows {risk_rating} risk with {assessment.deviation_percentage:.2f}% degradation. Ask user to explain the REASON for this performance degradation. Vague answers are not acceptable."
                return get_llama_response(context_msg, model_database, criteria_database)
            else:
                st.session_state.current_state = "assessment_complete"
                
                if log_assessment_to_gsheet(st.session_state.assessment_result, st.session_state.gsheet_client):
                    st.session_state.logged_to_gsheet = True
                
                # AUTO-CLEAR: Reset for next assessment
                st.session_state.model_id = None
                
                result = st.session_state.assessment_result
                context_msg = f"Assessment complete! Model {result['model_id']}: Current {result['current']} vs Baseline {result['baseline']}. Deviation: {result['deviation']:.2f}%, Risk: {result['risk_rating']}. Tell user assessment is complete (results shown below). Ask if they want to assess another model."
                return get_llama_response(context_msg, model_database, criteria_database)
                
        except Exception as e:
            return f"Error during assessment: {str(e)}"

    elif state == "reason_required":
        uninformative_phrases = ['no idea', 'don\'t know', 'don\'t care', 'not sure', 'dunno', 'idk', 'whatever', 'none', 'n/a', 'na']
        user_lower = user_message.lower().strip()
        
        if len(user_message.strip()) < 20 or any(phrase in user_lower for phrase in uninformative_phrases):
            context_msg = f"User provided uninformative response: '{user_message}'. This is a {st.session_state.assessment_result['risk_rating']} risk situation. Firmly explain that vague answers are not acceptable."
            return get_llama_response(context_msg, model_database, criteria_database)
        
        refined_reason = refine_text_with_llama(user_message, "reason", st.session_state.groq_client)
        st.session_state.degradation_reason = refined_reason
        st.session_state.current_state = "mitigation_required"
        
        context_msg = f"User explained the degradation reason. Acknowledge it briefly, then simply ask: 'What's your mitigation plan to address this issue?'"
        return get_llama_response(context_msg, model_database, criteria_database)
    
    elif state == "mitigation_required":
        uninformative_phrases = ['no idea', 'don\'t know', 'don\'t care', 'not sure', 'dunno', 'idk', 'whatever', 'none', 'n/a', 'na', 'will check', 'look into it', 'investigate']
        user_lower = user_message.lower().strip()
        
        if len(user_message.strip()) < 30 or any(phrase in user_lower for phrase in uninformative_phrases):
            context_msg = f"User provided uninformative mitigation plan: '{user_message}'. This is unacceptable for {st.session_state.assessment_result['risk_rating']} risk. Firmly explain they must provide a specific, actionable mitigation plan."
            return get_llama_response(context_msg, model_database, criteria_database)
        
        refined_mitigation = refine_text_with_llama(user_message, "mitigation", st.session_state.groq_client)
        st.session_state.mitigation_plan = refined_mitigation
        st.session_state.current_state = "assessment_complete"
        
        if log_assessment_to_gsheet_with_details(st.session_state.assessment_result, 
                                                   st.session_state.degradation_reason,
                                                   st.session_state.mitigation_plan,
                                                   st.session_state.gsheet_client):
            st.session_state.logged_to_gsheet = True
        
        # AUTO-CLEAR: Reset for next assessment
        st.session_state.model_id = None
        st.session_state.degradation_reason = None
        st.session_state.mitigation_plan = None
        
        result = st.session_state.assessment_result
        context_msg = f"Excellent. Assessment complete. Tell user everything is documented and logged. Ask if they want to assess another model."
        return get_llama_response(context_msg, model_database, criteria_database)
    
    elif state == "assessment_complete":
        if any(word in user_message.lower() for word in ['assess', 'another', 'new model', 'next']):
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            st.session_state.assessment_result = None
            st.session_state.logged_to_gsheet = False
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
            return get_llama_response("User wants to assess another model. Ask which model they'd like to assess next.", model_database, criteria_database)
        else:
            return get_llama_response(user_message, model_database, criteria_database)
    
    return get_llama_response(user_message, model_database, criteria_database)

# ============================================================================
# MAIN
# ============================================================================

def main():
    initialize_session_state()
    
    st.title("📊 Model Assessment")
    st.caption("Powered by Llama 3.1 via Groq")
    
    # Load databases
    try:
        model_database = pd.read_excel(MODEL_DATABASE_FILE)
        criteria_database = pd.read_excel(CRITERIA_DATABASE_FILE)
    except Exception as e:
        st.error(f"Error loading databases: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Assessment Controls")
        
        st.success(f"📚 {len(model_database)} models loaded")
        st.success(f"📏 {len(criteria_database)} criteria loaded")
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_state = "greeting"
            st.session_state.model_id = None
            st.session_state.assessment_result = None
            st.session_state.degradation_reason = None
            st.session_state.mitigation_plan = None
            st.rerun()
        
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("streamlit_app.py")
        
        st.divider()
        st.subheader("Current Status")
        st.write(f"**State:** {st.session_state.current_state}")
        if st.session_state.model_id:
            st.write(f"**Model:** {st.session_state.model_id}")
    
    # Chat messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message(msg['role'], avatar="👤"):
                st.markdown(msg['content'])
        else:
            with st.chat_message(msg['role'], avatar="🐱"):
                st.markdown(msg['content'])
    
    # Display assessment
    if (st.session_state.assessment_result and 
        st.session_state.current_state == "assessment_complete"):
        display_assessment_card(st.session_state.assessment_result)
        
        if st.session_state.get('logged_to_gsheet'):
            st.success("✅ Assessment logged to Google Sheets")
        
        report = generate_detailed_report(st.session_state.assessment_result)
        st.markdown(report)
        st.download_button(
            "📥 Download Report",
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
