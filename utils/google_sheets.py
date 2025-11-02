"""
Google Sheets utilities - extracted from main app
"""

import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime

@st.cache_resource
def get_gsheet_client():
    """Get authenticated Google Sheets client"""
    if 'gcp_service_account' not in st.secrets or 'GOOGLE_SHEET_ID' not in st.secrets:
        return None
    
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["gcp_service_account"], scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets connection error: {str(e)}")
        return None

def log_assessment_to_gsheet(assessment_dict, gsheet_client) -> bool:
    """Log assessment to Google Sheets with monthly sheet selection"""
    if gsheet_client is None:
        return False
    
    try:
        sheet = gsheet_client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        current_month = datetime.now().strftime('%B')
        
        try:
            worksheet = sheet.worksheet(current_month)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=current_month, rows=1000, cols=12)
            headers = ['timestamp', 'model_id', 'metric', 'baseline', 
                      'current_performance', 'deviation', 'deviation_risk', 
                      'standard_risk', 'final_risk_rating']
            worksheet.append_row(headers)
        
        row_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            assessment_dict['model_id'].lower(),  # Normalize to lowercase
            assessment_dict['metric'],
            assessment_dict['baseline'],
            assessment_dict['current'],
            f"{assessment_dict['deviation']:.2f}%",
            assessment_dict.get('deviation_risk', 'N/A'),
            assessment_dict.get('standard_risk', 'N/A'),
            assessment_dict['risk_rating']
        ]
        
        worksheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"Failed to log to Google Sheets: {str(e)}")
        return False

def log_assessment_to_gsheet_with_details(assessment_dict, reason, mitigation, gsheet_client) -> bool:
    """Log assessment with degradation reason and mitigation plan"""
    if gsheet_client is None:
        return False
    
    try:
        sheet = gsheet_client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        current_month = datetime.now().strftime('%B')
        
        try:
            worksheet = sheet.worksheet(current_month)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=current_month, rows=1000, cols=14)
            headers = ['timestamp', 'model_id', 'metric', 'baseline', 
                      'current_performance', 'deviation', 'deviation_risk',
                      'standard_risk', 'final_risk_rating',
                      'degradation_reason', 'mitigation_plan']
            worksheet.append_row(headers)
        
        row_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            assessment_dict['model_id'].lower(),  # Normalize to lowercase
            assessment_dict['metric'],
            assessment_dict['baseline'],
            assessment_dict['current'],
            f"{assessment_dict['deviation']:.2f}%",
            assessment_dict.get('deviation_risk', 'N/A'),
            assessment_dict.get('standard_risk', 'N/A'),
            assessment_dict['risk_rating'],
            reason if reason else 'N/A',
            mitigation if mitigation else 'N/A'
        ]
        
        worksheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"Failed to log to Google Sheets: {str(e)}")
        return False

def load_all_assessments(gsheet_client):
    """Load all assessment data from all monthly sheets"""
    if gsheet_client is None:
        return pd.DataFrame()
    
    try:
        sheet = gsheet_client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
        all_data = []
        
        for worksheet in sheet.worksheets():
            try:
                data = worksheet.get_all_records()
                if data:
                    all_data.extend(data)
            except:
                continue
        
        return pd.DataFrame(all_data)
    except Exception as e:
        st.error(f"Error loading assessments: {str(e)}")
        return pd.DataFrame()

def get_quick_stats(gsheet_client):
    """Get quick statistics for landing page"""
    df = load_all_assessments(gsheet_client)
    
    if df.empty:
        return {
            'total': 0,
            'this_month': 0,
            'high_risk': 0,
            'high_risk_change': 0,
            'avg_deviation': 0.0
        }
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    current_month = datetime.now().month
    
    this_month_data = df[df['timestamp'].dt.month == current_month]
    last_month_data = df[df['timestamp'].dt.month == current_month - 1]
    
    # Use final_risk_rating if available, otherwise fall back to risk_rating
    risk_col = 'final_risk_rating' if 'final_risk_rating' in df.columns else 'risk_rating'
    
    high_risk_this_month = len(this_month_data[this_month_data[risk_col].isin(['High', 'Critical'])])
    high_risk_last_month = len(last_month_data[last_month_data[risk_col].isin(['High', 'Critical'])])
    
    if 'deviation' in df.columns:
        df['deviation_numeric'] = pd.to_numeric(df['deviation'].astype(str).str.rstrip('%'), errors='coerce')
        avg_deviation = df['deviation_numeric'].mean()
    else:
        avg_deviation = 0.0
    
    return {
        'total': len(df),
        'this_month': len(this_month_data),
        'high_risk': high_risk_this_month,
        'high_risk_change': high_risk_this_month - high_risk_last_month,
        'avg_deviation': avg_deviation
    }
