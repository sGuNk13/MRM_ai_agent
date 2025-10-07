"""
Assessment logic - extracted from main app
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from groq import Groq
import streamlit as st

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
        r'\b[A-Z]{3}_[A-Z]{3}_\d+(?:_\d+)*\b',
        r'\bmodel_id_\d+\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, message.lower())
        for match in matches:
            if not model_database[model_database['model_id'].str.lower() == match.lower()].empty:
                actual_id = model_database[model_database['model_id'].str.lower() == match.lower()].iloc[0]['model_id']
                return actual_id
    
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

def refine_text_with_llama(text: str, field_type: str, groq_client) -> str:
    """Use Llama to refine user's answer into formal professional English"""
    if groq_client is None:
        return text
    
    try:
        if field_type == "reason":
            prompt = f"""Refine this explanation into 1-2 clear, professional sentences. Keep the original meaning, just improve grammar and formality. Do not add extra details.

Original: {text}

Refined:"""
        else:
            prompt = f"""Refine this mitigation plan into 2-3 clear, professional sentences. Keep it concise - just improve grammar and formality. Do not expand or add steps that weren't mentioned.

Original: {text}

Refined:"""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You refine text to be professional and grammatically correct. Keep responses BRIEF - only improve the original, don't expand it."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        
        refined = completion.choices[0].message.content.strip()
        
        if len(refined) > len(text) * 2:
            return text
        
        return refined if refined else text
        
    except Exception as e:
        return text
