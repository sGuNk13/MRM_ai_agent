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
    deviation_risk_rating: str
    standard_risk_rating: str
    final_risk_rating: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            'model_id': self.model_id,
            'metric': self.metric,
            'baseline': self.baseline_performance,
            'current': self.current_performance,
            'deviation': self.deviation_percentage,
            'deviation_risk': self.deviation_risk_rating,
            'standard_risk': self.standard_risk_rating,
            'risk_rating': self.final_risk_rating,
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
    """Calculate risk rating based on performance deviation (Fold 1)"""
    # Get thresholds from criteria (already in decimal form: 0.1, 0.15, 0.2)
    low_threshold = criteria.get('low_threshold', 0.1) * 100  # Convert to percentage
    medium_threshold = criteria.get('medium_threshold', 0.15) * 100
    high_threshold = criteria.get('high_threshold', 0.2) * 100
    
    # Positive deviation = improvement = Very Low Risk
    if deviation_percentage > 0:
        return "Very Low"
    
    # Negative deviation = degradation (more negative = worse)
    abs_degradation = abs(deviation_percentage)
    
    if abs_degradation <= low_threshold:
        return "Low"
    elif abs_degradation <= medium_threshold:
        return "Medium"
    elif abs_degradation <= high_threshold:
        return "High"
    else:
        return "Very High"

def calculate_deviation_risk_for_mape_nmae(baseline_standard_risk: str, current_standard_risk: str) -> str:
    """Calculate deviation risk for MAPE/NMAE using Sheet2 logic"""
    # Complete mapping based on the table
    risk_mapping = {
        # Baseline: Very Low
        ("Very Low", "Very Low"): "Low",
        ("Very Low", "Low"): "Low",
        ("Very Low", "Medium"): "Medium",
        ("Very Low", "High"): "High",
        ("Very Low", "Very High"): "High",
        
        # Baseline: Low
        ("Low", "Very Low"): "Low",  # Improvement
        ("Low", "Low"): "Low",
        ("Low", "Medium"): "Medium",
        ("Low", "High"): "High",
        ("Low", "Very High"): "High",
        
        # Baseline: Medium
        ("Medium", "Very Low"): "Low",  # Significant improvement
        ("Medium", "Low"): "Low",       # Improvement
        ("Medium", "Medium"): "Medium",
        ("Medium", "High"): "High",
        ("Medium", "Very High"): "High",
        
        # Baseline: High (any current is improvement or stable)
        ("High", "Very Low"): "Low",
        ("High", "Low"): "Low",
        ("High", "Medium"): "Medium",
        ("High", "High"): "Medium",
        ("High", "Very High"): "High",
        
        # Baseline: Very High (any current is improvement or stable)
        ("Very High", "Very Low"): "Low",
        ("Very High", "Low"): "Low",
        ("Very High", "Medium"): "Medium",
        ("Very High", "High"): "Medium",
        ("Very High", "Very High"): "High",
    }
    
    # Get the deviation risk from mapping
    key = (baseline_standard_risk, current_standard_risk)
    
    # Return mapped value or default
    return risk_mapping.get(key, "Medium")

def calculate_standard_risk_rating(current_performance: float, standard_criteria: Dict) -> str:
    """Calculate risk rating based on absolute performance vs standard (Fold 2)"""
    high_risk = standard_criteria.get('high_risk', 0.2)
    medium_risk = standard_criteria.get('medium_risk', 0.3)
    low_risk = standard_criteria.get('low_risk', 0.5)
    very_low = standard_criteria.get('very_low', 0.6)
    
    if current_performance <= high_risk:
        return "Very High"
    elif current_performance <= medium_risk:
        return "High"
    elif current_performance <= low_risk:
        return "Medium"
    elif current_performance <= very_low:
        return "Low"
    else:
        return "Very Low"

def calculate_standard_risk_rating_inverted(current_performance: float, standard_criteria: Dict) -> str:
    """Calculate risk rating for ERROR metrics (MAPE/NMAE) where higher is worse"""
    high_risk = standard_criteria.get('high_risk', 0.2)
    medium_risk = standard_criteria.get('medium_risk', 0.3)
    low_risk = standard_criteria.get('low_risk', 0.5)
    very_low = standard_criteria.get('very_low', 0.6)
    
    # Inverted logic: higher values = worse performance
    if current_performance >= very_low:
        return "Very High"
    elif current_performance >= low_risk:
        return "High"
    elif current_performance >= medium_risk:
        return "Medium"
    elif current_performance >= high_risk:
        return "Low"
    else:
        return "Very Low"

def get_worst_risk_rating(risk1: str, risk2: str) -> str:
    """Compare two risk ratings and return the worst one"""
    risk_hierarchy = {
        "Very Low": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }
    
    level1 = risk_hierarchy.get(risk1, 0)
    level2 = risk_hierarchy.get(risk2, 0)
    
    worst_level = max(level1, level2)
    
    for risk, level in risk_hierarchy.items():
        if level == worst_level:
            return risk
    
    return "Low"

def process_model_assessment(model_id: str, current_performance: float, 
                            model_database: pd.DataFrame, 
                            criteria_database: pd.DataFrame,
                            standard_database: pd.DataFrame) -> ModelPerformance:
    """Process complete model assessment with two-fold evaluation"""
    # Get model information
    model_info = find_model_info(model_id, model_database)
    if not model_info:
        raise ValueError(f"Model '{model_id}' not found in database")
    
    baseline_performance = model_info.get('baseline_performance', model_info.get('baseline'))
    metric = model_info.get('metric')
    standard_name = model_info.get('standard')
    
    if baseline_performance is None or metric is None:
        raise ValueError(f"Incomplete model info for '{model_id}'")
    
    # Check if metric is MAPE or NMAE (case-insensitive) - ERROR METRICS
    is_mape_nmae = metric.strip().upper() in ['MAPE', 'NMAE']
    
    # Calculate deviation percentage
    # For error metrics (MAPE/NMAE): positive deviation = worse performance (error increased)
    # For other metrics: positive deviation = better performance
    deviation_percentage = ((current_performance - baseline_performance) / baseline_performance) * 100
    
    # FOLD 2: Absolute performance vs standard (calculate first for MAPE/NMAE)
    standard_risk = "Very Low"  # Default if no standard defined
    baseline_standard_risk = "Very Low"  # For MAPE/NMAE comparison
    
    if standard_name and not pd.isna(standard_name):
        standard_row = standard_database[
            standard_database['standard'].str.strip().str.lower() == standard_name.strip().lower()
        ]
        
        if not standard_row.empty:
            standard_criteria = standard_row.iloc[0].to_dict()
            
            if is_mape_nmae:
                # Use INVERTED logic for error metrics (higher = worse)
                standard_risk = calculate_standard_risk_rating_inverted(current_performance, standard_criteria)
                baseline_standard_risk = calculate_standard_risk_rating_inverted(baseline_performance, standard_criteria)
            else:
                # Use normal logic for performance metrics (higher = better)
                standard_risk = calculate_standard_risk_rating(current_performance, standard_criteria)
    
    # FOLD 1: Deviation from baseline
    if is_mape_nmae:
        # For MAPE/NMAE: Use Sheet2 logic - compare baseline vs current standard risks
        deviation_risk = calculate_deviation_risk_for_mape_nmae(baseline_standard_risk, standard_risk)
    else:
        # For other metrics: Use Sheet1 logic - normal deviation calculation
        criteria_row = criteria_database[
            criteria_database['metric'].str.strip().str.lower() == metric.strip().lower()
        ]
        
        if criteria_row.empty:
            raise ValueError(f"Risk criteria not found for metric '{metric}'")
        
        criteria = criteria_row.iloc[0].to_dict()
        deviation_risk = calculate_risk_rating(deviation_percentage, criteria)
    
    # FINAL VERDICT: Take worst of both
    final_risk = get_worst_risk_rating(deviation_risk, standard_risk)
    
    return ModelPerformance(
        model_id=model_id,
        metric=metric,
        baseline_performance=baseline_performance,
        current_performance=current_performance,
        deviation_percentage=deviation_percentage,
        deviation_risk_rating=deviation_risk,
        standard_risk_rating=standard_risk,
        final_risk_rating=final_risk
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
            prompt = f"""Refine this text to be more formal and professional. Always improve the grammar and structure to make it business-appropriate. Keep the original meaning.

Original: {text}

Refined:"""
        else:  # mitigation
            prompt = f"""Refine this text to be more formal and professional. Always improve the grammar and structure to make it business-appropriate. Keep the original meaning.

Original: {text}

Refined:"""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional editor. Always refine text to be more formal and professional while keeping it concise. Improve grammar, structure, and clarity. Keep the same meaning but make it sound more business-appropriate."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        refined = completion.choices[0].message.content.strip()
        
        if len(refined) > len(text) * 2:
            return text
        
        return refined if refined else text
        
    except Exception as e:
        return text
