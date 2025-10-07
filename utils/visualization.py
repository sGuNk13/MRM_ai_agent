"""
Visualization functions
"""

import streamlit as st
from datetime import datetime

def display_assessment_card(assessment_dict):
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

def generate_detailed_report(assessment_dict) -> str:
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
