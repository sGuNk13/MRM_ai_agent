"""
Visualization functions
"""

import streamlit as st
from datetime import datetime

def display_assessment_card(assessment_dict):
    """Display assessment result card with two-fold assessment"""
    risk_colors = {
        "Very Low": "#2ECC71",
        "Low": "#27AE60",
        "Medium": "#F39C12",
        "High": "#E67E22",
        "Critical": "#C0392B"
    }
    
    final_risk_color = risk_colors.get(assessment_dict['risk_rating'], "#95a5a6")
    deviation_risk_color = risk_colors.get(assessment_dict.get('deviation_risk', 'Low'), "#95a5a6")
    standard_risk_color = risk_colors.get(assessment_dict.get('standard_risk', 'Low'), "#95a5a6")
    
    trend = "improved" if assessment_dict['deviation'] > 0 else "degraded" if assessment_dict['deviation'] < 0 else "unchanged"
    
    st.markdown(f"""
    <div style="background: white; padding: 25px; border-radius: 15px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0;">
        <h3 style="color: {final_risk_color}; margin-top: 0;">Assessment Results</h3>
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
            <tr style="border-bottom: 2px solid #ddd; background-color: #f8f9fa;">
                <td colspan="2" style="padding: 15px 0; font-weight: bold; font-size: 16px;">
                    Two-Fold Risk Assessment
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Fold 1 - Deviation Risk:</td>
                <td style="padding: 10px 0;">
                    <span style="background: {deviation_risk_color}; color: white; padding: 5px 12px; 
                                 border-radius: 12px; font-weight: bold;">{assessment_dict.get('deviation_risk', 'N/A')}</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px 0; font-weight: bold;">Fold 2 - Standard Risk:</td>
                <td style="padding: 10px 0;">
                    <span style="background: {standard_risk_color}; color: white; padding: 5px 12px; 
                                 border-radius: 12px; font-weight: bold;">{assessment_dict.get('standard_risk', 'N/A')}</span>
                </td>
            </tr>
            <tr style="background-color: #fff3cd;">
                <td style="padding: 10px 0; font-weight: bold;">Final Risk Rating:</td>
                <td style="padding: 10px 0;">
                    <span style="background: {final_risk_color}; color: white; padding: 5px 12px; 
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
        "Very Low": "Continue standard monitoring procedures with periodic performance reviews.",
        "Low": "Continue standard monitoring procedures with periodic performance reviews.",
        "Medium": "Implement enhanced monitoring and conduct root cause analysis within the next review cycle.",
        "High": "Immediate investigation required. Initiate model retraining process and validate data quality.",
        "Critical": "Emergency response required. Consider model rollback, immediate retraining, and stakeholder notification."
    }
    
    action = risk_actions[result['risk_rating']]
    
    # Two-fold assessment section
    fold1_risk = result.get('deviation_risk', 'N/A')
    fold2_risk = result.get('standard_risk', 'N/A')
    
    report = f"""
# Model Performance Assessment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

The model's performance has **{status}**, with a **{abs_deviation:.2f}% {direction}** in {result['metric']}. 

**Two-Fold Risk Assessment:**
- **Fold 1 (Deviation from Baseline):** {fold1_risk}
- **Fold 2 (Absolute Performance vs Standard):** {fold2_risk}
- **Final Risk Rating:** {result['risk_rating']} (worst of both folds)

---

## Model Information

- **Model ID:** {result['model_id']}
- **Metric:** {result['metric']}
- **Baseline Performance:** {result['baseline']}
- **Current Performance:** {result['current']}
- **Deviation:** {result['deviation']:.2f}%

---

## Risk Assessment Details

### Fold 1: Deviation from Baseline
**Risk Level:** {fold1_risk}

This assessment compares current performance against the model's historical baseline to detect degradation over time.

### Fold 2: Absolute Performance vs Industry Standard
**Risk Level:** {fold2_risk}

This assessment evaluates current performance against regulatory/industry standards to ensure minimum acceptable performance thresholds are met.

### Final Verdict
**Risk Rating:** {result['risk_rating']}

The final risk rating is determined by taking the worst (highest) risk level from both assessments, ensuring conservative risk management.

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
- {"Consider model recalibration if standard thresholds are violated" if fold2_risk in ['High', 'Critical'] else "Monitor for trend continuation"}

---

**Report ID:** {result['model_id']}-{datetime.now().strftime('%Y%m%d%H%M%S')}
"""
    
    return report
