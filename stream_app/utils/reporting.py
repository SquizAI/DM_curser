import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
from io import BytesIO
import os
import streamlit as st
from datetime import datetime

def generate_pdf_report(
    rules_df: pd.DataFrame,
    insights: list,
    visualizations: dict,
    report_title: str = "Association Rule Mining Report",
    include_charts: bool = True
) -> bytes:
    """
    Generate a PDF report with association rule mining results.
    
    Args:
        rules_df: DataFrame containing the association rules
        insights: List of textual insights
        visualizations: Dictionary of visualization figures/charts
        report_title: Title for the report
        include_charts: Whether to include charts in the report
        
    Returns:
        Bytes of the PDF file
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Set up fonts
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, report_title, ln=True, align="C")
    pdf.ln(10)
    
    # Date
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 5, f"This report analyzes {len(rules_df)} association rules mined from the transaction data. The analysis highlights key product relationships and provides actionable business insights.")
    pdf.ln(10)
    
    # Key Metrics
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Arial", "", 11)
    
    metrics = [
        f"Total Rules: {len(rules_df)}",
        f"Average Confidence: {rules_df['confidence'].mean():.2f}",
        f"Average Lift: {rules_df['lift'].mean():.2f}",
        f"Max Lift: {rules_df['lift'].max():.2f}"
    ]
    
    for metric in metrics:
        pdf.cell(0, 5, metric, ln=True)
    pdf.ln(10)
    
    # Top Rules Table
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Top 5 Rules by Lift", ln=True)
    
    # Table header
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 7, "Antecedents", border=1)
    pdf.cell(50, 7, "Consequents", border=1)
    pdf.cell(30, 7, "Support", border=1)
    pdf.cell(30, 7, "Confidence", border=1)
    pdf.cell(30, 7, "Lift", border=1, ln=True)
    
    # Table rows
    pdf.set_font("Arial", "", 9)
    top_rules = rules_df.sort_values('lift', ascending=False).head(5)
    
    for _, row in top_rules.iterrows():
        # Format antecedents and consequents for display
        antecedents = ', '.join(list(row['antecedents'])) if hasattr(row['antecedents'], '__iter__') else str(row['antecedents'])
        consequents = ', '.join(list(row['consequents'])) if hasattr(row['consequents'], '__iter__') else str(row['consequents'])
        
        # Truncate if too long
        antecedents = (antecedents[:47] + '...') if len(antecedents) > 50 else antecedents
        consequents = (consequents[:47] + '...') if len(consequents) > 50 else consequents
        
        pdf.cell(50, 7, antecedents, border=1)
        pdf.cell(50, 7, consequents, border=1)
        pdf.cell(30, 7, f"{row['support']:.3f}", border=1)
        pdf.cell(30, 7, f"{row['confidence']:.3f}", border=1)
        pdf.cell(30, 7, f"{row['lift']:.3f}", border=1, ln=True)
    
    pdf.ln(10)
    
    # Key Insights
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Insights", ln=True)
    pdf.set_font("Arial", "", 11)
    
    for i, insight in enumerate(insights[:5], 1):  # Limit to top 5 insights
        pdf.multi_cell(0, 5, f"{i}. {insight}")
        pdf.ln(5)
    
    pdf.ln(10)
    
    # Generate and include charts if requested
    if include_charts and visualizations:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Visualizations", ln=True)
        
        for title, fig in visualizations.items():
            if fig:
                # Convert matplotlib figure to image
                img_buf = BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                
                # Add figure to PDF
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, title, ln=True)
                pdf.image(img_buf, x=10, w=180)
                pdf.ln(10)
    
    # Recommendations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    
    recommendations = [
        "Consider bundling frequently co-occurring items to increase average order value.",
        "Optimize store layout based on the strongest associations to encourage additional purchases.",
        "Develop targeted marketing campaigns based on the discovered patterns.",
        "Update inventory management to ensure associated items are well-stocked."
    ]
    
    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 5, f"{i}. {rec}")
        pdf.ln(2)
    
    # Convert to bytes
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_bytes = pdf_output.getvalue()
    
    return pdf_bytes

def get_download_link(pdf_bytes, filename="association_rules_report.pdf", text="Download PDF Report"):
    """
    Generate a download link for the PDF report
    
    Args:
        pdf_bytes: Bytes of the PDF file
        filename: Name for the downloaded file
        text: Text to display for the download link
        
    Returns:
        HTML string with the download link
    """
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href 