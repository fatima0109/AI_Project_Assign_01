# File: visualization_dashboard.py
"""
Interactive visualization dashboard (optional extra credit).
"""

import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np


class VisualizationDashboard:
    def __init__(self, results_dir):
        """Initialize dashboard path"""
        self.results_dir = Path(results_dir)

    def create_interactive_dashboard(self):
        """Create interactive Streamlit dashboard"""
        st.set_page_config(layout="wide")
        st.title("📊 Model Performance Dashboard")

        # --- LOAD RESULTS ---
        comparison_file = self.results_dir / "model_comparison.csv"

        if not comparison_file.exists():
            st.error(f"Missing file: {comparison_file}")
            return

        comparison_df = pd.read_csv(comparison_file)

        # --- SIDEBAR FILTERS ---
        st.sidebar.header("Filters")
        selected_models = st.sidebar.multiselect(
            "Select Models",
            options=comparison_df["Model"].tolist(),
            default=comparison_df["Model"].tolist(),
        )

        filtered_df = comparison_df[comparison_df["Model"].isin(selected_models)]

        # --- DASHBOARD COLUMNS ---
        col1, col2 = st.columns(2)

        # --------------------------------------------------------------------
        #                           BAR CHART
        # --------------------------------------------------------------------
        with col1:
            st.subheader("Performance Metrics (Bar Chart)")

            fig = go.Figure()
            metrics = ["Accuracy", "F1-Macro", "Precision", "Recall"]

            for metric in metrics:
                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=filtered_df["Model"],
                        y=filtered_df[metric],
                        text=filtered_df[metric].round(3),
                        textposition="auto",
                    )
                )

            fig.update_layout(
                title="Model Performance Comparison",
                barmode="group",
                yaxis=dict(range=[0, 1]),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------------
        #                           RADAR CHART
        # --------------------------------------------------------------------
        with col2:
            st.subheader("Radar Chart Comparison")

            categories = ["Accuracy", "F1-Macro", "Precision", "Recall"]
            fig = go.Figure()

            for _, row in filtered_df.iterrows():
                fig.add_trace(
                    go.Scatterpolar(
                        r=[row[m] for m in categories],
                        theta=categories,
                        fill="toself",
                        name=row["Model"],
                    )
                )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

        # ====================================================================
        #                      CONFUSION MATRICES
        # ====================================================================
        st.header("Confusion Matrices")

        for model in selected_models:
            cm_path = self.results_dir / f"{model}_confusion_matrix.csv"

            if cm_path.exists():
                st.subheader(f"{model} - Confusion Matrix")
                cm = pd.read_csv(cm_path, index_col=0)

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"{model} Confusion Matrix",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No confusion matrix found for {model}")

        # ====================================================================
        #                         ERROR ANALYSIS
        # ====================================================================
        st.header("Error Analysis (Misclassified Samples)")

        error_file = self.results_dir / "error_analysis.csv"

        if error_file.exists():
            errors_df = pd.read_csv(error_file)
            st.dataframe(errors_df)
        else:
            st.info("No error analysis file found (expected: error_analysis.csv).")

        # ====================================================================
        #                        EXPORT PDF REPORT
        # ====================================================================
        st.sidebar.header("Export")

        if st.sidebar.button("Generate PDF Report"):
            self.generate_pdf_report()
            st.success("PDF Report Generated Successfully!")

    # ==================================================================================
    #                      PDF REPORT GENERATION
    # ==================================================================================
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        from fpdf import FPDF

        comparison_df = pd.read_csv(self.results_dir / "model_comparison.csv")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Model Performance Report", ln=1, align="C")

        pdf.set_font("Arial", size=12)
        pdf.ln(5)
        pdf.cell(200, 10, txt="Overall Model Metrics:", ln=1)

        # Write comparison table into PDF
        pdf.set_font("Arial", size=10)

        # Header
        for col in comparison_df.columns:
            pdf.cell(40, 10, txt=str(col), border=1)
        pdf.ln()

        # Rows
        for _, row in comparison_df.iterrows():
            for col in comparison_df.columns:
                pdf.cell(40, 10, txt=str(row[col]), border=1)
            pdf.ln()

        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(200, 10, txt="Generated by VisualizationDashboard", ln=1, align="C")

        # Save file
        pdf.output(str(self.results_dir / "performance_report.pdf"))

if __name__ == "__main__":
    print("dashboard.py executed successfully!")
