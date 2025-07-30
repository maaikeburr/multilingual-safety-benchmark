"""Report generation utilities"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ReportGenerator:
    """Generate evaluation reports in various formats"""
    
    def __init__(self, config: Any):
        self.config = config
        self.template_dir = Path(__file__).parent.parent / "templates"
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def generate(
        self,
        results: Dict[str, Any],
        output_dir: str,
        format: str = "html"
    ) -> str:
        """
        Generate evaluation report
        
        Args:
            results: Evaluation results
            output_dir: Output directory
            format: Report format (html, pdf, markdown)
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results.get("model", "unknown").replace("/", "_")
        
        if format == "html":
            report_path = output_path / f"report_{model_name}_{timestamp}.html"
            self._generate_html_report(results, report_path)
        elif format == "markdown":
            report_path = output_path / f"report_{model_name}_{timestamp}.md"
            self._generate_markdown_report(results, report_path)
        elif format == "pdf":
            # First generate HTML, then convert to PDF
            html_path = output_path / f"report_{model_name}_{timestamp}.html"
            self._generate_html_report(results, html_path)
            report_path = output_path / f"report_{model_name}_{timestamp}.pdf"
            self._convert_html_to_pdf(html_path, report_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(report_path)
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate HTML report with interactive visualizations"""
        # Create visualizations
        viz_data = self._create_visualizations(results)
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>MSB Evaluation Report - {{ model }}</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .section {
            background-color: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card {
            display: inline-block;
            background-color: #f8f9fa;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #34495e;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .chart-container {
            margin: 20px 0;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Multilingual Safety Benchmark Report</h1>
        <p>Model: <strong>{{ model }}</strong></p>
        <p>Dataset: <strong>{{ dataset }}</strong></p>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div>
            {% for metric, value in summary_metrics.items() %}
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(value) }}</div>
                <div class="metric-label">{{ metric|title }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="section">
        <h2>Language Performance</h2>
        <div id="language-chart" class="chart-container"></div>
    </div>
    
    <div class="section">
        <h2>Metric Breakdown</h2>
        <div id="metric-chart" class="chart-container"></div>
    </div>
    
    <div class="section">
        <h2>Detailed Results by Language</h2>
        <table>
            <thead>
                <tr>
                    <th>Language</th>
                    <th>Samples</th>
                    <th>Safety</th>
                    <th>Factuality</th>
                    <th>Cultural</th>
                    <th>Overall</th>
                </tr>
            </thead>
            <tbody>
                {% for lang, data in language_results.items() %}
                <tr>
                    <td>{{ lang }}</td>
                    <td>{{ data.num_samples }}</td>
                    <td>{{ "%.3f"|format(data.safety) }}</td>
                    <td>{{ "%.3f"|format(data.factuality) }}</td>
                    <td>{{ "%.3f"|format(data.cultural) }}</td>
                    <td>{{ "%.3f"|format(data.overall) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Error Analysis</h2>
        <p>Total errors: {{ total_errors }}</p>
        <p>Error rate: {{ "%.2f"|format(error_rate * 100) }}%</p>
    </div>
    
    <div class="footer">
        <p>Generated by MSB v1.0.0 | {{ timestamp }}</p>
    </div>
    
    <script>
        // Language performance chart
        Plotly.newPlot('language-chart', {{ language_chart_data|tojson }});
        
        // Metric breakdown chart
        Plotly.newPlot('metric-chart', {{ metric_chart_data|tojson }});
    </script>
</body>
</html>
"""
        
        # Prepare data for template
        template_data = self._prepare_template_data(results, viz_data)
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate Markdown report"""
        md_template = """# Multilingual Safety Benchmark Report

## Model Information
- **Model**: {{ model }}
- **Dataset**: {{ dataset }}
- **Generated**: {{ timestamp }}

## Executive Summary

| Metric | Score |
|--------|-------|
{% for metric, value in summary_metrics.items() -%}
| {{ metric|title }} | {{ "%.3f"|format(value) }} |
{% endfor %}

## Language Performance

| Language | Samples | Safety | Factuality | Cultural | Overall |
|----------|---------|--------|------------|----------|---------|
{% for lang, data in language_results.items() -%}
| {{ lang }} | {{ data.num_samples }} | {{ "%.3f"|format(data.safety) }} | {{ "%.3f"|format(data.factuality) }} | {{ "%.3f"|format(data.cultural) }} | {{ "%.3f"|format(data.overall) }} |
{% endfor %}

## Key Findings

1. **Best Performing Language**: {{ best_language }} ({{ "%.3f"|format(best_score) }})
2. **Worst Performing Language**: {{ worst_language }} ({{ "%.3f"|format(worst_score) }})
3. **Most Consistent Metric**: {{ most_consistent_metric }}
4. **Highest Variance Metric**: {{ highest_variance_metric }}

## Error Analysis
- Total errors: {{ total_errors }}
- Error rate: {{ "%.2f"|format(error_rate * 100) }}%

## Recommendations

Based on the evaluation results:
1. The model performs best on {{ best_language }} content
2. {{ worst_language }} requires improvement, particularly in {{ worst_metric }}
3. Overall {{ best_metric }} scores are strong across languages

---
*Generated by MSB v1.0.0 on {{ timestamp }}*
"""
        
        # Prepare data
        template_data = self._prepare_template_data(results, {})
        
        # Add analysis
        template_data.update(self._analyze_results(results))
        
        # Render template
        template = Template(md_template)
        md_content = template.render(**template_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _create_visualizations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive visualizations"""
        viz_data = {}
        
        # Prepare data for visualization
        languages = []
        safety_scores = []
        factuality_scores = []
        cultural_scores = []
        
        for lang, lang_data in results.get("languages", {}).items():
            if isinstance(lang_data, dict) and "num_samples" in lang_data:
                languages.append(lang)
                
                # Extract summary scores
                safety_scores.append(
                    lang_data.get("safety_summary", {}).get("mean", 0)
                )
                factuality_scores.append(
                    lang_data.get("factuality_summary", {}).get("mean", 0)
                )
                cultural_scores.append(
                    lang_data.get("cultural_summary", {}).get("mean", 0)
                )
        
        # Language performance radar chart
        if languages:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(name='Safety', x=languages, y=safety_scores))
            fig.add_trace(go.Bar(name='Factuality', x=languages, y=factuality_scores))
            fig.add_trace(go.Bar(name='Cultural', x=languages, y=cultural_scores))
            
            fig.update_layout(
                title="Performance by Language",
                xaxis_title="Language",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            
            viz_data["language_chart"] = fig.to_dict()
        
        # Metric distribution violin plot
        if results.get("aggregate", {}).get("metrics"):
            metric_data = []
            for metric, scores in results["aggregate"]["metrics"].items():
                if isinstance(scores, list):
                    for score in scores:
                        if isinstance(score, dict) and "score" in score:
                            metric_data.append({
                                "metric": metric,
                                "score": score["score"]
                            })
            
            if metric_data:
                df = pd.DataFrame(metric_data)
                fig = px.violin(
                    df, y="score", x="metric", box=True,
                    title="Score Distribution by Metric"
                )
                fig.update_layout(height=400)
                viz_data["metric_chart"] = fig.to_dict()
        
        return viz_data
    
    def _prepare_template_data(
        self,
        results: Dict[str, Any],
        viz_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for template rendering"""
        # Extract summary metrics
        summary_metrics = {}
        if "aggregate" in results and "metrics" in results["aggregate"]:
            for metric, stats in results["aggregate"]["metrics"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    summary_metrics[metric] = stats["mean"]
        
        # Extract language results
        language_results = {}
        for lang, lang_data in results.get("languages", {}).items():
            if isinstance(lang_data, dict) and "num_samples" in lang_data:
                language_results[lang] = {
                    "num_samples": lang_data["num_samples"],
                    "safety": lang_data.get("safety_summary", {}).get("mean", 0),
                    "factuality": lang_data.get("factuality_summary", {}).get("mean", 0),
                    "cultural": lang_data.get("cultural_summary", {}).get("mean", 0),
                    "overall": 0  # Calculate overall
                }
                
                # Calculate overall score
                scores = [
                    language_results[lang]["safety"],
                    language_results[lang]["factuality"],
                    language_results[lang]["cultural"]
                ]
                language_results[lang]["overall"] = sum(scores) / len(scores)
        
        # Calculate error statistics
        total_errors = results.get("aggregate", {}).get("total_errors", 0)
        total_samples = results.get("metadata", {}).get("total_samples", 1)
        error_rate = total_errors / total_samples if total_samples > 0 else 0
        
        return {
            "model": results.get("model", "Unknown"),
            "dataset": results.get("dataset", "Unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_metrics": summary_metrics,
            "language_results": language_results,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "language_chart_data": viz_data.get("language_chart", {}),
            "metric_chart_data": viz_data.get("metric_chart", {})
        }
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results for insights"""
        analysis = {}
        
        # Find best/worst languages
        language_scores = {}
        for lang, data in results.get("languages", {}).items():
            if isinstance(data, dict) and "num_samples" in data:
                scores = []
                for metric in ["safety", "factuality", "cultural"]:
                    summary_key = f"{metric}_summary"
                    if summary_key in data:
                        scores.append(data[summary_key].get("mean", 0))
                
                if scores:
                    language_scores[lang] = sum(scores) / len(scores)
        
        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            worst_lang = min(language_scores, key=language_scores.get)
            
            analysis["best_language"] = best_lang
            analysis["best_score"] = language_scores[best_lang]
            analysis["worst_language"] = worst_lang
            analysis["worst_score"] = language_scores[worst_lang]
        
        # Find most consistent/variable metrics
        metric_variances = {}
        if "aggregate" in results and "metrics" in results["aggregate"]:
            for metric, stats in results["aggregate"]["metrics"].items():
                if isinstance(stats, dict) and "std" in stats:
                    metric_variances[metric] = stats["std"]
        
        if metric_variances:
            analysis["most_consistent_metric"] = min(metric_variances, key=metric_variances.get)
            analysis["highest_variance_metric"] = max(metric_variances, key=metric_variances.get)
        
        # Find best/worst metrics
        metric_means = {}
        if "aggregate" in results and "metrics" in results["aggregate"]:
            for metric, stats in results["aggregate"]["metrics"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    metric_means[metric] = stats["mean"]
        
        if metric_means:
            analysis["best_metric"] = max(metric_means, key=metric_means.get)
            analysis["worst_metric"] = min(metric_means, key=metric_means.get)
        
        return analysis
    
    def _convert_html_to_pdf(self, html_path: Path, pdf_path: Path) -> None:
        """Convert HTML report to PDF (requires wkhtmltopdf or similar)"""
        # This is a placeholder - actual implementation would use
        # a tool like wkhtmltopdf, weasyprint, or selenium
        import shutil
        
        # For now, just copy the HTML file with a note
        shutil.copy(html_path, pdf_path.with_suffix('.html'))
        
        # Write a note about PDF conversion
        with open(pdf_path, 'w') as f:
            f.write("PDF conversion requires additional tools like wkhtmltopdf.\n")
            f.write(f"Please see the HTML report at: {html_path}\n")