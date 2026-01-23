"""
Simple Confusion Matrix for Sentiment Analysis Evaluation

Usage:
    python scripts/simple_confusion_matrix.py --input path/to/data.xlsx
    python scripts/simple_confusion_matrix.py --input data.xlsx --output results/

    # With MLflow logging (per 99Technologies AI Standards)
    python scripts/simple_confusion_matrix.py --input data.xlsx --mlflow
"""

import argparse
import os
import subprocess
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

# MLflow tracking (99Technologies standard)
MLFLOW_TRACKING_URI = "http://mlflow.99tech.internal:5000"
MLFLOW_EXPERIMENT_NAME = "s2a-sentiment-analysis"

# Sentiment labels in order (negative to positive)
SENTIMENT_LABELS = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]


def clean_sentiment(value):
    """Clean and normalize sentiment labels"""
    if pd.isna(value):
        return None

    # Convert to string and strip whitespace
    cleaned = str(value).strip().title()

    # Fix common typos
    if cleaned == "Postive":
        cleaned = "Positive"

    return cleaned


def load_and_clean_data(excel_path):
    """Load Excel file and clean sentiment columns"""
    df = pd.read_excel(excel_path)

    print(f"Loaded {len(df)} rows from {excel_path}")
    print(f"Columns: {list(df.columns)}")

    # Clean sentiment columns
    df['Predicted'] = df['Sentiment'].apply(clean_sentiment)
    df['Actual'] = df['Ground Truth'].apply(clean_sentiment)

    # Show cleaning results
    print(f"\nAfter cleaning:")
    print(f"  Predicted unique: {df['Predicted'].unique()}")
    print(f"  Actual unique: {df['Actual'].unique()}")

    # Remove rows with invalid labels
    valid_mask = df['Predicted'].isin(SENTIMENT_LABELS) & df['Actual'].isin(SENTIMENT_LABELS)
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        print(f"\nWarning: Removing {invalid_count} rows with invalid labels")
        invalid_rows = df[~valid_mask][['Predicted', 'Actual']]
        print(invalid_rows)

    df = df[valid_mask]
    print(f"\nUsing {len(df)} valid rows for evaluation")

    return df


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Plot and save confusion matrix"""
    # Filter labels to only those present in data
    present_labels = [l for l in labels if l in y_true.values or l in y_pred.values]

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=present_labels,
        yticklabels=present_labels,
        cbar_kws={'label': 'Count'}
    )

    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.ylabel('Actual Sentiment (Ground Truth)', fontsize=12)
    plt.title('Sentiment Analysis Confusion Matrix\n(LLM Predictions vs Human Annotations)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {output_path}")


def calculate_adjacent_accuracy(y_true, y_pred, labels):
    """Calculate accuracy allowing off-by-one predictions"""
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    adjacent_correct = 0
    total = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual in label_to_idx and predicted in label_to_idx:
            distance = abs(label_to_idx[actual] - label_to_idx[predicted])
            if distance <= 1:
                adjacent_correct += 1
            total += 1

    return adjacent_correct / total if total > 0 else 0


def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def log_to_mlflow(metrics, artifacts, params, run_name):
    """Log experiment to MLflow (per 99Technologies AI Standards)"""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=run_name):
            # Log parameters (Section 4.3 - What MUST Be Logged)
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log artifacts (confusion matrix, report)
            for artifact_path in artifacts:
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)

            print(f"\n[MLflow] Logged to: {MLFLOW_TRACKING_URI}")
            print(f"[MLflow] Experiment: {MLFLOW_EXPERIMENT_NAME}")
            print(f"[MLflow] Run: {run_name}")

    except ImportError:
        print("\n[MLflow] Warning: mlflow not installed. Run: pip install mlflow")
    except Exception as e:
        print(f"\n[MLflow] Warning: Could not log to MLflow: {e}")
        print("[MLflow] Results saved locally. Upload manually when server is available.")


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix for sentiment analysis")
    parser.add_argument("--input", "-i", required=True, help="Path to Excel file")
    parser.add_argument("--output", "-o", default="results", help="Output directory (default: results/)")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow server")
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: auto-generated)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and clean data
    df = load_and_clean_data(args.input)

    if len(df) == 0:
        print("Error: No valid data to evaluate")
        return

    y_true = df['Actual']
    y_pred = df['Predicted']

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    adjacent_acc = calculate_adjacent_accuracy(y_true, y_pred, SENTIMENT_LABELS)

    # Generate classification report
    present_labels = [l for l in SENTIMENT_LABELS if l in y_true.values or l in y_pred.values]
    report = classification_report(y_true, y_pred, labels=present_labels, zero_division=0)

    # Calculate per-class metrics for MLflow
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Exact accuracy: {accuracy:.1%}")
    print(f"Adjacent accuracy (within 1 level): {adjacent_acc:.1%}")
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Value counts comparison
    print("=" * 60)
    print("DISTRIBUTION COMPARISON")
    print("=" * 60)
    print("\nGround Truth (Actual):")
    print(y_true.value_counts().sort_index())
    print("\nLLM Predictions:")
    print(y_pred.value_counts().sort_index())

    # Plot confusion matrix
    plot_path = os.path.join(args.output, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(y_true, y_pred, SENTIMENT_LABELS, plot_path)

    # Save detailed report
    report_path = os.path.join(args.output, f"evaluation_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write("SENTIMENT ANALYSIS EVALUATION REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Input file: {args.input}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Exact accuracy: {accuracy:.1%}\n")
        f.write(f"Adjacent accuracy: {adjacent_acc:.1%}\n\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write(report)
        f.write("\n\nGROUND TRUTH DISTRIBUTION:\n")
        f.write(y_true.value_counts().sort_index().to_string())
        f.write("\n\nPREDICTION DISTRIBUTION:\n")
        f.write(y_pred.value_counts().sort_index().to_string())

    print(f"Report saved: {report_path}")

    # Log to MLflow if requested (per 99Technologies AI Standards Section 4)
    if args.mlflow:
        run_name = args.run_name or f"sentiment-eval-{timestamp}"

        # Parameters to log (Section 4.3)
        params = {
            "dataset_path": os.path.basename(args.input),
            "dataset_size": len(df),
            "num_classes": len(present_labels),
            "sentiment_labels": ",".join(present_labels),
            "model": "mistral-nemo",  # From analysis_client.py
            "git_commit": get_git_commit(),
        }

        # Metrics to log
        metrics = {
            "accuracy": accuracy,
            "adjacent_accuracy": adjacent_acc,
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
        }

        # Add per-class metrics
        for i, label in enumerate(present_labels):
            safe_label = label.lower().replace(" ", "_")
            metrics[f"precision_{safe_label}"] = float(precision[i])
            metrics[f"recall_{safe_label}"] = float(recall[i])
            metrics[f"f1_{safe_label}"] = float(f1[i])

        # Artifacts to log
        artifacts = [plot_path, report_path]

        log_to_mlflow(metrics, artifacts, params, run_name)


if __name__ == "__main__":
    main()
