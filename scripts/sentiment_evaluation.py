"""
Sentiment Analysis Evaluation Script

This script evaluates the sentiment analysis model by comparing predictions
against human-annotated ground truth labels.

Usage:
    # 1. First, create your annotated dataset (see sample_annotations.json)
    # 2. Run evaluation:
    python scripts/sentiment_evaluation.py --input data/annotations.json --output results/

    # Or with Triton URL:
    python scripts/sentiment_evaluation.py --input data/annotations.json --triton-url localhost:2001

Input Format (JSON):
    [
        {
            "id": "sample_001",
            "transcription": "Thank you so much for your help today...",
            "ground_truth": "Very Positive"
        },
        ...
    ]

Output:
    - Confusion matrix (PNG image)
    - Classification report (text)
    - Detailed results (JSON)
    - Misclassification analysis (for prompt tuning)
"""

import json
import asyncio
import argparse
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Optional imports for visualization
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not installed. Install with: pip install numpy")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Confusion matrix plot will be skipped.")
    print("Install with: pip install matplotlib seaborn")

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligent_pipeline.analysis_client import AsyncAnalysis


# Constants
SENTIMENT_LABELS = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
SENTIMENT_ORDER = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}


@dataclass
class EvaluationSample:
    """Single evaluation sample with prediction results"""
    id: str
    transcription: str
    ground_truth: str
    predicted: Optional[str] = None
    confidence: Optional[float] = None
    key_indicators: Optional[List[str]] = None
    is_correct: bool = False
    is_adjacent: bool = False  # Within 1 level
    error_distance: int = 0  # How many levels off
    raw_response: Optional[Dict] = None


@dataclass
class EvaluationMetrics:
    """Overall evaluation metrics"""
    total_samples: int
    exact_accuracy: float
    adjacent_accuracy: float
    mean_confidence: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: str


def calculate_error_distance(ground_truth: str, predicted: str) -> int:
    """Calculate how many sentiment levels apart the prediction is"""
    if ground_truth not in SENTIMENT_ORDER or predicted not in SENTIMENT_ORDER:
        return -1  # Invalid label
    return abs(SENTIMENT_ORDER[ground_truth] - SENTIMENT_ORDER[predicted])


def is_adjacent(ground_truth: str, predicted: str) -> bool:
    """Check if prediction is within 1 level of ground truth"""
    distance = calculate_error_distance(ground_truth, predicted)
    return distance <= 1


async def evaluate_single_sample(
    client: AsyncAnalysis,
    sample: Dict,
    sample_idx: int,
    total: int
) -> EvaluationSample:
    """Evaluate a single sample and return results"""

    eval_sample = EvaluationSample(
        id=sample.get("id", f"sample_{sample_idx}"),
        transcription=sample["transcription"],
        ground_truth=sample["ground_truth"]
    )

    try:
        print(f"  [{sample_idx + 1}/{total}] Processing {eval_sample.id}...", end=" ")

        # Call the analysis client
        response = await client.analyze_call_async(
            eval_sample.transcription,
            request_id=eval_sample.id
        )

        # Parse response
        result = json.loads(response)

        if result.get("success"):
            analysis = result.get("analysis", {})
            ai_analysis = analysis.get("ai_analysis", {})
            sentiment = ai_analysis.get("sentiment", {})

            eval_sample.predicted = sentiment.get("category")
            eval_sample.confidence = sentiment.get("confidence")
            eval_sample.key_indicators = sentiment.get("key_indicators", [])
            eval_sample.raw_response = analysis

            # Calculate metrics
            if eval_sample.predicted:
                eval_sample.is_correct = eval_sample.predicted == eval_sample.ground_truth
                eval_sample.is_adjacent = is_adjacent(eval_sample.ground_truth, eval_sample.predicted)
                eval_sample.error_distance = calculate_error_distance(
                    eval_sample.ground_truth, eval_sample.predicted
                )

        status = "CORRECT" if eval_sample.is_correct else f"WRONG (predicted: {eval_sample.predicted})"
        print(status)

    except Exception as e:
        print(f"ERROR: {e}")
        eval_sample.raw_response = {"error": str(e)}

    return eval_sample


def build_confusion_matrix(results: List[EvaluationSample]) -> List[List[int]]:
    """Build confusion matrix from results"""
    n_classes = len(SENTIMENT_LABELS)
    matrix = [[0] * n_classes for _ in range(n_classes)]

    for sample in results:
        if sample.ground_truth in SENTIMENT_ORDER and sample.predicted in SENTIMENT_ORDER:
            true_idx = SENTIMENT_ORDER[sample.ground_truth]
            pred_idx = SENTIMENT_ORDER[sample.predicted]
            matrix[true_idx][pred_idx] += 1

    return matrix


def generate_classification_report(results: List[EvaluationSample]) -> str:
    """Generate a text classification report"""
    # Count per-class metrics
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})

    for sample in results:
        gt = sample.ground_truth
        pred = sample.predicted

        if gt in SENTIMENT_ORDER:
            class_stats[gt]["support"] += 1

            if pred == gt:
                class_stats[gt]["tp"] += 1
            else:
                class_stats[gt]["fn"] += 1
                if pred in SENTIMENT_ORDER:
                    class_stats[pred]["fp"] += 1

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("CLASSIFICATION REPORT")
    lines.append("=" * 70)
    lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 70)

    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in SENTIMENT_LABELS:
        stats = class_stats[label]
        support = stats["support"]
        total_support += support

        # Calculate metrics
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

        lines.append(f"{label:<20} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}")

    lines.append("-" * 70)

    # Weighted averages
    if total_support > 0:
        lines.append(f"{'Weighted Avg':<20} {weighted_precision/total_support:>10.2f} "
                    f"{weighted_recall/total_support:>10.2f} {weighted_f1/total_support:>10.2f} "
                    f"{total_support:>10}")

    lines.append("=" * 70)

    return "\n".join(lines)


def plot_confusion_matrix(matrix: List[List[int]], output_path: str):
    """Plot and save confusion matrix"""
    if not PLOTTING_AVAILABLE or not NUMPY_AVAILABLE:
        print("Skipping confusion matrix plot (matplotlib/seaborn/numpy not installed)")
        return

    plt.figure(figsize=(10, 8))

    # Convert to numpy array
    cm = np.array(matrix)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
        cbar_kws={'label': 'Count'}
    )

    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.title('Sentiment Analysis Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def generate_misclassification_analysis(results: List[EvaluationSample]) -> str:
    """Generate analysis of misclassifications for prompt tuning"""
    lines = []
    lines.append("=" * 70)
    lines.append("MISCLASSIFICATION ANALYSIS (for prompt tuning)")
    lines.append("=" * 70)

    # Group misclassifications by error type
    error_patterns = defaultdict(list)

    for sample in results:
        if not sample.is_correct and sample.predicted:
            error_key = f"{sample.ground_truth} -> {sample.predicted}"
            error_patterns[error_key].append(sample)

    if not error_patterns:
        lines.append("No misclassifications found!")
        return "\n".join(lines)

    # Sort by frequency
    sorted_errors = sorted(error_patterns.items(), key=lambda x: -len(x[1]))

    for error_type, samples in sorted_errors:
        lines.append(f"\n{'-' * 50}")
        lines.append(f"ERROR PATTERN: {error_type} ({len(samples)} occurrences)")
        lines.append(f"{'-' * 50}")

        for sample in samples[:3]:  # Show max 3 examples per pattern
            lines.append(f"\n  Sample ID: {sample.id}")
            lines.append(f"  Confidence: {sample.confidence}")
            lines.append(f"  Key Indicators: {sample.key_indicators}")
            lines.append(f"  Transcription (first 200 chars):")
            lines.append(f"    \"{sample.transcription[:200]}...\"")

    lines.append("\n" + "=" * 70)
    lines.append("RECOMMENDATIONS FOR PROMPT TUNING:")
    lines.append("=" * 70)

    # Generate recommendations based on error patterns
    for error_type, samples in sorted_errors[:3]:
        gt, pred = error_type.split(" -> ")
        distance = calculate_error_distance(gt, pred)

        if distance == 1:
            lines.append(f"\n- {error_type}: Add clearer distinction between {gt} and {pred}")
            lines.append(f"  Consider adding example phrases that differentiate these two levels")
        elif distance >= 2:
            lines.append(f"\n- {error_type}: MAJOR ERROR (off by {distance} levels)")
            lines.append(f"  Review the key_indicators the model identified - likely misinterpreting tone")

    return "\n".join(lines)


async def run_evaluation(
    annotations_path: str,
    triton_url: str = "localhost:2001",
    output_dir: str = "results"
) -> EvaluationMetrics:
    """Run full evaluation pipeline"""

    print("=" * 70)
    print("SENTIMENT ANALYSIS EVALUATION")
    print("=" * 70)
    print(f"Annotations file: {annotations_path}")
    print(f"Triton URL: {triton_url}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Load annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"\nLoaded {len(annotations)} annotated samples")

    # Validate annotations
    for i, sample in enumerate(annotations):
        if "transcription" not in sample:
            raise ValueError(f"Sample {i} missing 'transcription' field")
        if "ground_truth" not in sample:
            raise ValueError(f"Sample {i} missing 'ground_truth' field")
        if sample["ground_truth"] not in SENTIMENT_LABELS:
            raise ValueError(f"Sample {i} has invalid ground_truth: {sample['ground_truth']}. "
                           f"Must be one of: {SENTIMENT_LABELS}")

    # Initialize client
    client = AsyncAnalysis(url=triton_url)

    # Run evaluation
    print("\nRunning predictions...")
    results = []
    for idx, sample in enumerate(annotations):
        result = await evaluate_single_sample(client, sample, idx, len(annotations))
        results.append(result)

    # Calculate metrics
    correct = sum(1 for r in results if r.is_correct)
    adjacent = sum(1 for r in results if r.is_adjacent)
    confidences = [r.confidence for r in results if r.confidence is not None]

    exact_accuracy = correct / len(results) if results else 0
    adjacent_accuracy = adjacent / len(results) if results else 0
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Per-class accuracy
    per_class = {}
    for label in SENTIMENT_LABELS:
        class_samples = [r for r in results if r.ground_truth == label]
        if class_samples:
            per_class[label] = sum(1 for r in class_samples if r.is_correct) / len(class_samples)
        else:
            per_class[label] = 0.0

    # Build confusion matrix
    confusion_matrix = build_confusion_matrix(results)

    # Generate classification report
    classification_report = generate_classification_report(results)

    # Create metrics object
    metrics = EvaluationMetrics(
        total_samples=len(results),
        exact_accuracy=exact_accuracy,
        adjacent_accuracy=adjacent_accuracy,
        mean_confidence=mean_confidence,
        per_class_accuracy=per_class,
        confusion_matrix=confusion_matrix,
        classification_report=classification_report
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    detailed_results = {
        "timestamp": timestamp,
        "config": {
            "annotations_file": annotations_path,
            "triton_url": triton_url,
            "sentiment_labels": SENTIMENT_LABELS
        },
        "summary": {
            "total_samples": metrics.total_samples,
            "exact_accuracy": metrics.exact_accuracy,
            "adjacent_accuracy": metrics.adjacent_accuracy,
            "mean_confidence": metrics.mean_confidence,
            "per_class_accuracy": metrics.per_class_accuracy
        },
        "confusion_matrix": metrics.confusion_matrix,
        "samples": [asdict(r) for r in results]
    }

    results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")

    # Save classification report
    report_path = os.path.join(output_dir, f"classification_report_{timestamp}.txt")
    misclass_analysis = generate_misclassification_analysis(results)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(classification_report)
        f.write("\n\n")
        f.write(misclass_analysis)
    print(f"Classification report saved to: {report_path}")

    # Plot confusion matrix
    plot_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(confusion_matrix, plot_path)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Samples: {metrics.total_samples}")
    print(f"Exact Accuracy: {metrics.exact_accuracy:.1%} ({correct}/{len(results)})")
    print(f"Adjacent Accuracy: {metrics.adjacent_accuracy:.1%} ({adjacent}/{len(results)})")
    print(f"Mean Confidence: {metrics.mean_confidence:.2f}")
    print(f"\nPer-Class Accuracy:")
    for label, acc in metrics.per_class_accuracy.items():
        print(f"  {label}: {acc:.1%}")
    print("=" * 70)

    print("\n" + classification_report)
    print("\n" + misclass_analysis)

    return metrics


def create_sample_annotations_file(output_path: str):
    """Create a sample annotations file as a template"""
    sample_data = [
        {
            "id": "sample_001",
            "transcription": "Thank you so much for resolving my issue! You've been incredibly helpful and I really appreciate your patience. I'll definitely recommend your service to my friends.",
            "ground_truth": "Very Positive"
        },
        {
            "id": "sample_002",
            "transcription": "That works for me, thanks for the help. The refund should come through in a few days then.",
            "ground_truth": "Positive"
        },
        {
            "id": "sample_003",
            "transcription": "I need to check my account balance and see when my next payment is due.",
            "ground_truth": "Neutral"
        },
        {
            "id": "sample_004",
            "transcription": "I've been waiting for 20 minutes already. This is frustrating. Can you please just transfer me to someone who can help?",
            "ground_truth": "Negative"
        },
        {
            "id": "sample_005",
            "transcription": "This is absolutely unacceptable! I've been a customer for 10 years and this is how you treat me? I want to cancel everything and speak to your manager immediately!",
            "ground_truth": "Very Negative"
        }
    ]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Sample annotations file created: {output_path}")
    print("Edit this file with your own transcriptions and ground truth labels.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sentiment analysis model against ground truth annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample annotations template:
  python scripts/sentiment_evaluation.py --create-sample data/sample_annotations.json

  # Run evaluation:
  python scripts/sentiment_evaluation.py --input data/annotations.json --output results/

  # With custom Triton URL:
  python scripts/sentiment_evaluation.py --input data/annotations.json --triton-url host.docker.internal:2001
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to JSON file with annotated samples"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )

    parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:2001",
        help="Triton server URL (default: localhost:2001)"
    )

    parser.add_argument(
        "--create-sample",
        type=str,
        metavar="PATH",
        help="Create a sample annotations file at the specified path"
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_annotations_file(args.create_sample)
        return

    if not args.input:
        parser.error("--input is required (or use --create-sample to create a template)")

    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    # Run evaluation
    asyncio.run(run_evaluation(
        annotations_path=args.input,
        triton_url=args.triton_url,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()
