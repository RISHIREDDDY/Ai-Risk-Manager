"""
evaluate.py — Quantitative Evaluation Layer for AI Risk Manager.

Runs empirical benchmark evaluation over the held-out test dataset (is_test_set = 1)
using scikit-learn to measure precision, recall, F1, confusion matrix, and financial cost-benefit metrics.

Usage:
    python evaluate.py [--rescore] [--delay SECONDS]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure src directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent import analyze_dispute
from database import get_connection

# Standard Stripe India / Razorpay dispute fee for lost chargebacks
DISPUTE_FEE_INR = 1000.0


def fetch_test_records() -> list[dict]:
    """Fetch all held-out test records (is_test_set = 1) from the database."""
    conn = get_connection()
    try:
        query = """
            SELECT
                d.dispute_id,
                d.order_id,
                d.reason_code,
                d.disputed_amount,
                d.ground_truth_label,
                d.status,
                d.created_at,
                t.customer_id,
                t.amount AS txn_amount,
                t.currency,
                t.payment_method,
                t.avs_match,
                t.cvv_match,
                t.upi_vpa_match,
                s.carrier_status,
                s.gps_match,
                s.signature_obtained
            FROM disputes d
            LEFT JOIN transactions t ON d.order_id = t.order_id
            LEFT JOIN shipping_logs s ON d.order_id = s.order_id
            WHERE d.is_test_set = 1
            ORDER BY d.created_at ASC
        """
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_latest_audit_decision(dispute_id: str) -> Optional[dict]:
    """Check if an audit log entry already exists for this dispute."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT decision, confidence, tools_called, reasoning, evidence_letter, created_at
            FROM audit_logs
            WHERE dispute_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (dispute_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def run_evaluation(
    rescore: bool = False,
    delay: float = 4.0,
    verbose: bool = True,
) -> dict:
    """
    Run empirical benchmark evaluation on the held-out test set.

    Args:
        rescore: If True, forces the agent to re-score all test records via Gemini API.
                 If False, reuses existing audit_logs where available.
        delay: Delay in seconds between API calls to prevent rate-limit exhaustion.
        verbose: If True, prints real-time progress to stdout.

    Returns:
        Dictionary containing all evaluation metrics, confusion matrix, financial analysis, and case details.
    """
    records = fetch_test_records()

    if not records:
        raise ValueError("No test set records found (is_test_set = 1). Run generate_data.py first.")

    if verbose:
        print("\n" + "=" * 70)
        print("   AI RISK MANAGER -- QUANTITATIVE EVALUATION BENCHMARK")
        print("=" * 70)
        print(f"  Total Held-Out Test Records : {len(records)}")
        print(f"  Mode                         : {'Full Re-Score (API)' if rescore else 'Use Cached/Score Missing'}")
        print(f"  Dispute Fee per Loss        : INR {DISPUTE_FEE_INR:,.2f}")
        print("-" * 70)

    y_true: List[int] = []  # 1 = valid_defense (CONTEST), 0 = lost_cause (ACCEPT)
    y_pred: List[int] = []  # 1 = CONTEST_DISPUTE, 0 = ACCEPT_DISPUTE
    case_results: List[dict] = []

    for i, record in enumerate(records):
        dispute_id = record["dispute_id"]
        order_id = record["order_id"]
        amount = float(record["disputed_amount"])
        ground_truth = record["ground_truth_label"]
        expected_decision = "CONTEST_DISPUTE" if ground_truth == "valid_defense" else "ACCEPT_DISPUTE"
        true_binary = 1 if ground_truth == "valid_defense" else 0

        # Retrieve prediction
        audit_log = None if rescore else get_latest_audit_decision(dispute_id)

        if audit_log is not None:
            predicted_decision = audit_log["decision"]
            confidence = float(audit_log["confidence"])
            reasoning = audit_log["reasoning"]
            source = "cached_audit_log"
        else:
            if i > 0 and delay > 0:
                time.sleep(delay)
            if verbose:
                print(f"  [{i+1}/{len(records)}] Scoring {dispute_id} via Agent reasoning...")
            agent_result = analyze_dispute(dispute_id)
            predicted_decision = agent_result.get("decision", "ACCEPT_DISPUTE")
            confidence = float(agent_result.get("confidence", 0.0))
            reasoning = agent_result.get("reasoning", "")
            source = "live_gemini_agent"

        # Map PARTIAL_CONTEST to CONTEST for binary classification
        # (ground-truth labels are binary: valid_defense vs lost_cause)
        pred_binary = 1 if predicted_decision in ("CONTEST_DISPUTE", "PARTIAL_CONTEST") else 0
        y_true.append(true_binary)
        y_pred.append(pred_binary)

        is_correct = (true_binary == pred_binary)

        # Classify metric type and financial cost
        if true_binary == 1 and pred_binary == 1:
            classification_type = "TP"
            # Successfully won winnable dispute
            cost_benefit = amount
            note = f"Defended INR {amount:,.2f} revenue"
        elif true_binary == 0 and pred_binary == 0:
            classification_type = "TN"
            # Accepted non-winnable claim, saving dispute fee
            cost_benefit = DISPUTE_FEE_INR
            note = f"Saved INR {DISPUTE_FEE_INR:,.2f} dispute fee by not fighting"
        elif true_binary == 0 and pred_binary == 1:
            classification_type = "FP"
            # Wrongly fought lost cause, lost dispute amount + INR 1,000 fee
            cost_benefit = -(amount + DISPUTE_FEE_INR)
            note = f"Lost INR {amount + DISPUTE_FEE_INR:,.2f} (order + dispute fee)"
        else:  # true_binary == 1 and pred_binary == 0
            classification_type = "FN"
            # Wrongly conceded valid dispute, lost dispute amount
            cost_benefit = -amount
            note = f"Forfeited INR {amount:,.2f} winnable revenue"

        case_results.append({
            "dispute_id": dispute_id,
            "order_id": order_id,
            "disputed_amount": amount,
            "payment_method": record.get("payment_method", "N/A"),
            "carrier_status": record.get("carrier_status", "N/A"),
            "ground_truth_label": ground_truth,
            "expected_decision": expected_decision,
            "predicted_decision": predicted_decision,
            "confidence": confidence,
            "is_correct": is_correct,
            "classification_type": classification_type,
            "financial_impact_inr": cost_benefit,
            "financial_note": note,
            "reasoning": reasoning[:150] + "..." if len(reasoning) > 150 else reasoning,
            "source": source,
        })

        if verbose:
            status_icon = "[CORRECT]" if is_correct else "[MISMATCH]"
            print(
                f"  {status_icon} {dispute_id} | Expected: {expected_decision:<15} | "
                f"Pred: {predicted_decision:<15} | Type: {classification_type} | "
                f"Conf: {confidence:.2f}"
            )

    # ---------------------------------------------------------------------------
    # Scikit-Learn Metrics Computation
    # ---------------------------------------------------------------------------
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    target_names = ["lost_cause (ACCEPT)", "valid_defense (CONTEST)"]
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    # ---------------------------------------------------------------------------
    # Financial Cost-Benefit Analysis
    # ---------------------------------------------------------------------------
    total_disputed = sum(c["disputed_amount"] for c in case_results)
    tp_revenue_saved = sum(c["financial_impact_inr"] for c in case_results if c["classification_type"] == "TP")
    tn_fees_saved = sum(c["financial_impact_inr"] for c in case_results if c["classification_type"] == "TN")
    fp_cost = sum(abs(c["financial_impact_inr"]) for c in case_results if c["classification_type"] == "FP")
    fn_cost = sum(abs(c["financial_impact_inr"]) for c in case_results if c["classification_type"] == "FN")

    gross_value_created = tp_revenue_saved + tn_fees_saved
    total_cost_of_errors = fp_cost + fn_cost
    net_financial_benefit = gross_value_created - total_cost_of_errors

    # Baseline cost: If merchant accepted 100% of disputes without fighting
    baseline_naive_accept_cost = sum(c["disputed_amount"] for c in case_results)
    # Baseline cost: If merchant contested 100% of disputes blindly
    lost_cause_count = sum(1 for c in case_results if c["ground_truth_label"] == "lost_cause")
    baseline_naive_contest_loss = sum(
        c["disputed_amount"] + DISPUTE_FEE_INR for c in case_results if c["ground_truth_label"] == "lost_cause"
    )

    summary_metrics = {
        "dataset_size": len(case_results),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "raw_matrix": cm.tolist(),
        },
        "financial_impact": {
            "dispute_fee_per_case_inr": DISPUTE_FEE_INR,
            "total_disputed_amount_evaluated_inr": round(total_disputed, 2),
            "tp_revenue_saved_inr": round(tp_revenue_saved, 2),
            "tn_fees_saved_inr": round(tn_fees_saved, 2),
            "false_positive_cost_inr": round(fp_cost, 2),
            "false_negative_cost_inr": round(fn_cost, 2),
            "gross_value_created_inr": round(gross_value_created, 2),
            "total_cost_of_errors_inr": round(total_cost_of_errors, 2),
            "net_financial_benefit_inr": round(net_financial_benefit, 2),
        },
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "cases": case_results,
    }

    if verbose:
        print_evaluation_report(summary_metrics)

    return summary_metrics


def print_evaluation_report(metrics: dict) -> None:
    """Pretty print the evaluation metrics to stdout."""
    cm = metrics["confusion_matrix"]
    fin = metrics["financial_impact"]

    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS REPORT")
    print("=" * 70)
    print(f"  Accuracy           : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision (Contest): {metrics['precision'] * 100:.2f}%")
    print(f"  Recall (Contest)   : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score           : {metrics['f1_score'] * 100:.2f}%")
    print("-" * 70)

    print("  CONFUSION MATRIX:")
    print("                      Actual Lost Cause     Actual Valid Defense")
    print(f"    Pred ACCEPT     : TN = {cm['true_negative']:<18} FN = {cm['false_negative']:<18}")
    print(f"    Pred CONTEST    : FP = {cm['false_positive']:<18} TP = {cm['true_positive']:<18}")
    print("-" * 70)

    print("  FINANCIAL IMPACT (INR):")
    print(f"    Total Evaluated Volume      : INR {fin['total_disputed_amount_evaluated_inr']:>12,.2f}")
    print(f"    Revenue Saved (TP)          : INR {fin['tp_revenue_saved_inr']:>12,.2f}")
    print(f"    Dispute Fees Saved (TN)     : INR {fin['tn_fees_saved_inr']:>12,.2f}")
    print(f"    False Positive Loss (FP)    : INR {fin['false_positive_cost_inr']:>12,.2f}")
    print(f"    False Negative Loss (FN)    : INR {fin['false_negative_cost_inr']:>12,.2f}")
    print(f"    ---------------------------------------------------------")
    print(f"    NET FINANCIAL BENEFIT       : INR {fin['net_financial_benefit_inr']:>12,.2f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Risk Manager on held-out test set.")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Force re-scoring of all test disputes via Gemini API (uses rate-limiting delay).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Delay in seconds between Gemini API calls during re-scoring (default: 4.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics as JSON file.",
    )

    args = parser.parse_args()

    results = run_evaluation(rescore=args.rescore, delay=args.delay, verbose=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[evaluate] Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
