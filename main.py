"""
main.py — FastAPI Service Layer for AI Risk Manager (Chargeback Evidence Responder).

Exposes RESTful endpoints for:
  1. GET  /api/disputes/pending    — Active dispute queue for Streamlit inbox table
  2. POST /api/disputes/score      — Triggers Gemini Agent reasoning, logs to audit_logs, returns decision
  3. GET  /api/audit-logs          — Searchable decision history & audit trail
  4. GET  /api/evaluation/metrics  — Empirical evaluation & benchmark metrics over held-out test set
  5. GET  /health                  — System health and database connectivity check
"""

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, cast

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import os
import sys

# Ensure src directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent import analyze_dispute
from database import get_connection

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Risk Manager — Chargeback Evidence Responder API",
    description=(
        "Service layer exposing endpoints for chargeback dispute triage, "
        "Gemini-powered agentic evidence reasoning, audit trail retrieval, "
        "and empirical benchmark evaluation."
    ),
    version="1.0.0",
)

# Enable CORS for Streamlit and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    dispute_id: str = Field(..., description="Unique dispute ID to score (e.g. 'DSP-XXXXXXXX')")


class ScoreResponse(BaseModel):
    dispute_id: str
    decision: str
    confidence: float
    reasoning: str
    evidence_letter: str
    tools_used: List[str] = []
    customer_notification: str = ""
    contested_amount: Optional[float] = None
    conceded_amount: Optional[float] = None
    sarcasm_detected: bool = False
    fairness_flags: List[str] = []


class DisputeItem(BaseModel):
    dispute_id: str
    order_id: str
    customer_id: Optional[str] = None
    reason_code: str
    disputed_amount: float
    currency: str = "INR"
    payment_method: Optional[str] = None
    ground_truth_label: str
    is_test_set: int
    status: str
    created_at: str
    avs_match: Optional[int] = None
    cvv_match: Optional[int] = None
    upi_vpa_match: Optional[int] = None
    ip_distance_km: Optional[float] = None


class DisputeListResponse(BaseModel):
    total: int
    disputes: List[DisputeItem]


class AuditLogItem(BaseModel):
    id: int
    dispute_id: str
    decision: str
    confidence: float
    tools_called: List[str]
    reasoning: str
    evidence_letter: str
    customer_notification: str = ""
    contested_amount: Optional[float] = None
    conceded_amount: Optional[float] = None
    fairness_flags: List[str] = []
    created_at: str


class AuditLogListResponse(BaseModel):
    total: int
    logs: List[AuditLogItem]


class ConfusionMatrixDetails(BaseModel):
    true_negative: int = Field(..., description="Correctly accepted lost causes (TN)")
    false_positive: int = Field(..., description="Wrongly contested lost causes (FP)")
    false_negative: int = Field(..., description="Wrongly accepted valid defenses (FN)")
    true_positive: int = Field(..., description="Correctly contested valid defenses (TP)")
    raw_matrix: List[List[int]]


class FinancialImpact(BaseModel):
    dispute_fee_per_case: float
    total_disputed_amount_evaluated: float
    false_positive_cost: float = Field(
        ...,
        description="Loss from wrongly fighting lost cause (amount + dispute fee)"
    )
    false_negative_cost: float = Field(
        ...,
        description="Loss from conceding valid defense (forfeited revenue)"
    )
    true_positive_saved: float = Field(
        ...,
        description="Winnable revenue successfully defended"
    )
    true_negative_saved: float = Field(
        ...,
        description="Dispute fees avoided by accepting non-winnable claims early"
    )
    net_financial_benefit: float = Field(
        ...,
        description="Total money saved minus losses incurred from errors"
    )


class EvaluationCaseDetail(BaseModel):
    dispute_id: str
    order_id: str
    disputed_amount: float
    ground_truth_label: str
    ground_truth_decision: str
    predicted_decision: str
    confidence: float
    is_correct: bool
    classification_type: str  # TP, TN, FP, FN
    financial_impact_inr: float


class EvaluationMetricsResponse(BaseModel):
    total_test_records: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrixDetails
    financial_impact: FinancialImpact
    classification_report: Dict[str, Any]
    test_cases: List[EvaluationCaseDetail]


# ---------------------------------------------------------------------------
# Health & Root Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root() -> Dict[str, Any]:
    """Root status and API metadata endpoint."""
    return {
        "service": "AI Risk Manager API",
        "status": "operational",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            "/api/disputes/pending",
            "/api/disputes/score",
            "/api/audit-logs",
            "/api/evaluation/metrics",
            "/health",
        ],
    }


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Check database connection and system readiness."""
    try:
        conn = get_connection()
        total_disputes = conn.execute("SELECT COUNT(*) FROM disputes").fetchone()[0]
        total_logs = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "total_disputes": total_disputes,
            "total_audit_logs": total_logs,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connectivity issue: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Endpoint 1: GET /api/disputes/pending
# ---------------------------------------------------------------------------
@app.get(
    "/api/disputes/pending",
    response_model=DisputeListResponse,
    tags=["Disputes"],
    summary="Get active dispute queue for Streamlit inbox table",
)
def get_pending_disputes(
    status_filter: Optional[str] = Query(
        "pending",
        alias="status",
        description="Filter by dispute status ('pending', 'scored', or 'all')",
    ),
    is_test_set: Optional[int] = Query(
        None,
        description="Filter by test set flag (0 for train/inbox, 1 for held-out test, None for all)",
    ),
    limit: int = Query(100, ge=1, le=500, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> DisputeListResponse:
    """
    Retrieve list of disputes with enriched transaction metadata for the
    inbox table view.
    """
    conn = get_connection()
    try:
        where_clauses = []
        params: List[Any] = []

        if status_filter and status_filter.lower() != "all":
            where_clauses.append("d.status = ?")
            params.append(status_filter.lower())

        if is_test_set is not None:
            where_clauses.append("d.is_test_set = ?")
            params.append(is_test_set)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get total count
        count_sql = f"SELECT COUNT(*) FROM disputes d {where_sql}"
        total = conn.execute(count_sql, params).fetchone()[0]

        # Get records with joined transaction data
        query = f"""
            SELECT
                d.dispute_id,
                d.order_id,
                d.reason_code,
                d.disputed_amount,
                d.ground_truth_label,
                d.is_test_set,
                d.status,
                d.created_at,
                t.customer_id,
                t.currency,
                t.payment_method,
                t.avs_match,
                t.cvv_match,
                t.upi_vpa_match,
                t.ip_distance_km
            FROM disputes d
            LEFT JOIN transactions t ON d.order_id = t.order_id
            {where_sql}
            ORDER BY d.created_at DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, params + [limit, offset]).fetchall()

        disputes = []
        for row in rows:
            disputes.append(
                DisputeItem(
                    dispute_id=row["dispute_id"],
                    order_id=row["order_id"],
                    customer_id=row["customer_id"],
                    reason_code=row["reason_code"],
                    disputed_amount=float(row["disputed_amount"]),
                    currency=row["currency"] or "INR",
                    payment_method=row["payment_method"],
                    ground_truth_label=row["ground_truth_label"],
                    is_test_set=int(row["is_test_set"]),
                    status=row["status"],
                    created_at=row["created_at"],
                    avs_match=row["avs_match"],
                    cvv_match=row["cvv_match"],
                    upi_vpa_match=row["upi_vpa_match"],
                    ip_distance_km=row["ip_distance_km"],
                )
            )

        return DisputeListResponse(total=total, disputes=disputes)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Endpoint 2: POST /api/disputes/score
# ---------------------------------------------------------------------------
@app.post(
    "/api/disputes/score",
    response_model=ScoreResponse,
    tags=["Disputes"],
    summary="Trigger agent reasoning to score a dispute and log decision",
)
def score_dispute_endpoint(payload: ScoreRequest) -> ScoreResponse:
    """
    Score a dispute using the Gemini reasoning agent with FastMCP evidence tools.
    Persists decision in `audit_logs` and marks dispute status as 'scored'.
    """
    dispute_id = payload.dispute_id.strip()

    # Validate dispute exists
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT dispute_id FROM disputes WHERE dispute_id = ?",
            (dispute_id,),
        ).fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dispute with ID '{dispute_id}' not found in database.",
            )
    finally:
        conn.close()

    try:
        # Run agent full pipeline: gather evidence -> call Gemini -> log to audit_logs -> update status
        result = analyze_dispute(dispute_id)

        return ScoreResponse(
            dispute_id=dispute_id,
            decision=result.get("decision", "ACCEPT_DISPUTE"),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning", "No reasoning provided."),
            evidence_letter=result.get("evidence_letter", "No evidence letter generated."),
            tools_used=result.get("tools_used", []),
            customer_notification=result.get("customer_notification", ""),
            contested_amount=result.get("contested_amount"),
            conceded_amount=result.get("conceded_amount"),
            sarcasm_detected=result.get("sarcasm_detected", False),
            fairness_flags=result.get("fairness_flags", []),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent reasoning failed for dispute {dispute_id}: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Endpoint 3: GET /api/audit-logs
# ---------------------------------------------------------------------------
@app.get(
    "/api/audit-logs",
    response_model=AuditLogListResponse,
    tags=["Audit"],
    summary="Get searchable decision history and audit trail",
)
def get_audit_logs(
    search: Optional[str] = Query(
        None,
        description="Search query across dispute_id, decision, reasoning, or evidence letter",
    ),
    dispute_id: Optional[str] = Query(None, description="Exact dispute ID filter"),
    decision: Optional[str] = Query(
        None,
        description="Filter by decision ('CONTEST_DISPUTE' or 'ACCEPT_DISPUTE')",
    ),
    limit: int = Query(100, ge=1, le=500, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> AuditLogListResponse:
    """
    Retrieve full audit logs of past agent decisions with MCP tools invoked,
    confidence ratings, and full reasoning traces.
    """
    conn = get_connection()
    try:
        where_clauses = []
        params: List[Any] = []

        if dispute_id:
            where_clauses.append("dispute_id = ?")
            params.append(dispute_id.strip())

        if decision:
            where_clauses.append("decision = ?")
            params.append(decision.strip())

        if search:
            search_pattern = f"%{search.strip()}%"
            where_clauses.append(
                "(dispute_id LIKE ? OR decision LIKE ? OR reasoning LIKE ? OR evidence_letter LIKE ?)"
            )
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total matching records
        count_sql = f"SELECT COUNT(*) FROM audit_logs {where_sql}"
        total = conn.execute(count_sql, params).fetchone()[0]

        query = f"""
            SELECT
                id,
                dispute_id,
                decision,
                confidence,
                tools_called,
                reasoning,
                evidence_letter,
                customer_notification,
                contested_amount,
                conceded_amount,
                fairness_flags,
                created_at
            FROM audit_logs
            {where_sql}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, params + [limit, offset]).fetchall()

        logs = []
        for r in rows:
            # Parse tools_called JSON string safely
            tools_called_raw = r["tools_called"]
            tools_list: List[str] = []
            if tools_called_raw:
                try:
                    parsed = json.loads(tools_called_raw)
                    if isinstance(parsed, list):
                        tools_list = parsed
                    elif isinstance(parsed, str):
                        tools_list = [parsed]
                except Exception:
                    tools_list = [str(tools_called_raw)]

            # Parse fairness_flags JSON string safely
            fairness_flags_raw = r["fairness_flags"] if "fairness_flags" in r.keys() else None
            fairness_list: List[str] = []
            if fairness_flags_raw:
                try:
                    parsed_flags = json.loads(fairness_flags_raw)
                    if isinstance(parsed_flags, list):
                        fairness_list = parsed_flags
                except Exception:
                    fairness_list = []

            logs.append(
                AuditLogItem(
                    id=r["id"],
                    dispute_id=r["dispute_id"],
                    decision=r["decision"],
                    confidence=float(r["confidence"]),
                    tools_called=tools_list,
                    reasoning=r["reasoning"] or "",
                    evidence_letter=r["evidence_letter"] or "",
                    customer_notification=r["customer_notification"] if ("customer_notification" in r.keys() and r["customer_notification"]) else "",
                    contested_amount=float(r["contested_amount"]) if ("contested_amount" in r.keys() and r["contested_amount"] is not None) else None,
                    conceded_amount=float(r["conceded_amount"]) if ("conceded_amount" in r.keys() and r["conceded_amount"] is not None) else None,
                    fairness_flags=fairness_list,
                    created_at=r["created_at"],
                )
            )

        return AuditLogListResponse(total=total, logs=logs)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Endpoint 4: GET /api/evaluation/metrics
# ---------------------------------------------------------------------------
@app.get(
    "/api/evaluation/metrics",
    response_model=EvaluationMetricsResponse,
    tags=["Evaluation"],
    summary="Run empirical benchmark evaluation over held-out test set",
)
def get_evaluation_metrics(
    rescore: bool = Query(
        False,
        description="If True, force re-runs agent on all test cases (may use API calls). If False, uses existing audit log decisions.",
    ),
    batch_delay: float = Query(
        4.0,
        ge=0.0,
        le=10.0,
        description="Delay in seconds between Gemini API calls if re-scoring test cases",
    ),
) -> EvaluationMetricsResponse:
    """
    Run empirical evaluation against the 20% held-out test dataset (is_test_set = 1).
    Computes precision, recall, F1, confusion matrix, and cost-benefit financial metrics.
    """
    conn = get_connection()
    try:
        # Fetch all held-out test records
        query = """
            SELECT
                d.dispute_id,
                d.order_id,
                d.reason_code,
                d.disputed_amount,
                d.ground_truth_label,
                d.status
            FROM disputes d
            WHERE d.is_test_set = 1
            ORDER BY d.created_at ASC
        """
        test_rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    if not test_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No test set records found (is_test_set = 1). Run generate_data.py to populate synthetic test records.",
        )

    # Standard dispute fee per lost chargeback in India (Stripe India standard)
    DISPUTE_FEE = 1000.0

    y_true: List[int] = []  # 1 = valid_defense (CONTEST), 0 = lost_cause (ACCEPT)
    y_pred: List[int] = []  # 1 = CONTEST_DISPUTE, 0 = ACCEPT_DISPUTE

    test_cases: List[EvaluationCaseDetail] = []

    for i, row in enumerate(test_rows):
        dispute_id = row["dispute_id"]
        order_id = row["order_id"]
        disputed_amount = float(row["disputed_amount"])
        ground_truth_label = row["ground_truth_label"]
        expected_decision = "CONTEST_DISPUTE" if ground_truth_label == "valid_defense" else "ACCEPT_DISPUTE"
        true_binary = 1 if ground_truth_label == "valid_defense" else 0

        # Retrieve prediction
        predicted_decision = None
        confidence = 0.0

        if not rescore:
            # Check if an audit log already exists
            conn = get_connection()
            log_row = conn.execute(
                "SELECT decision, confidence FROM audit_logs WHERE dispute_id = ? ORDER BY id DESC LIMIT 1",
                (dispute_id,),
            ).fetchone()
            conn.close()

            if log_row:
                predicted_decision = log_row["decision"]
                confidence = float(log_row["confidence"])

        # If no previous score or rescore is requested, invoke agent
        if predicted_decision is None:
            if i > 0 and batch_delay > 0:
                time.sleep(batch_delay)
            res = analyze_dispute(dispute_id)
            predicted_decision = res.get("decision", "ACCEPT_DISPUTE")
            confidence = float(res.get("confidence", 0.0))

        pred_binary = 1 if predicted_decision in ("CONTEST_DISPUTE", "PARTIAL_CONTEST") else 0

        y_true.append(true_binary)
        y_pred.append(pred_binary)

        is_correct = (true_binary == pred_binary)

        # Categorize prediction
        if true_binary == 1 and pred_binary == 1:
            classification_type = "TP"
            financial_impact = disputed_amount  # Won winnable dispute
        elif true_binary == 0 and pred_binary == 0:
            classification_type = "TN"
            financial_impact = DISPUTE_FEE  # Avoided dispute fee
        elif true_binary == 0 and pred_binary == 1:
            classification_type = "FP"
            financial_impact = -(disputed_amount + DISPUTE_FEE)  # Wrongly contested and lost
        else:  # true_binary == 1 and pred_binary == 0
            classification_type = "FN"
            financial_impact = -disputed_amount  # Forfeited winnable dispute

        test_cases.append(
            EvaluationCaseDetail(
                dispute_id=dispute_id,
                order_id=order_id,
                disputed_amount=disputed_amount,
                ground_truth_label=ground_truth_label,
                ground_truth_decision=expected_decision,
                predicted_decision=predicted_decision,
                confidence=confidence,
                is_correct=is_correct,
                classification_type=classification_type,
                financial_impact_inr=financial_impact,
            )
        )

    # Calculate scikit-learn metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix extraction
    # labels=[0, 1] ensures 2x2 matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Classification report dict
    report_dict = cast(
        Dict[str, Any],
        classification_report(
            y_true,
            y_pred,
            target_names=["lost_cause (ACCEPT)", "valid_defense (CONTEST)"],
            output_dict=True,
            zero_division=0,
        ),
    )

    # Financial Cost Aggregations
    total_disputed = sum(c.disputed_amount for c in test_cases)
    fp_cost = sum(abs(c.financial_impact_inr) for c in test_cases if c.classification_type == "FP")
    fn_cost = sum(abs(c.financial_impact_inr) for c in test_cases if c.classification_type == "FN")
    tp_saved = sum(c.financial_impact_inr for c in test_cases if c.classification_type == "TP")
    tn_saved = sum(c.financial_impact_inr for c in test_cases if c.classification_type == "TN")
    net_benefit = (tp_saved + tn_saved) - (fp_cost + fn_cost)

    return EvaluationMetricsResponse(
        total_test_records=len(test_cases),
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        confusion_matrix=ConfusionMatrixDetails(
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            true_positive=tp,
            raw_matrix=cm.tolist(),
        ),
        financial_impact=FinancialImpact(
            dispute_fee_per_case=DISPUTE_FEE,
            total_disputed_amount_evaluated=round(total_disputed, 2),
            false_positive_cost=round(fp_cost, 2),
            false_negative_cost=round(fn_cost, 2),
            true_positive_saved=round(tp_saved, 2),
            true_negative_saved=round(tn_saved, 2),
            net_financial_benefit=round(net_benefit, 2),
        ),
        classification_report=report_dict,
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint for Local Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("[main] Starting AI Risk Manager FastAPI Service on http://0.0.0.0:8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
