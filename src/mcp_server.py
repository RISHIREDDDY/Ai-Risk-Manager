"""
mcp_server.py — FastMCP server for AI Risk Manager.

Exposes strictly READ-ONLY tools over the SQLite database so that the
Gemini reasoning agent can pull evidence context without any ability to
modify, delete, or fabricate data.

Tools:
  1. get_transaction(dispute_id)        — payment & identity verification signals
  2. get_delivery_proof(order_id)       — shipping & delivery confirmation
  3. check_communication_logs(order_id) — customer support chat transcripts

Defense-only constraint:
  - Every tool is a SELECT query; no INSERT/UPDATE/DELETE is ever executed.
  - Tools return only factual data from the database.
  - No tool can write, modify, or generate synthetic evidence.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastmcp import FastMCP
from database import get_connection

# ---------------------------------------------------------------------------
# Server initialisation
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="AI Risk Manager MCP",
    instructions=(
        "You are a chargeback defense assistant. Use these read-only tools "
        "to gather factual evidence about disputed transactions. Never "
        "fabricate evidence. Only compile what exists in the database."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: get_transaction
# ---------------------------------------------------------------------------
@mcp.tool()
def get_transaction(dispute_id: str) -> dict:
    """
    Fetch full transaction and dispute details for a given dispute ID.

    Returns payment method (UPI/CARD), AVS match, CVV match, UPI VPA match,
    IP address, IP-to-billing distance, order amount, reason code, and
    disputed amount. This data helps evaluate whether the cardholder
    actually authorised the transaction.

    Args:
        dispute_id: The unique dispute identifier (e.g. 'DSP-A1B2C3D4').

    Returns:
        A dictionary with transaction and dispute details, or an error message.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                d.dispute_id,
                d.order_id,
                d.reason_code,
                d.disputed_amount,
                d.status         AS dispute_status,
                d.created_at     AS dispute_date,
                t.customer_id,
                t.amount         AS transaction_amount,
                t.currency,
                t.payment_method,
                t.avs_match,
                t.cvv_match,
                t.upi_vpa_match,
                t.ip_address,
                t.ip_city,
                t.ip_distance_km,
                t.created_at     AS transaction_date
            FROM disputes d
            JOIN transactions t ON d.order_id = t.order_id
            WHERE d.dispute_id = ?
            """,
            (dispute_id,),
        ).fetchone()

        if not row:
            return {"error": f"No dispute found with ID '{dispute_id}'."}

        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 2: get_delivery_proof
# ---------------------------------------------------------------------------
@mcp.tool()
def get_delivery_proof(order_id: str) -> dict:
    """
    Fetch shipping and delivery evidence for a given order ID.

    Returns carrier name, delivery status, tracking number, GPS drop-off
    match (whether the parcel GPS matches the billing address), GPS coordinates
    (dropoff_lat, dropoff_lng, dropoff_location, gps_accuracy_meters), signature
    confirmation, and delivery timestamp. This is critical evidence for
    'product not received' disputes.

    Args:
        order_id: The unique order identifier (e.g. 'ORD-A1B2C3D4').

    Returns:
        A dictionary with shipping evidence, or an error message.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                order_id,
                carrier,
                carrier_status,
                tracking_number,
                gps_match,
                dropoff_lat,
                dropoff_lng,
                dropoff_location,
                gps_accuracy_meters,
                signature_obtained,
                delivered_at
            FROM shipping_logs
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

        if not row:
            return {"error": f"No shipping record found for order '{order_id}'."}

        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 3: check_communication_logs
# ---------------------------------------------------------------------------
@mcp.tool()
def check_communication_logs(order_id: str) -> dict:
    """
    Fetch customer support chat transcripts for a given order ID.

    Returns the communication channel and full chat transcript. This helps
    identify friendly fraud patterns — for example, a customer claiming
    non-delivery despite delivery evidence, or contradicting their own
    prior statements.

    Args:
        order_id: The unique order identifier (e.g. 'ORD-A1B2C3D4').

    Returns:
        A dictionary with the chat transcript, or an error message.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                order_id,
                channel,
                chat_transcript,
                created_at AS chat_date
            FROM communication_logs
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

        if not row:
            return {"error": f"No communication logs found for order '{order_id}'."}

        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 4: get_customer_dispute_history (Fairness: two-sided justice)
# ---------------------------------------------------------------------------
@mcp.tool()
def get_customer_dispute_history(customer_id: str) -> dict:
    """
    Fetch the dispute history for a customer to enable fair differentiation.

    A first-time customer with many successful orders who files 1 dispute
    deserves benefit of the doubt. A customer who has filed 10 disputes in
    3 months likely indicates a friendly fraud pattern. This tool enables
    the agent to weigh customer history for two-sided fairness.

    Args:
        customer_id: The unique customer identifier (e.g. 'CUST-A1B2C3').

    Returns:
        A dictionary with total orders, total disputes, dispute rate,
        past dispute outcomes, and customer tenure.
    """
    conn = get_connection()
    try:
        # Total orders for this customer
        total_orders = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]

        if total_orders == 0:
            return {"error": f"No customer found with ID '{customer_id}'."}

        # Total disputes filed
        disputes_row = conn.execute(
            """
            SELECT COUNT(*) as total_disputes
            FROM disputes d
            JOIN transactions t ON d.order_id = t.order_id
            WHERE t.customer_id = ?
            """,
            (customer_id,),
        ).fetchone()
        total_disputes = disputes_row[0]

        # Past dispute decisions from audit logs
        past_decisions = conn.execute(
            """
            SELECT d.dispute_id, a.decision, a.confidence, d.disputed_amount
            FROM disputes d
            JOIN transactions t ON d.order_id = t.order_id
            LEFT JOIN audit_logs a ON d.dispute_id = a.dispute_id
            WHERE t.customer_id = ?
            ORDER BY d.created_at DESC
            LIMIT 10
            """,
            (customer_id,),
        ).fetchall()

        # Customer first order date (tenure)
        first_order = conn.execute(
            "SELECT MIN(created_at) FROM transactions WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()[0]

        dispute_rate = round(total_disputes / total_orders, 3) if total_orders > 0 else 0.0

        return {
            "customer_id": customer_id,
            "total_orders": total_orders,
            "total_disputes": total_disputes,
            "dispute_rate": dispute_rate,
            "customer_since": first_order,
            "is_first_time_disputer": total_disputes <= 1,
            "risk_flag": "HIGH_RISK_SERIAL_DISPUTER" if dispute_rate > 0.3 else (
                "MODERATE" if dispute_rate > 0.15 else "LOW_RISK_LOYAL_CUSTOMER"
            ),
            "past_decisions": [
                {
                    "dispute_id": r["dispute_id"],
                    "decision": r["decision"],
                    "confidence": float(r["confidence"]) if r["confidence"] else None,
                    "amount": float(r["disputed_amount"]),
                }
                for r in past_decisions
            ],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 5: get_customer_evidence (Fairness: consider counter-evidence)
# ---------------------------------------------------------------------------
@mcp.tool()
def get_customer_evidence(dispute_id: str) -> dict:
    """
    Fetch any counter-evidence or notes provided by the customer for this dispute.

    This enables two-sided justice: the agent considers not just merchant
    evidence (delivery GPS, signatures) but also the customer's side
    (police FIR, CCTV proof, photos of damaged goods, etc.).

    Args:
        dispute_id: The unique dispute identifier (e.g. 'DSP-A1B2C3D4').

    Returns:
        A dictionary with customer evidence notes, or a message if none provided.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT dispute_id, customer_evidence_notes FROM disputes WHERE dispute_id = ?",
            (dispute_id,),
        ).fetchone()

        if not row:
            return {"error": f"No dispute found with ID '{dispute_id}'."}

        notes = row["customer_evidence_notes"]
        return {
            "dispute_id": dispute_id,
            "customer_evidence_provided": notes is not None and len(str(notes).strip()) > 0,
            "customer_evidence_notes": notes or "No counter-evidence submitted by customer.",
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point — run as standalone MCP server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[mcp_server] Starting AI Risk Manager MCP server...")
    mcp.run()
