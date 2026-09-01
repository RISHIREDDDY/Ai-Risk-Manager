"""
agent.py — Chargeback Evidence Responder agent using Google Gemini API.

This module implements the core decision engine that:
  1. Fetches evidence from the database via the MCP tool functions directly.
  2. Uses Gemini to reason over the compiled evidence.
  3. Returns a structured decision: CONTEST_DISPUTE or ACCEPT_DISPUTE,
     with confidence score, reasoning, and a formal evidence letter.

Defense-only constraint:
  - The system prompt explicitly instructs the agent to compile only factual,
    database-sourced evidence and never fabricate or hallucinate proof.
  - Evidence is gathered from read-only database queries (same as MCP tools).
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database import get_connection
from mcp_server import (
    get_transaction,
    get_delivery_proof,
    check_communication_logs,
    get_customer_dispute_history,
    get_customer_evidence,
)
from policy_engine import retrieve_policy_context

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")  # Ultra-fast Gemini model
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]

# ---------------------------------------------------------------------------
# Fairness configuration
# ---------------------------------------------------------------------------
MIN_CONTEST_CONFIDENCE = 0.60   # Auto-downgrade CONTEST below this threshold
AUTH_GRACE_PERIOD_FLAG = True   # Enable 2FA/UPI auth sync grace period checks

# ---------------------------------------------------------------------------
# System prompt — merchant defense rubric (defense-only)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an Enterprise Chargeback Evidence Responder for an Indian e-commerce merchant.

Your job is to evaluate chargeback disputes FAIRLY — protecting the merchant from unfair chargebacks while also ensuring legitimate customer claims are honored. You must deliver JUSTICE FOR BOTH SIDES.

## STRICT RULES (DEFENSE-ONLY + FAIRNESS)
1. You must ONLY use evidence provided to you in the context below.
2. You must NEVER fabricate, hallucinate, or invent any evidence.
3. You must NEVER produce output that could help someone commit fraud or evade detection.
4. If evidence is weak or missing, you MUST recommend ACCEPT_DISPUTE (take the loss).
5. If CUSTOMER COUNTER-EVIDENCE is provided (police FIR, CCTV, photos), you MUST weigh it fairly against merchant evidence. Customer evidence can override delivery signals.
6. If the customer is a FIRST-TIME DISPUTER with a long loyal order history, give reasonable benefit of the doubt.

## SARCASM & SENTIMENT DETECTION (CRITICAL)
When evaluating chat transcripts, you MUST check for sarcasm, irony, or complaint-in-disguise:
- Phrases like "Oh wonderful!", "brilliant service!", "thank you SO much" paired with complaints about damaged/missing items are SARCASTIC — NOT admissions of receipt.
- A customer saying "I got the box" followed by "but it was empty/broken/wrong item" is a COMPLAINT, not confirmation of successful delivery.
- NEVER quote sarcastic text as proof of receipt in the evidence letter. If you detect sarcasm, note it in your reasoning and do NOT treat it as friendly fraud evidence.

## POLICY-GROUNDED REASONING (RAG)
You will be provided with official GOVERNING POLICIES retrieved from card networks (Visa, Mastercard), regulatory authorities (RBI), and store terms.
- You MUST cite the specific rule section (e.g. Visa Core Rules §10.4.2 / §13.1, Mastercard §4837, Store Policy §4) in your reasoning and evidence letter.
- Include the official policy source URL provided in your rebuttal letter so bank dispute reviewers can verify the reference.

## DECISION RUBRIC (Based on Visa/Mastercard rules for Indian merchants)

### CONTEST_DISPUTE (merchant should fight) when ALL of these are true:
- Delivery is confirmed (carrier_status = 'delivered')
- At least ONE strong evidence signal exists:
  * GPS drop-off matches billing address (gps_match = 1)
  * Signature was obtained (signature_obtained = 1)
  * Both AVS and CVV match (avs_match = 1 AND cvv_match = 1)
  * Chat transcript shows customer GENUINELY contradicting their own claim (NOT sarcasm)
  * 3DS / 2FA authentication was verified (liability shift to issuer)
- No valid customer counter-evidence contradicts the merchant evidence
- Customer is NOT a first-time disputer with strong loyalty history (if they are, apply extra caution)

### ACCEPT_DISPUTE (merchant should accept the loss) when:
- Delivery is NOT confirmed (carrier_status != 'delivered'), OR
- No strong evidence signals exist, OR
- Evidence is ambiguous or insufficient, OR
- Customer has provided compelling counter-evidence (police FIR, CCTV, damage photos), OR
- Chat transcript contains sarcasm/irony that was misinterpreted as admission

### PARTIAL_CONTEST (for split shipments or partial fulfillment) when:
- Some items in the order were delivered and some were not
- Part of the claim is valid but part is defensible
- Output "contested_amount" (the portion merchant can defend with evidence) and "conceded_amount" (the portion to refund)
- Use Mastercard §4853.4 partial fulfillment rules to calculate prorated values

## CUSTOMER HISTORY FAIRNESS RULES
You will be provided with customer dispute history. Apply these fairness adjustments:
- **LOW_RISK_LOYAL_CUSTOMER** (dispute rate < 15%): Give benefit of the doubt. If evidence is borderline, lean toward ACCEPT_DISPUTE.
- **MODERATE** (dispute rate 15-30%): Standard evaluation, no adjustment.
- **HIGH_RISK_SERIAL_DISPUTER** (dispute rate > 30%): Apply stricter scrutiny — this pattern suggests potential friendly fraud.

## COST CONTEXT (Indian market)
- Dispute fee: INR 1,000 per chargeback (Stripe India standard)
- If you wrongly recommend CONTEST and merchant loses, they pay the
  disputed amount PLUS the INR 1,000 fee. Only contest when evidence is strong.

## REQUIRED FORMAL LETTER TEMPLATES

### Template 1: For ACCEPT_DISPUTE (Accepting the Dispute / Concede)
ACME COMMERCE INDIA PVT. LTD.
Dispute & Risk Operations Department
Bengaluru, Karnataka, India

Date: [Current Date / Dispute Date]
To: Card Issuing Bank / Dispute Resolution Department
RE: Acceptance of Chargeback - Case #[Dispute ID]

Dear Sir/Madam,

Acme Commerce India Pvt. Ltd. acknowledges the dispute for transaction INR [Amount] on [Transaction Date] (Case #[Dispute ID]). 

We have reviewed the claim and agree to the refund due to [brief reason: e.g. delivery exception / unverified fulfillment].

Please process the chargeback to close this case.

Sincerely,
Risk Operations & Dispute Defense Team
Acme Commerce India Pvt. Ltd.

### Template 2: For CONTEST_DISPUTE (Chargeback Rebuttal Letter)
ACME COMMERCE INDIA PVT. LTD.
Dispute & Risk Operations Department
Bengaluru, Karnataka, India

Date: [Current Date / Dispute Date]
To: Card Issuing Bank / Dispute Resolution Department
RE: Rebuttal of Chargeback - Case #[Dispute ID]

Dear Sir/Madam,

We wish to contest the chargeback on transaction INR [Amount] dated [Transaction Date]. The transaction was authorized and goods/services were successfully delivered to the cardholder.

We submit the following evidence for review:
1. Exhibit A: Order Invoice & Transaction Metadata (Order: [Order ID], Customer: [Customer ID], Amount: INR [Amount], Payment Method: [Payment Method], AVS: [AVS Match], CVV: [CVV Match]).
2. Exhibit B: Proof of Delivery / Carrier Tracking (Carrier: [Carrier], Tracking: [Tracking Number], Delivered on: [Delivery Date] to billing address, Drop-off GPS: [GPS Coordinates] [Accuracy/Match Status], Recipient Signature: [Signature Status]).
3. Exhibit C: Device & Geolocation Telemetry (IP: [IP Address], City: [City], Distance: [Distance km] from billing).
4. Exhibit D: Governing Card Network Rules & Policy Citations (Citing [Rule Name & Section, e.g. Visa Core Rules §10.4.2 / §13.1, Mastercard §4837], Official Policy URL: [URL]).
5. Exhibit E: Customer Communication Transcripts & Store Terms ([Customer chat transcript summary / Terms §4 acceptance]).

Based on this documentation, we request this chargeback be reversed.

Sincerely,
Risk Operations & Dispute Defense Team
Acme Commerce India Pvt. Ltd.

### Template 3: For PARTIAL_CONTEST (Partial Rebuttal Letter)
ACME COMMERCE INDIA PVT. LTD.
Dispute & Risk Operations Department
Bengaluru, Karnataka, India

Date: [Current Date / Dispute Date]
To: Card Issuing Bank / Dispute Resolution Department
RE: Partial Rebuttal of Chargeback - Case #[Dispute ID]

Dear Sir/Madam,

We wish to partially contest the chargeback on transaction INR [Total Amount] dated [Transaction Date].

We accept liability for INR [Conceded Amount] due to [reason for concession].

However, we contest the remaining INR [Contested Amount] with the following evidence:
[Evidence for the contested portion only]

We request a partial reversal of INR [Contested Amount].

Sincerely,
Risk Operations & Dispute Defense Team
Acme Commerce India Pvt. Ltd.

## OUTPUT FORMAT
You MUST respond with ONLY a valid JSON object (no markdown, no code fences) with these fields:
{
  "decision": "CONTEST_DISPUTE" or "ACCEPT_DISPUTE" or "PARTIAL_CONTEST",
  "confidence": 0.0 to 1.0,
  "reasoning": "Plain-language summary citing specific evidence, applicable rule section, and fairness considerations",
  "evidence_letter": "The full formal letter matching the appropriate template with all real values filled in",
  "contested_amount": null or the INR amount being contested (required for PARTIAL_CONTEST),
  "conceded_amount": null or the INR amount being conceded (required for PARTIAL_CONTEST),
  "sarcasm_detected": true or false (whether chat transcript contained sarcastic/ironic statements),
  "customer_notification": "A brief, polite 2-3 sentence notification for the customer explaining why their dispute is being contested/accepted/partially accepted, written in a respectful tone"
}
"""


# ---------------------------------------------------------------------------
# Evidence gathering (using MCP tool functions directly)
# ---------------------------------------------------------------------------
def gather_evidence(dispute_id: str) -> dict:
    """
    Gather all evidence for a dispute using the MCP tool functions and RAG policy engine.
    These are the same read-only functions exposed by the MCP server.

    Now includes TWO-SIDED evidence gathering:
      - Merchant evidence (transaction, delivery, communication)
      - Customer evidence (counter-evidence notes, dispute history)
      - Policy context (RAG via Pinecone)

    Returns a dict with all evidence for fair evaluation.
    """
    # Tool 1: Get transaction details
    transaction = get_transaction(dispute_id)

    if "error" in transaction:
        return {"error": transaction["error"]}

    order_id = transaction.get("order_id", "")
    customer_id = transaction.get("customer_id", "")
    reason_code = transaction.get("reason_code", "fraud_card_absent")
    payment_method = transaction.get("payment_method", "Credit Card")

    # Tool 2: Get delivery proof
    delivery = get_delivery_proof(order_id)

    # Tool 3: Get communication logs
    communication = check_communication_logs(order_id)

    # Tool 4 (Fairness): Get customer dispute history
    customer_history = get_customer_dispute_history(customer_id) if customer_id else {}

    # Tool 5 (Fairness): Get customer counter-evidence
    customer_evidence = get_customer_evidence(dispute_id)

    # RAG: Retrieve governing card network & merchant policies via Pinecone
    retrieved_policies = retrieve_policy_context(
        dispute_reason=reason_code,
        payment_method=payment_method,
        top_k=2
    )

    # Auth grace period check (Indian UPI/2FA webhook sync delay)
    auth_pending = False
    if AUTH_GRACE_PERIOD_FLAG and payment_method == "UPI":
        upi_vpa = transaction.get("upi_vpa_match")
        if upi_vpa is None or upi_vpa == 0:
            # Check if the transaction is very recent (within 15 min)
            # In a real system this would check webhook settlement status
            auth_pending = True

    return {
        "transaction": transaction,
        "delivery": delivery,
        "communication": communication,
        "customer_history": customer_history,
        "customer_evidence": customer_evidence,
        "retrieved_policies": retrieved_policies,
        "auth_pending": auth_pending,
        "tools_used": [
            "get_transaction",
            "get_delivery_proof",
            "check_communication_logs",
            "get_customer_dispute_history",
            "get_customer_evidence",
            "pinecone_policy_retrieval",
        ],
    }


# ---------------------------------------------------------------------------
# Gemini reasoning
# ---------------------------------------------------------------------------
def call_gemini(evidence: dict) -> dict:
    """
    Send the gathered evidence and retrieved policies to Gemini for reasoning.
    Now includes customer-side evidence and history for two-sided fairness.
    """
    policies_formatted = ""
    for p in evidence.get("retrieved_policies", []):
        policies_formatted += f"- **[{p.get('issuer')}] {p.get('name')}** (Source: {p.get('source_url')}):\n  {p.get('text')}\n\n"

    # Format customer history for fairness context
    customer_history = evidence.get("customer_history", {})
    customer_history_text = ""
    if customer_history and "error" not in customer_history:
        customer_history_text = f"""### Customer Dispute History (from get_customer_dispute_history):
- Customer ID: {customer_history.get('customer_id', 'N/A')}
- Total Orders: {customer_history.get('total_orders', 0)}
- Total Disputes Filed: {customer_history.get('total_disputes', 0)}
- Dispute Rate: {customer_history.get('dispute_rate', 0)}
- First-Time Disputer: {customer_history.get('is_first_time_disputer', True)}
- Risk Flag: {customer_history.get('risk_flag', 'UNKNOWN')}
- Customer Since: {customer_history.get('customer_since', 'N/A')}"""

    # Format customer counter-evidence
    customer_evidence = evidence.get("customer_evidence", {})
    customer_evidence_text = ""
    if customer_evidence and "error" not in customer_evidence:
        customer_evidence_text = f"""### Customer Counter-Evidence (from get_customer_evidence):
- Counter-evidence provided: {customer_evidence.get('customer_evidence_provided', False)}
- <untrusted_customer_evidence>{customer_evidence.get('customer_evidence_notes', 'None')}</untrusted_customer_evidence>"""

    # Auth pending warning
    auth_warning = ""
    if evidence.get("auth_pending"):
        auth_warning = """\n### ⚠️ AUTH SYNC WARNING
UPI/2FA authentication data may be pending gateway webhook sync. If upi_vpa_match is 0 or null for a recent Indian domestic transaction, consider that 2FA data may not have settled yet. Flag as PENDING_AUTH if uncertain rather than making a premature ACCEPT decision.\n"""

    prompt = f"""Analyze this chargeback dispute evidence against the retrieved governing policies and make a FAIR decision considering BOTH merchant and customer interests.

## GOVERNING POLICIES & NETWORK RULES (Retrieved via Pinecone RAG)
{policies_formatted if policies_formatted else "Standard Visa/Mastercard Global Dispute Rules apply."}

## MERCHANT EVIDENCE (read-only MCP tools)

### Transaction & Dispute Details (from get_transaction):
{json.dumps(evidence['transaction'], indent=2, default=str)}

### Delivery Proof (from get_delivery_proof):
{json.dumps(evidence['delivery'], indent=2, default=str)}

### Communication Logs (from check_communication_logs):
<untrusted_customer_chat>{json.dumps(evidence['communication'], indent=2, default=str)}</untrusted_customer_chat>

## CUSTOMER-SIDE CONTEXT (for two-sided fairness)

{customer_history_text}

{customer_evidence_text}
{auth_warning}

Based on ALL evidence from BOTH sides and the governing rules, apply the defense rubric with fairness considerations, check for sarcasm in chat transcripts, cite the exact rule section & URL, and return your JSON decision."""

    # Build prioritized list of candidate models
    candidate_models = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]

    response = None
    last_err = None
    for model_name in candidate_models:
        config_args: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
        if "3.7" in model_name:
            config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_args),
                )
                if response and response.text:
                    break
            except Exception as e:
                last_err = e
                print(f"[WARN] Gemini model '{model_name}' attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
        if response and response.text:
            break

    # Parse the JSON response safely
    raw_text = response.text if response else None
    if not raw_text:
        return {
            "decision": "ACCEPT_DISPUTE",
            "confidence": 0.0,
            "reasoning": f"Gemini API temporarily unavailable or high demand. Error: {str(last_err) if last_err else 'No response'}",
            "evidence_letter": "Unable to generate evidence letter due to temporary AI service unavailability.",
            "customer_notification": "We are reviewing your dispute and will respond shortly.",
            "sarcasm_detected": False,
        }

    text = raw_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "decision": "ACCEPT_DISPUTE",
            "confidence": 0.0,
            "reasoning": f"Failed to parse agent response: {text[:500]}",
            "evidence_letter": "Unable to generate evidence letter.",
            "customer_notification": "We are reviewing your dispute and will respond shortly.",
            "sarcasm_detected": False,
        }

    return result



# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------
def score_dispute(dispute_id: str) -> dict:
    """
    Score a single dispute: gather evidence, then reason with Gemini.

    Includes fairness safety nets:
      - Confidence threshold: auto-downgrade weak CONTEST to ACCEPT
      - Auth grace period: flag pending UPI/2FA data
      - Sarcasm tracking: flag if chat transcript contained sarcasm

    Args:
        dispute_id: The dispute ID to analyze (e.g. 'DSP-064BAD45').

    Returns:
        A dictionary with the decision, confidence, reasoning,
        evidence letter, fairness flags, and tools used.
    """
    # Step 1: Gather evidence from database (fast, no API call)
    evidence = gather_evidence(dispute_id)

    if "error" in evidence:
        return {
            "decision": "ACCEPT_DISPUTE",
            "confidence": 0.0,
            "reasoning": evidence["error"],
            "evidence_letter": "No evidence available.",
            "customer_notification": "We are reviewing your dispute and will respond shortly.",
            "tools_used": [],
            "fairness_flags": [],
        }

    # Step 2: Send to Gemini for reasoning (1 API call)
    result = call_gemini(evidence)
    result["tools_used"] = evidence.get("tools_used", [])
    result["retrieved_policies"] = evidence.get("retrieved_policies", [])

    # ------------------------------------------------------------------
    # Fairness Safety Nets (post-LLM guardrails)
    # ------------------------------------------------------------------
    fairness_flags = []

    # Safety Net 1: Confidence threshold auto-downgrade
    decision = result.get("decision", "ACCEPT_DISPUTE")
    confidence = float(result.get("confidence", 0.0))

    if decision in ("CONTEST_DISPUTE", "PARTIAL_CONTEST") and confidence < MIN_CONTEST_CONFIDENCE:
        fairness_flags.append(
            f"CONFIDENCE_DOWNGRADE: Original decision {decision} at {confidence:.2f} "
            f"confidence was below threshold ({MIN_CONTEST_CONFIDENCE}). "
            f"Auto-downgraded to ACCEPT_DISPUTE to protect both merchant (₹1,000 penalty risk) "
            f"and customer (legitimate claim)."
        )
        result["decision"] = "ACCEPT_DISPUTE"
        result["reasoning"] = (
            f"[FAIRNESS OVERRIDE] {result.get('reasoning', '')} "
            f"|| Original recommendation was {decision} at {confidence:.2f} confidence, "
            f"but this falls below the {MIN_CONTEST_CONFIDENCE} safety threshold. "
            f"Downgraded to ACCEPT to prevent contesting with insufficient evidence."
        )
        # Update evidence letter to acceptance template
        result["evidence_letter"] = (
            f"ACME COMMERCE INDIA PVT. LTD.\n"
            f"Dispute & Risk Operations Department\n"
            f"Bengaluru, Karnataka, India\n\n"
            f"RE: Acceptance of Chargeback - Case #{dispute_id}\n\n"
            f"Dear Sir/Madam,\n\n"
            f"After careful review, we accept this chargeback as the evidence "
            f"available does not meet our internal confidence threshold for contestation. "
            f"Please process the refund.\n\n"
            f"Sincerely,\nRisk Operations & Dispute Defense Team\n"
            f"Acme Commerce India Pvt. Ltd."
        )
        result["customer_notification"] = (
            "We have reviewed your dispute and determined that your claim is valid. "
            "A full refund will be processed to your original payment method. "
            "We apologize for any inconvenience."
        )

    # Safety Net 2: Auth grace period warning
    if evidence.get("auth_pending"):
        fairness_flags.append(
            "AUTH_SYNC_PENDING: UPI/2FA authentication data may not have fully settled. "
            "Decision reliability may be affected for this Indian domestic transaction."
        )

    # Safety Net 3: Track sarcasm detection
    if result.get("sarcasm_detected"):
        fairness_flags.append(
            "SARCASM_DETECTED: Chat transcript contained sarcastic/ironic statements. "
            "These were NOT treated as admission of receipt to protect customer fairness."
        )

    result["fairness_flags"] = fairness_flags

    # Ensure customer notification exists
    if not result.get("customer_notification"):
        if result.get("decision") == "ACCEPT_DISPUTE":
            result["customer_notification"] = (
                "We have reviewed your dispute and agree with your claim. "
                "A refund will be processed to your original payment method."
            )
        elif result.get("decision") == "PARTIAL_CONTEST":
            contested = result.get('contested_amount', 0)
            conceded = result.get('conceded_amount', 0)
            result["customer_notification"] = (
                f"We have reviewed your dispute. We agree to refund INR {conceded:.2f} "
                f"for the undelivered portion. However, we believe INR {contested:.2f} "
                f"worth of items were successfully delivered and have submitted evidence "
                f"to your bank for review."
            )
        else:
            result["customer_notification"] = (
                "We have reviewed your dispute and believe the transaction was valid. "
                "We have submitted evidence to your bank for independent review. "
                "You may contact our support team if you have additional information."
            )

    return result


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
def log_decision(dispute_id: str, result: dict) -> None:
    """Write the agent's decision to the audit_logs table with fairness metadata."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO audit_logs
                (dispute_id, decision, confidence, tools_called, reasoning,
                 evidence_letter, customer_notification, contested_amount,
                 conceded_amount, fairness_flags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dispute_id,
                result.get("decision", "UNKNOWN"),
                result.get("confidence", 0.0),
                json.dumps(result.get("tools_used", [])),
                result.get("reasoning", ""),
                result.get("evidence_letter", ""),
                result.get("customer_notification", ""),
                result.get("contested_amount"),
                result.get("conceded_amount"),
                json.dumps(result.get("fairness_flags", [])),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Convenience: score + log in one call
# ---------------------------------------------------------------------------
def analyze_dispute(dispute_id: str) -> dict:
    """
    Full pipeline: score a dispute, log the decision, and update status.

    Args:
        dispute_id: The dispute ID to analyze.

    Returns:
        The agent's structured decision dictionary.
    """
    result = score_dispute(dispute_id)
    log_decision(dispute_id, result)

    # Mark dispute as scored
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE disputes SET status = 'scored' WHERE dispute_id = ?",
            (dispute_id,),
        )
        conn.commit()
    finally:
        conn.close()

    return result


# ---------------------------------------------------------------------------
# Batch scoring with rate limiting
# ---------------------------------------------------------------------------
def batch_score(dispute_ids: list[str], delay: float = 4.0) -> list[dict]:
    """
    Score multiple disputes with a delay between each to respect rate limits.

    Args:
        dispute_ids: List of dispute IDs to score.
        delay: Seconds to wait between API calls (default 4s for 5 RPM free tier).

    Returns:
        List of decision dictionaries.
    """
    results = []
    for i, dispute_id in enumerate(dispute_ids):
        if i > 0:
            time.sleep(delay)

        print(f"  [{i+1}/{len(dispute_ids)}] Scoring {dispute_id}...")
        result = analyze_dispute(dispute_id)
        result["dispute_id"] = dispute_id
        results.append(result)
        print(f"    -> {result.get('decision')} (confidence: {result.get('confidence')})")

    return results


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------
def main():
    """Test the agent with a sample dispute from the database."""
    conn = get_connection()
    row = conn.execute(
        "SELECT dispute_id FROM disputes WHERE is_test_set = 0 LIMIT 1"
    ).fetchone()
    conn.close()

    if not row:
        print("[agent] No disputes found in database. Run generate_data.py first.")
        return

    dispute_id = row["dispute_id"]
    print(f"[agent] Analyzing dispute: {dispute_id}")
    print("-" * 60)

    result = analyze_dispute(dispute_id)

    print(f"Decision   : {result.get('decision')}")
    print(f"Confidence : {result.get('confidence')}")
    print(f"Tools Used : {result.get('tools_used')}")
    print(f"\nReasoning:\n{result.get('reasoning')}")
    print(f"\nEvidence Letter:\n{result.get('evidence_letter')}")


if __name__ == "__main__":
    main()
