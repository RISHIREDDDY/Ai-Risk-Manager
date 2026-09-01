# AI Risk Manager — Build Specification

## Problem statement

Merchants lose money in three ways: **fraud** (stolen card/identity used to buy), **returns**
(legitimate but costly reverse logistics), and **chargebacks** (bank reverses a payment,
sometimes unfairly). Build a working **detector, verifier, or auto-responder** for ONE of these
loss classes. The system must report measured **precision and recall on a held-out test set** —
not a single cherry-picked example.

## Why now

AI-enabled fraud is rising in Indian BFSI (Banking, Financial Services, Insurance), while returns
and chargebacks quietly erode merchant margin without much tooling to manage them. This is a risk
+ ML problem, not a UX/product problem.

## Pick ONE direction

1. **Chargeback evidence responder** — given a disputed payment, auto-compile evidence
   (delivery proof, order details, communication history) and draft a dispute response.
2. **Return-risk scorer** — given an order at checkout time, predict probability of return.
3. **Fraud-spike detector** — monitor a stream of transactions and flag anomalous spikes/patterns.
4. **Abuse-ring sentinel** — detect clusters/networks of colluding accounts, not single bad actors.

> Decide the direction before scaffolding code. Each direction needs a different data shape and
> different MCP tools (see below).

## Architecture

```
Synthetic transaction data (Faker + pandas)
        │
        ▼
MCP server (exposes data-access tools)
        │
        ▼
Google Gemini API (gemini-3.5-flash-lite, MCP tools, scores/decides)
        │
        ├──► FastAPI backend (serves /score-transaction, /get-exceptions)
        │
        ├──► Evaluation layer (scikit-learn metrics vs held-out labels)
        │
        └──► Streamlit dashboard (live demo + audit trail)
```

**Core approach: use MCP + LLM reasoning instead of training a classifier from scratch.**
Gemini receives a transaction, calls MCP tools to pull supporting context (history, velocity,
device/customer profile, past disputes), and returns a decision + plain-language explanation.
No model training loop required — this is the fast path for a hackathon timeframe. Accuracy is
still proven empirically against labeled data (see Evaluation).

## MCP integration approach

Build a single MCP server that exposes read-only tools over your synthetic dataset. Example
tools (adapt to your chosen direction):

- `get_transaction(transaction_id)` — full details of one transaction
- `get_customer_history(customer_id)` — past orders, disputes, returns for this customer
- `check_velocity(customer_id, window)` — transaction count/amount in a recent time window
- `get_device_fingerprint(transaction_id)` — device/IP/session metadata
- `get_dispute_history(transaction_id)` — prior chargeback/dispute records (for responder direction)
- `get_delivery_proof(order_id)` — shipping/delivery confirmation (for chargeback responder)

Gemini is given these tools via the Gemini API's tool-use interface, plus a system prompt that
states the scoring/decision rubric explicitly. The LLM never has direct database or payment
write-access — MCP tools are read-only; any resulting action (flag, block, respond) is executed
by your own backend code after the LLM's decision, never by the LLM directly.

## Data

- Generate 50+ synthetic transaction records using **Faker** + **pandas**.
- Include ground-truth labels (fraud/not-fraud, returned/not-returned, valid/invalid dispute)
  so you can measure accuracy honestly.
- Split into a held-out test set (e.g. 80/20) that the agent has not seen before scoring.

## Evaluation (the bar)

- Report **precision, recall, and confusion matrix** using `scikit-learn` metrics
  (`precision_score`, `recall_score`, `classification_report`, `confusion_matrix`) comparing the
  agent's decisions against the held-out ground-truth labels.
- Explicitly report **false-positive cost**: what it costs the merchant when a legitimate
  transaction/customer is wrongly flagged (lost sale, annoyed customer, etc.) — not just raw
  accuracy.
- Do not present a single cherry-picked "it worked!" example as proof — the full batch result is
  required.

## Hard constraint: defense-only

The system must only **detect, verify, or respond defensively**. It must never produce output
that could be repurposed to commit fraud, evade detection, or fabricate fraudulent evidence.
Anything offense-capable disqualifies the submission — this applies to prompts, MCP tools, and
any generated text (e.g. chargeback evidence must be truthful, sourced from real order data,
never fabricated).

## Tech stack

| Layer | Tool |
|---|---|
| Synthetic data | Faker, pandas |
| Storage | SQLite or CSV |
| Agent reasoning | Google Gemini API (gemini-3.5-flash-lite, google-genai, MCP tool use) |
| Context access | MCP server (Python or TypeScript MCP SDK) |
| Evaluation | scikit-learn metrics |
| Backend | FastAPI + Uvicorn |
| Demo UI | Streamlit |

## Deliverables

1. MCP server with read-only data-access tools for the chosen direction.
2. Gemini-agent backend that scores/responds per transaction using those tools.
3. Evaluation script producing precision, recall, confusion matrix, and false-positive cost
   commentary on the held-out set.
4. Audit trail: a log (JSON or DB table) of every decision — transaction id, tools called,
   reasoning, final action.
5. Streamlit dashboard showing live scoring plus the final accuracy report.
