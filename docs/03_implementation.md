# AI Risk Manager — Phase-wise Implementation Plan

## Goal Description
Build a working **Chargeback Evidence Responder** system that automatically gathers evidence (delivery proofs, chat transcripts, payment verification) using FastMCP tools, evaluates case merit using **Google Gemini API (`google-genai`)** with `gemini-3.5-flash-lite` for agentic reasoning against card network rubrics, and outputs structured dispute evidence packets. We execute this plan using **Subagent-Driven Development**.

---

## User Review Required

> [!IMPORTANT]
> - **Environment Credentials:** The project requires a `GEMINI_API_KEY` configured in a `.env` file for the Google Gemini API.
> - **Data Strategy:** Synthetic data (50+ transactions) will be stored in a local SQLite database (`risk_manager.db`) with an 80/20 train/test split.
> - **Execution Method:** We will use **Subagent-Driven Development**, dispatching isolated subagents for each phase below, followed by strict spec-compliance and code-quality reviews before proceeding.

---

## Execution Strategy: Subagent-Driven Development

To ensure high-quality code without context pollution, this plan will be executed strictly via subagents:
1. We will create a `task.md` checklist based on the phases below.
2. For each phase, an **Implementer Subagent** will be dispatched with isolated context to write the code.
3. Once the Implementer finishes, a **Spec-Reviewer Subagent** and a **Code-Quality Reviewer Subagent** will verify the work.
4. We will only move to the next phase when all tests and reviews pass.

---

## Proposed Implementation Phases

### Phase 1: Environment Setup & Synthetic Data Engine
Set up project dependencies, database schema, and synthetic data generation script.

#### [NEW] requirements.txt
- Include `fastapi`, `uvicorn`, `fastmcp`, `google-genai`, `pandas`, `faker`, `scikit-learn`, `streamlit`, `python-dotenv`.

#### [NEW] database.py
- SQLite database connection & schema definition:
  - `transactions` (id, order_id, customer_id, amount, currency, payment_method, avs_match, cvv_match, upi_vpa_match, ip_address, ip_city, ip_distance_km, created_at)
  - `shipping_logs` (id, order_id, carrier, carrier_status, tracking_number, gps_match, dropoff_lat, dropoff_lng, dropoff_location, gps_accuracy_meters, signature_obtained, delivered_at)
  - `communication_logs` (id, order_id, channel, chat_transcript, created_at)
  - `disputes` (dispute_id, order_id, reason_code, disputed_amount, ground_truth_label, is_test_set, status, created_at)
  - `audit_logs` (id, dispute_id, decision, confidence, tools_called, reasoning, evidence_letter, created_at)

#### [NEW] generate_data.py
- Script using `Faker` and `pandas` to generate 50+ realistic synthetic transaction records across major Indian metros.
- Explicit GPS coordinate modeling (Latitude, Longitude, Landmark, Accuracy Radius) correlated with `gps_match`.
- Populate ground-truth labels (`valid_defense` vs `lost_cause`).
- Flag 20% of records as `is_test_set = 1` for held-out empirical evaluation.

---

### Phase 2: FastMCP Server Layer (Read-Only Data Access)
Build a FastMCP server exposing strictly read-only context tools over the SQLite database.

#### [NEW] mcp_server.py
- MCP server providing 3 read-only tools:
  - `get_transaction(dispute_id)`: Fetches payment details (UPI/Card), AVS match, CVV match, UPI VPA match, IP address, IP city, and IP distance.
  - `get_delivery_proof(order_id)`: Fetches carrier tracking, carrier status, GPS drop-off match (`gps_match`), exact coordinates (`dropoff_lat`, `dropoff_lng`), delivery location (`dropoff_location`), horizontal accuracy (`gps_accuracy_meters`), and signature confirmation.
  - `check_communication_logs(order_id)`: Fetches support chat transcripts to catch friendly fraud.

---

### Phase 3: Agent Reasoning Layer (Google Gemini API & MCP Evidence Gathering)
Implement the core decision engine combining deterministic read-only MCP evidence gathering with Gemini reasoning (4-to-1 API call optimization).

#### [NEW] agent.py
- Use `google-genai` with `gemini-3.5-flash-lite` initialized with `.env` API key.
- Pre-gather evidence via `mcp_server.py` functions in 0 API calls, then execute 1 structured Gemini reasoning call.
- System prompt containing merchant defense rubric based on Visa/Mastercard rules.
- Output formatting: Decision (`CONTEST_DISPUTE` / `ACCEPT_DISPUTE`), confidence score (0.0 to 1.0), reasoning summary, and formal dispute evidence letter.
- Audit logging: writes every decision to `audit_logs` table.

---

### Phase 3.5: Dynamic Live-URL Policy Ingestion & Pinecone Vector Knowledge Engine
Implement a zero-maintenance, dynamic Retrieval-Augmented Generation (RAG) knowledge engine that scrapes and indexes live documentation URLs into **Pinecone Vector Database**, ensuring the AI agent grounds decisions in current network mandates without requiring static PDF downloads.

#### [NEW] policy_sources.json
- Configuration registry of authoritative live documentation endpoints:
  - Stripe Dispute Guidelines (`https://stripe.com/docs/disputes`)
  - Razorpay Chargebacks & Disputes Documentation (`https://razorpay.com/docs/payments/disputes/`)
  - Visa Compelling Evidence 3.0 (CE 3.0) Rule Standards
  - Mastercard Dispute Resolution Guide
  - Merchant Live Terms of Sale & Refund Policy URL

#### [NEW] policy_sync.py
- Live URL crawler using web extraction (Firecrawl MCP / HTTP scraper) to fetch real-time text from registry URLs.
- Semantic rule chunker splitting documentation by Reason Codes (`10.4` Fraud, `13.1` Non-Receipt, `4837`, `4853`) and Product Fulfillment Types (`PHYSICAL_GOODS`, `DIGITAL_SAAS`).
- Pinecone Indexer: Generates vector embeddings for rule chunks and upserts them to the **Pinecone Vector Index** with rich metadata (`rule_id`, `reason_code`, `source_url`, `synced_at`).

#### [NEW] policy_engine.py
- Pinecone Query Interface: Exposes `retrieve_governing_policy(reason_code, payment_method, product_type)` performing semantic vector search against the Pinecone index.
- Returns top-k matching official clauses along with their live authoritative URLs.
- Grounded Rebuttal Synthesis: Injects retrieved live policy clauses and direct hyperlinks into `agent.py` for formal chargeback defense packets with verifiable network citations.

---

### Phase 4: Service Layer (FastAPI Backend)
Expose HTTP REST endpoints for scoring disputes, retrieving audit logs, and running evaluation benchmarks.

#### [NEW] main.py
- `GET /api/disputes/pending`: Returns active dispute queue for the Streamlit inbox table.
- `POST /api/disputes/score`: Triggers Antigravity Agent, writes to `audit_logs`, and returns decision.
- `GET /api/audit-logs`: Returns searchable decision history.
- `GET /api/evaluation/metrics`: Runs batch evaluation over held-out test set.

---

### Phase 5: Quantitative Evaluation Layer
Implement scikit-learn empirical benchmark evaluation over held-out test set.

#### [NEW] evaluate.py
- Batch-scores all test records (`is_test_set = 1`).
- Computes `precision_score`, `recall_score`, `confusion_matrix`, `classification_report`.
- Calculates False-Positive Cost (cost of wrongly fighting or accepting disputes).

---

### Phase 6: Presentation Layer (Streamlit Dashboard)
Create an interactive three-tab Streamlit dashboard.

#### [NEW] app.py
- **Tab 1 · Live Dispute Desk (Inbox Queue)**: Interactive data table of pending disputes, "Analyze" button, live tool execution history, decision meter, and drafted evidence letter.
- **Tab 2 · Audit Trail**: Searchable audit log table showing transaction IDs, MCP tools invoked, raw LLM reasoning, and decisions.
- **Tab 3 · Accuracy Report**: Quantitative dashboard rendering scikit-learn precision, recall, confusion matrix, and false-positive cost analysis.

---

### Phase 7: Two-Sided Justice & Fairness Guardrails
To ensure justice for both the merchant and the customer, the system implements a robust 4-layer fairness framework that considers counter-evidence, applies safety nets, and maintains transparency.

#### [UPDATED] agent.py
- **Decision Fairness (Threshold Autodowngrade):** Introduced a strict `0.60` confidence threshold. Any `CONTEST_DISPUTE` decision falling below this threshold is automatically downgraded to `ACCEPT_DISPUTE`, ensuring the customer receives the benefit of the doubt on borderline cases.
- **Decision Fairness (Partial Contest):** Added support for `PARTIAL_CONTEST` decisions to handle split shipments or partial fulfillment, splitting the disputed amount into `contested_amount` and `conceded_amount`.
- **Communication Fairness:** Generates a `customer_notification` draft explaining the outcome of the dispute directly to the customer in a polite and transparent manner.

#### [UPDATED] mcp_server.py
- **Evidence Fairness:** Added `get_customer_dispute_history` and `get_customer_evidence` tools. The agent now considers customer counter-evidence (e.g., CCTV, police FIR) alongside merchant evidence, avoiding a one-sided evaluation.

#### [UPDATED] database.py & main.py
- **Procedural Fairness:** Added `contested_amount`, `conceded_amount`, `customer_notification`, and `fairness_flags` fields to the `audit_logs` and `disputes` schema for complete transparency and accountability.

---

## Verification Plan

### Automated Verification
- Run `python generate_data.py` and verify SQLite DB tables contain 50+ labeled records with 80/20 test split.
- Run `python evaluate.py` to confirm batch scoring runs cleanly against test records and outputs scikit-learn metrics.

### Manual Verification
- Start FastAPI backend (`uvicorn main:app --reload`).
- Start Streamlit app (`streamlit run app.py`).
- Test Tab 1: Click "Analyze" on a pending dispute, verify live MCP tool execution, and inspect the generated evidence letter.
- Test Tab 2: Confirm audit log entries appear after scoring.
- Test Tab 3: Verify precision, recall, confusion matrix, and false-positive cost rendering cleanly.
