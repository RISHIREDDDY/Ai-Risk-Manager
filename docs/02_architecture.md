# Dispute Resolution Agent — Layered Architecture

This document presents the overall system architecture for the **AI Risk Manager (Chargeback Evidence Responder)** using **Google Gemini API**, **FastMCP**, **FastAPI**, **SQLite**, **scikit-learn**, and **Streamlit**.

---

## 1. Executive Abstraction & Visual Architecture

### The Problem It Solves
Merchants lose significant margin to "friendly fraud" and illegitimate chargebacks, but fighting them manually is too slow and expensive. This architecture solves that by acting as an **Automated Chargeback Evidence Responder**. It autonomously gathers relevant shipping, customer, and payment data, evaluates it against strict Visa/Mastercard rules, and drafts a compelling, network-compliant dispute response—saving the merchant money with zero manual investigation.

### How the Architecture Works (Diagram Explanation)
The system is built on a modern, decoupled **Agentic LLM stack**. 
Instead of relying on a black-box machine learning classifier, it uses **Google Gemini** as a reasoning engine (Agent Reasoning Layer). When a dispute arrives via the **Streamlit UI** (Presentation Layer), the backend (Service Layer) wakes up the agent. The agent then dynamically pulls required context—like GPS delivery matches or chat logs—from a secure SQLite database (Data Layer) using explicitly defined, read-only **FastMCP Tools** (MCP Server Layer). Finally, the agent synthesizes this evidence into a formal defense p```mermaid
graph TD
    subgraph Presentation Layer
        UI[Streamlit Dashboard<br/>Tab 1: Live Dispute Desk<br/>Tab 2: Audit Trail<br/>Tab 3: Accuracy Report]
    end

    subgraph Service Layer
        API[FastAPI + Uvicorn<br/>/api/disputes/score<br/>/api/audit-logs<br/>/api/evaluation/metrics]
    end

    subgraph Agent Reasoning Layer
        LLM[Google Gemini Reasoning Engine<br/>Evidence Evaluation & Legal Synthesis]
    end

    subgraph Dynamic RAG Knowledge Layer
        URLS[Live Authoritative URLs<br/>• Stripe Dispute Docs<br/>• Razorpay Chargeback Rules<br/>• Visa / Mastercard Portals<br/>• Merchant Live Terms of Service]
        CRAWLER[Automated Web Ingestion & Chunker<br/>Semantic Policy Parser]
        PINECONE[(Pinecone Vector DB<br/>Embedded Policy Rules & URLs)]
        
        URLS --> CRAWLER
        CRAWLER --> PINECONE
    end

    subgraph MCP Server Layer
        MCP[FastMCP Server<br/>Read-Only Database Context Tools]
    end

    subgraph Data Layer
        DB[(SQLite Database<br/>Transactions · Shipping · Support)]
    end

    UI -- HTTP POST/GET --> API
    API -- Dispute Triage --> LLM
    LLM -- Semantic Rule Query --> PINECONE
    PINECONE -- Matched Policy Clauses + Live URLs --> LLM
    LLM -- Evidence Gathering --> MCP
    MCP -- SQL Queries --> DB
```

---

## 2. Layered Architecture Breakdown

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  📊 Presentation Layer — Streamlit Dashboard                                                          │
│                                                                                                        │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │ 🔨 Tab 1 · Live Dispute Desk │  │ 🔍 Tab 2 · Audit Trail      │  │ 📊 Tab 3 · Accuracy Report   │  │
│  │ Interactive response gen    │  │ Searchable decision history  │  │ Precision · Recall · FP cost │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                            HTTP REST / JSON
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ⚙️ Service Layer — FastAPI + Uvicorn                                                                  │
│                                                                                                        │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │ POST /api/disputes/score     │  │ GET /api/audit-logs          │  │ GET /api/evaluation/metrics │  │
│  │ Triggers reasoning loop      │  │ Retrieves decision history   │  │ Runs benchmark batch         │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                             agent invocation
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🧠 Agent Reasoning Layer — Google Gemini Engine                                                      │
│                                                                                                        │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │ ✦ Gemini 2.5 / 1.5 Flash     │  │ 📋 Policy Grounded Reasoning │  │ 📄 Response Packet           │  │
│  │ LLM Multi-Agent Orchestration│  │ Case Facts vs. Network Rules │  │ Clause Citations + Live URLs │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                    │                                                               │
     Semantic Vector Query                                                Read-Only Data Ingestion
                    ▼                                                               ▼
┌──────────────────────────────────────────────┐      ┌──────────────────────────────────────────────────┐
│  🌲 Pinecone Vector Knowledge Layer (RAG)    │      │  🔌 MCP Server Layer — FastMCP (Read-Only)       │
│                                              │      │                                                  │
│  • Live Web Ingestion (Stripe / Razorpay)    │      │  • get_transaction(dispute_id)                  │
│  • Visa Compelling Evidence 3.0 (§10.4)      │      │  • get_delivery_proof(order_id)                  │
│  • Mastercard Chargeback Guide (§4837/4853)  │      │  • check_communication_logs(order_id)            │
│  • Merchant Dynamic Store Terms & Refund URL │      │                                                  │
└──────────────────────────────────────────────┘      └──────────────────────────────────────────────────┘
                    ▲                                                               │
         Automated Live URL Sync                                              read-only SQL
                    │                                                               ▼
┌──────────────────────────────────────────────┐      ┌──────────────────────────────────────────────────┐
│  🌐 Authoritative Live Policy URLs           │      │  🗄️ Data Layer — SQLite Database                 │
│  Stripe / Razorpay / Visa / Merchant Website │      │  disputes · transactions · shipping · chat logs │
└──────────────────────────────────────────────┘      └──────────────────────────────────────────────────┘
```

---

## 3. Detailed Layer Specifications

### 3.1 Presentation Layer — Streamlit Dashboard
The front-end user interface providing interactive merchant workflows and real-time model evaluation metrics:
- **Tab 1 · Live Dispute Desk (The Inbox)**: Triage queue displaying pending disputes. Clicking "Analyse" triggers the multi-agent investigation, retrieves case evidence, evaluates policy compliance via Pinecone RAG, and renders a ready-to-submit Chargeback Rebuttal Letter.
- **Tab 2 · Audit Trail**: Searchable, transparent audit log displaying transaction IDs, verified evidence, raw LLM reasoning, confidence scores, and final decisions (`CONTEST_DISPUTE` vs `ACCEPT_DISPUTE`).
- **Tab 3 · Accuracy Report**: Quantitative evaluation dashboard showing Precision, Recall, Confusion Matrix, and False-Positive Cost analysis on the held-out test dataset.

### 3.2 Service Layer — FastAPI + Uvicorn
The backend application services managing request orchestration, decision persistence, and batch evaluation runs:
- `POST /api/disputes/score`: Accepts `dispute_id`, initiates the Gemini reasoning loop with FastMCP evidence tools and Pinecone RAG policy retrieval, logs audit trail, and returns final decision.
- `GET /api/audit-logs`: Retrieves past audit entries for the Streamlit audit trail UI.
- `GET /api/evaluation/metrics`: Runs batch evaluation across the 20% held-out test dataset using `scikit-learn` and returns benchmark metrics.

### 3.3 Agent Reasoning Layer — Google Gemini Engine
The core decision engine combining LLM reasoning with automated context retrieval:
- **Gemini 2.5 / 1.5 Flash**: Orchestrates multi-agent reasoning over case facts and retrieved policy clauses.
- **Policy-Grounded Reasoning**: Evaluates incoming dispute details against exact network guidelines (e.g., carrier signature proof, GPS drop-off confirmation within 50m, chat transcripts acknowledging delivery).
- **Response Packet with Live URL Citations**: Synthesizes evidence into a formal rebuttal letter citing exact Visa/Mastercard reason code rules, merchant policy clauses, and live verified source links.

### 3.4 Dynamic RAG Knowledge Layer (Pinecone + Live URL Ingestion)
Retrieval-Augmented Generation layer grounding agent decisions in real-time dispute policies without static, stale PDF files:
- **Live URL Policy Registry (`policy_sources.json`)**: Configured with live documentation endpoints (Stripe Dispute Guides, Razorpay Chargeback Policy, Visa CE 3.0 standards, Merchant Live Refund Policy URL).
- **Automated Web Loader & Chunker**: Dynamically extracts text from live URLs, slices it into semantic rule chunks (by reason code & product type), and generates vector embeddings.
- **Pinecone Vector Database**: Persistent vector store storing embeddings with metadata (`rule_id`, `reason_code`, `source_url`, `last_synced_timestamp`).
- **Zero-Stale-Rule Guarantee**: Whenever card networks update biannual mandates (April & October releases), re-syncing updates Pinecone instantly without requiring code modifications or manual PDF downloads.

### 3.5 MCP Server Layer — FastMCP (Read-Only Tools)
Model Context Protocol server providing **strictly read-only** context tools to the LLM agent:
- `get_transaction(dispute_id)`: Retrieves payment method (e.g., UPI, Credit Card), AVS match, CVV match, UPI VPA match, and digital footprint (IP distance from billing zip).
- `get_delivery_proof(order_id)`: Fetches carrier tracking status, GPS drop-off match, and signature confirmation status.
- `check_communication_logs(order_id)`: Queries customer service email and live chat transcripts.

### 3.6 Data Layer — SQLite Database
The underlying data integration and storage layer:
- **Relational Tables**: Contains synthetic and live tables (`disputes`, `transactions`, `shipping_logs`, `communication_logs`, `audit_logs`).
- **Held-out Split**: Enforces an 80/20 train/test split, holding out 20% of data for unbiased empirical evaluation.

---

## 4. How the Agent Executes Actions (Step-by-Step Flow)

```text
 1. MERCHANT TRIGGER ───► POST /api/disputes/score { dispute_id: "DISP-104" }
                                    │
                                    ▼
 2. INITIALIZATION      ───► FastAPI sends System Rubric + MCP Tool Definitions to Gemini API
                                    │
                                    ▼
 3. STEP 1 TOOL CALL    ───► Gemini invokes get_dispute_details("DISP-104")
                             • Returns: Reason: "Item Not Received", Order: "ORD-8921", Amt: $150
                                    │
                                    ▼
 4. STEP 2 TOOL CALL    ───► Gemini invokes get_delivery_proof("ORD-8921")
                             • Returns: Status: "Delivered", Carrier: "FedEx", Signed: True, GPS Match: True
                                    │
                                    ▼
 5. STEP 3 TOOL CALL    ───► Gemini invokes check_communication_logs("ORD-8921")
                             • Returns: Customer chat acknowledging package receipt on Aug 16
                                    │
                                    ▼
 6. RUBRIC EVALUATION   ───► Gemini matches evidence against rule matrix:
                             [Delivered + Signed + Chat Acknowledgment] ==> Decision: CONTEST_DISPUTE
                                    │
                                    ▼
 7. OUTPUT SYNTHESIS    ───► Gemini returns Decision: CONTEST_DISPUTE (Conf: 0.95)
                             + Drafted Chargeback Evidence Response Letter
                                    │
                                    ▼
 8. AUDIT LOGGING       ───► FastAPI writes full audit entry (ID, tools called, prompt, decision)
                             to Audit Trail Log DB/JSON
                                    │
                                    ▼
 9. UI PRESENTATION     ───► Streamlit renders dispute score, confidence meter, formal response letter,
                             and tool audit execution history
```

### Detailed Agent Action Breakdown:

1. **Trigger & Payload Reception**:
   - The merchant views the pending dispute queue (Data Table) on the Streamlit dashboard and clicks **"Analyze"** on a specific row (e.g., John Doe's case).
   - Streamlit grabs the hidden `dispute_id` for that row (e.g. `DISP-104`) and dispatches an HTTP POST request to `FastAPI`: `/api/disputes/score`.

2. **Context Setup & Tool Registration**:
   - FastAPI loads the **System Rubric System Prompt** (defining Visa/Mastercard dispute rules).
   - FastAPI formats the FastMCP tool contracts (`get_dispute_details`, `get_delivery_proof`, `get_customer_history`, `check_communication_logs`) into Gemini function-calling JSON schemas and initializes the request to `Google Gemini API` (`gemini-2.5-flash`).

3. **Autonomous Dynamic Tool Execution Loop**:
   - **Action 1 — Get Dispute Details**: Gemini identifies it needs dispute context and emits a function call `get_dispute_details(dispute_id="DISP-104")`. FastAPI executes the read-only SQL query via FastMCP and returns the dispute reason code (`Item Not Received`).
   - **Action 2 — Retrieve Carrier Evidence**: Seeing the reason is "Item Not Received", Gemini calls `get_delivery_proof(order_id="ORD-8921")`. FastMCP queries `shipping_logs` and returns carrier tracking, GPS drop-off data, and signature proof image URLs.
   - **Action 3 — Check Customer History & Chat Logs**: Gemini calls `check_communication_logs(order_id="ORD-8921")` to verify whether the customer contacted support prior to filing the dispute.

4. **Evidence Evaluation & Rubric Matching**:
   - Gemini evaluates the gathered evidence against the merchant defense rubric:
     - *Rubric Rule*: If `carrier_status == "Delivered"` AND `signature_attached == True` AND `gps_matched == True` $\rightarrow$ **Action**: `CONTEST_DISPUTE` (High Confidence).
     - *Rubric Rule*: If `carrier_status == "Lost"` $\rightarrow$ **Action**: `ACCEPT_DISPUTE`.

5. **Response Synthesis & Formal Package Drafting**:
   - Gemini formats a structured JSON response containing:
     - `decision`: `"CONTEST_DISPUTE"`
     - `confidence_score`: `0.95`
     - `reasoning_summary`: *"Item was delivered on Aug 15 with valid signature and matching GPS coordinates. Customer also confirmed delivery via support chat on Aug 16."*
     - `dispute_letter_markdown`: Formatted formal defense packet addressing the credit card issuer.

6. **Audit Trail Logging & UI Rendering**:
   - **Zero Database Write Mutation**: The agent never modifies payment records or databases.
   - FastAPI logs the complete audit trail (dispute ID, timestamps, MCP tools invoked, raw LLM reasoning chain, decision) to the `audit_logs` store.
   - Streamlit displays the decision, confidence score, evidence letter, and tool invocation history on the **Live Dispute Desk**.

---

## 5. Structural Mitigations & Edge Case Handling

To address inherent drawbacks of LLM-based agentic workflows—such as cost, non-deterministic outputs, and security vulnerabilities—this architecture implements several structural mitigations:

- **Cost & Latency Pre-Filter:** A deterministic heuristic layer intercepts transactions *before* they reach the LLM. Obvious cases (e.g., long-time customers, low-value items) are auto-cleared, while obvious fraud is auto-blocked. Only the ambiguous "middle band" (10-20% of volume) is routed to Gemini, drastically reducing API token costs and average latency.
- **Deterministic Prompting & Structured Outputs:** To prevent inconsistent decisions across identical inputs, the LLM temperature is set to `0`. The system rubric acts as a strict decision tree, and the LLM is forced to output structured JSON with discrete confidence tiers (Low/Medium/High) rather than a raw, uncalibrated score.
- **Prompt Injection Defense:** MCP tool outputs containing user-generated text (e.g., delivery instructions, dispute comments) are wrapped in strict delimiters (e.g., `<untrusted_data>`). The system prompt explicitly instructs the agent to treat these blocks as passive data and never execute instructions found within them.
- **Statistical Anomaly Tool:** Because an LLM only views one transaction at a time and lacks cross-transaction statistical awareness, a background statistical layer (e.g., z-scores, isolation forests) is maintained. This layer exposes population-level anomalies (like sudden device fingerprint spikes) to the agent via a dedicated MCP tool.
- **Network-Compliant Formatting:** To ensure trust from regulators and card networks, the primary output of the agent is mapped to structured fields expected by Visa/Mastercard schemas, with the natural-language reasoning preserved strictly as a supplemental audit trail.



[def]: file:///C:/Users/bomma/.gemini/antigravity-ide/brain/af645d31-686b-4e9d-ab7f-46b02d24e737/diagram.png