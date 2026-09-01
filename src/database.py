"""
database.py — SQLite schema definition and connection helper for AI Risk Manager.

Tables:
  - transactions: Payment details (UPI/Card), AVS, CVV, IP metadata.
  - shipping_logs: Carrier delivery status, GPS match, signature.
  - communication_logs: Customer support chat transcripts per order.
  - disputes: Chargeback dispute records with ground-truth labels.
  - audit_logs: Every AI decision logged for full audit trail.
"""

import sqlite3
import os

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "risk_manager.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the SQLite database with row-factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str = DB_PATH, drop_existing: bool = False) -> None:
    """
    Initialize SQLite database and create all required tables if they don't exist.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if drop_existing:
        cursor.executescript("""
            DROP TABLE IF EXISTS audit_logs;
            DROP TABLE IF EXISTS disputes;
            DROP TABLE IF EXISTS communication_logs;
            DROP TABLE IF EXISTS shipping_logs;
            DROP TABLE IF EXISTS transactions;
        """)

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id        TEXT    UNIQUE NOT NULL,
            customer_id     TEXT    NOT NULL,
            amount          REAL    NOT NULL,
            currency        TEXT    NOT NULL DEFAULT 'INR',
            payment_method  TEXT    NOT NULL,          -- 'UPI' or 'CARD'
            avs_match       INTEGER NOT NULL DEFAULT 0, -- 1 = match, 0 = mismatch
            cvv_match       INTEGER NOT NULL DEFAULT 0,
            upi_vpa_match   INTEGER NOT NULL DEFAULT 0,
            ip_address      TEXT,
            ip_city         TEXT,              -- e.g. 'Bengaluru', 'Mumbai', 'Delhi'
            ip_distance_km  REAL    DEFAULT 0.0, -- km between billing and IP geo
            created_at      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS shipping_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id            TEXT    UNIQUE NOT NULL,
            carrier             TEXT    NOT NULL,
            carrier_status      TEXT    NOT NULL,       -- 'delivered', 'in_transit', 'returned', 'lost'
            tracking_number     TEXT,
            gps_match           INTEGER NOT NULL DEFAULT 0, -- 1 = GPS drop matches billing addr
            dropoff_lat         REAL,           -- Carrier drop-off GPS latitude
            dropoff_lng         REAL,           -- Carrier drop-off GPS longitude
            dropoff_location    TEXT,           -- Reverse-geocoded landmark/address
            gps_accuracy_meters REAL    DEFAULT 10.0,   -- Carrier GPS horizontal accuracy (m)
            signature_obtained  INTEGER NOT NULL DEFAULT 0,
            delivered_at        TEXT,
            FOREIGN KEY (order_id) REFERENCES transactions(order_id)
        );

        CREATE TABLE IF NOT EXISTS communication_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id        TEXT    NOT NULL,
            channel         TEXT    NOT NULL DEFAULT 'chat', -- 'chat', 'email', 'phone'
            chat_transcript TEXT    NOT NULL,
            created_at      TEXT    NOT NULL,
            FOREIGN KEY (order_id) REFERENCES transactions(order_id)
        );

        CREATE TABLE IF NOT EXISTS disputes (
            dispute_id          TEXT    PRIMARY KEY,
            order_id            TEXT    NOT NULL,
            reason_code         TEXT    NOT NULL,       -- e.g. 'product_not_received', 'not_as_described'
            disputed_amount     REAL    NOT NULL,
            ground_truth_label  TEXT    NOT NULL,       -- 'valid_defense' or 'lost_cause'
            is_test_set         INTEGER NOT NULL DEFAULT 0,  -- 1 = held-out test record
            status              TEXT    NOT NULL DEFAULT 'pending', -- 'pending', 'scored'
            customer_evidence_notes TEXT DEFAULT NULL,  -- Counter-evidence provided by customer (police FIR, photos, etc.)
            created_at          TEXT    NOT NULL,
            FOREIGN KEY (order_id) REFERENCES transactions(order_id)
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            dispute_id              TEXT    NOT NULL,
            decision                TEXT    NOT NULL,           -- 'CONTEST_DISPUTE', 'ACCEPT_DISPUTE', or 'PARTIAL_CONTEST'
            confidence              REAL   NOT NULL,
            tools_called            TEXT,                       -- JSON list of MCP tools invoked
            reasoning               TEXT,                       -- LLM reasoning summary
            evidence_letter         TEXT,                       -- Drafted dispute response letter
            customer_notification   TEXT   DEFAULT NULL,        -- Polite notification draft for customer transparency
            contested_amount        REAL   DEFAULT NULL,        -- Amount merchant is contesting (for PARTIAL_CONTEST)
            conceded_amount         REAL   DEFAULT NULL,        -- Amount merchant concedes (for PARTIAL_CONTEST)
            fairness_flags          TEXT   DEFAULT NULL,        -- JSON flags: confidence_downgrade, sarcasm_detected, etc.
            created_at              TEXT    NOT NULL,
            FOREIGN KEY (dispute_id) REFERENCES disputes(dispute_id)
        );
    """)

    conn.commit()
    conn.close()
    print(f"[database] Initialized schema at {DB_PATH}")


if __name__ == "__main__":
    init_db()
