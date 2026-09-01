"""
generate_data.py — Synthetic data engine for AI Risk Manager.

Generates 60 realistic Indian e-commerce transactions with correlated
shipping, communication, and dispute records. Ground-truth labels are
assigned using a deterministic rubric so the evaluation layer can
measure precision/recall honestly.

Cost assumptions (from real-world Stripe India / Razorpay 2024 data):
  - Average order value: ₹1,500–₹5,000 range
  - Dispute fee per chargeback: ₹1,000 (Stripe India standard)

Usage:
    python generate_data.py
"""

import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

import os
import sys

# Ensure src directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database import init_db, get_connection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_RECORDS = 60
TEST_SET_RATIO = 0.20  # 80/20 split
RANDOM_SEED = 42

fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Indian carrier names for realism
CARRIERS = ["BlueDart", "Delhivery", "DTDC", "Ecom Express", "India Post"]

# Visa/Mastercard reason codes relevant to Indian e-commerce
REASON_CODES = [
    "product_not_received",       # Visa 13.1 / MC 4855
    "not_as_described",           # Visa 13.3 / MC 4853
    "unauthorized_transaction",   # Visa 10.4 / MC 4837
    "duplicate_charge",           # Visa 12.6 / MC 4834
    "credit_not_processed",       # Visa 13.6 / MC 4860
]

# Chat transcript templates — realistic Indian English
INDIAN_LOCATIONS = [
    {"city": "Bengaluru", "region": "Karnataka", "lat": 12.9352, "lng": 77.6245, "landmark": "Koramangala 4th Block, Bengaluru, KA (560034)"},
    {"city": "Mumbai", "region": "Maharashtra", "lat": 19.0760, "lng": 72.8777, "landmark": "Bandra Kurla Complex, Mumbai, MH (400051)"},
    {"city": "New Delhi", "region": "Delhi", "lat": 28.6139, "lng": 77.2090, "landmark": "Connaught Place, New Delhi, DL (110001)"},
    {"city": "Hyderabad", "region": "Telangana", "lat": 17.4435, "lng": 78.3772, "landmark": "Hitec City, Madhapur, Hyderabad, TS (500081)"},
    {"city": "Chennai", "region": "Tamil Nadu", "lat": 13.0827, "lng": 80.2707, "landmark": "T. Nagar, Chennai, TN (600017)"},
    {"city": "Pune", "region": "Maharashtra", "lat": 18.5204, "lng": 73.8567, "landmark": "Viman Nagar, Pune, MH (411014)"},
    {"city": "Kolkata", "region": "West Bengal", "lat": 22.5726, "lng": 88.3639, "landmark": "Salt Lake Sector V, Kolkata, WB (700091)"},
    {"city": "Ahmedabad", "region": "Gujarat", "lat": 23.0225, "lng": 72.5714, "landmark": "SG Highway, Ahmedabad, GJ (380015)"},
]

FRIENDLY_FRAUD_CHATS = [
    "Customer: I never received my order.\nAgent: Our records show it was delivered on {date} and signed by '{name}'.\nCustomer: That's not me. I want a refund.",
    "Customer: This is not what I ordered at all.\nAgent: Could you share a photo of the item received?\nCustomer: I don't have it anymore, I threw it away.\nAgent: Without evidence we cannot process this further.",
    "Customer: I didn't make this purchase. Someone used my card.\nAgent: We see the order was placed from your registered device and IP.\nCustomer: It must have been hacked.",
]

LEGITIMATE_DISPUTE_CHATS = [
    "Customer: My package shows delivered but I never got it.\nAgent: We're sorry to hear that. We'll investigate with the carrier.\nCustomer: Please do, I've been waiting for 2 weeks.",
    "Customer: The product arrived damaged — screen is cracked.\nAgent: We apologize. Could you share a photo?\nCustomer: Sure, attaching now.\nAgent: Thank you, we'll process a replacement.",
    "Customer: I was charged twice for the same order.\nAgent: Let us check... You're right, we see a duplicate. We'll reverse one charge immediately.",
]


def _generate_order_id() -> str:
    """Generate an order ID in a format resembling Indian payment gateways."""
    return f"ORD-{uuid.uuid4().hex[:8].upper()}"


def _generate_customer_id() -> str:
    return f"CUST-{uuid.uuid4().hex[:6].upper()}"


def _generate_dispute_id() -> str:
    return f"DSP-{uuid.uuid4().hex[:8].upper()}"


def _random_indian_ip() -> str:
    """Generate a random IP in common Indian ISP ranges."""
    first_octet = random.choice([49, 59, 103, 106, 122, 157, 182, 203])
    return f"{first_octet}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def generate_records() -> list[dict]:
    """
    Generate synthetic records with correlated evidence signals.

    Ground-truth labeling rubric (deterministic, defense-only):
      A dispute is labeled 'valid_defense' (merchant can win) when:
        - Delivery was confirmed (carrier_status == 'delivered')
        - AND at least one strong signal: GPS match, signature, AVS match, or
          chat transcript shows friendly fraud pattern.
      Otherwise it is labeled 'lost_cause' (merchant should accept the loss).
    """
    records = []
    num_test = int(NUM_RECORDS * TEST_SET_RATIO)
    test_indices = set(random.sample(range(NUM_RECORDS), num_test))

    for i in range(NUM_RECORDS):
        order_id = _generate_order_id()
        customer_id = _generate_customer_id()
        dispute_id = _generate_dispute_id()

        # Choose primary customer location
        cust_loc = random.choice(INDIAN_LOCATIONS)

        # -- Transaction ---------------------------------------------------
        payment_method = random.choice(["UPI", "CARD"])
        amount = round(random.uniform(500, 8000), 2)  # ₹500–₹8,000 range
        created_at = fake.date_time_between(
            start_date="-90d", end_date="-2d"
        ).isoformat()

        # Evidence signals — probabilities tuned for ~55% valid_defense rate
        avs_match = 1 if random.random() < 0.65 else 0
        cvv_match = 1 if random.random() < 0.70 else 0
        upi_vpa_match = 1 if (payment_method == "UPI" and random.random() < 0.60) else 0
        ip_address = _random_indian_ip()
        ip_city = cust_loc["city"]
        ip_distance_km = round(random.uniform(0.5, 12.0), 1) if avs_match == 1 else round(random.uniform(150, 1800), 1)

        # -- Shipping -------------------------------------------------------
        # Delivered cases get stronger defense evidence
        is_delivered = random.random() < 0.70
        carrier_status = "delivered" if is_delivered else random.choice(
            ["in_transit", "returned", "lost"]
        )
        gps_match = 1 if (is_delivered and random.random() < 0.75) else 0
        signature_obtained = 1 if (is_delivered and random.random() < 0.60) else 0
        delivered_at = (
            (datetime.fromisoformat(created_at) + timedelta(days=random.randint(2, 7))).isoformat()
            if is_delivered else None
        )

        if is_delivered and gps_match == 1:
            dropoff_lat = round(float(cust_loc["lat"]) + random.uniform(-0.0003, 0.0003), 5)
            dropoff_lng = round(float(cust_loc["lng"]) + random.uniform(-0.0003, 0.0003), 5)
            dropoff_location = cust_loc["landmark"]
            gps_accuracy_meters = round(random.uniform(3.5, 8.5), 1)
        elif is_delivered:
            # Mismatched location
            dropoff_lat = round(float(cust_loc["lat"]) + random.uniform(0.12, 0.35), 5)
            dropoff_lng = round(float(cust_loc["lng"]) + random.uniform(0.12, 0.35), 5)
            dropoff_location = f"Out of Area Hub ({cust_loc['city']} Outer Ring)"
            gps_accuracy_meters = round(random.uniform(45.0, 120.0), 1)
        else:
            dropoff_lat = None
            dropoff_lng = None
            dropoff_location = None
            gps_accuracy_meters = None

        # -- Communication --------------------------------------------------
        reason_code = random.choice(REASON_CODES)

        if is_delivered and random.random() < 0.50:
            # Friendly fraud pattern — customer contradicts delivery evidence
            chat = random.choice(FRIENDLY_FRAUD_CHATS).format(
                date=delivered_at[:10] if delivered_at else "N/A",
                name=fake.first_name()
            )
            has_friendly_fraud_signal = True
        else:
            chat = random.choice(LEGITIMATE_DISPUTE_CHATS)
            has_friendly_fraud_signal = False

        # -- Ground-truth label (deterministic rubric) ----------------------
        strong_signals = sum([
            gps_match == 1,
            signature_obtained == 1,
            avs_match == 1 and cvv_match == 1,
            has_friendly_fraud_signal,
        ])

        if is_delivered and strong_signals >= 1:
            ground_truth = "valid_defense"
        else:
            ground_truth = "lost_cause"

        is_test = 1 if i in test_indices else 0

        records.append({
            # Transaction
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "currency": "INR",
            "payment_method": payment_method,
            "avs_match": avs_match,
            "cvv_match": cvv_match,
            "upi_vpa_match": upi_vpa_match,
            "ip_address": ip_address,
            "ip_city": ip_city,
            "ip_distance_km": ip_distance_km,
            "txn_created_at": created_at,
            # Shipping
            "carrier": random.choice(CARRIERS),
            "carrier_status": carrier_status,
            "tracking_number": fake.bothify("??########").upper(),
            "gps_match": gps_match,
            "dropoff_lat": dropoff_lat,
            "dropoff_lng": dropoff_lng,
            "dropoff_location": dropoff_location,
            "gps_accuracy_meters": gps_accuracy_meters,
            "signature_obtained": signature_obtained,
            "delivered_at": delivered_at,
            # Communication
            "chat_transcript": chat,
            "chat_created_at": (
                datetime.fromisoformat(created_at) + timedelta(days=random.randint(8, 15))
            ).isoformat(),
            # Dispute
            "dispute_id": dispute_id,
            "reason_code": reason_code,
            "disputed_amount": amount,  # Full amount disputed
            "ground_truth_label": ground_truth,
            "is_test_set": is_test,
            "dispute_created_at": (
                datetime.fromisoformat(created_at) + timedelta(days=random.randint(10, 30))
            ).isoformat(),
        })

    return records


def insert_records(records: list[dict]) -> None:
    """Insert generated records into the SQLite database."""
    conn = get_connection()
    cursor = conn.cursor()

    for r in records:
        # transactions
        cursor.execute("""
            INSERT OR IGNORE INTO transactions
                (order_id, customer_id, amount, currency, payment_method,
                 avs_match, cvv_match, upi_vpa_match, ip_address, ip_city, ip_distance_km, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["order_id"], r["customer_id"], r["amount"], r["currency"],
            r["payment_method"], r["avs_match"], r["cvv_match"],
            r["upi_vpa_match"], r["ip_address"], r["ip_city"], r["ip_distance_km"],
            r["txn_created_at"],
        ))

        # shipping_logs
        cursor.execute("""
            INSERT OR IGNORE INTO shipping_logs
                (order_id, carrier, carrier_status, tracking_number,
                 gps_match, dropoff_lat, dropoff_lng, dropoff_location, gps_accuracy_meters,
                 signature_obtained, delivered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["order_id"], r["carrier"], r["carrier_status"],
            r["tracking_number"], r["gps_match"],
            r["dropoff_lat"], r["dropoff_lng"], r["dropoff_location"], r["gps_accuracy_meters"],
            r["signature_obtained"], r["delivered_at"],
        ))

        # communication_logs
        cursor.execute("""
            INSERT OR IGNORE INTO communication_logs
                (order_id, channel, chat_transcript, created_at)
            VALUES (?, 'chat', ?, ?)
        """, (r["order_id"], r["chat_transcript"], r["chat_created_at"]))

        # disputes
        cursor.execute("""
            INSERT OR IGNORE INTO disputes
                (dispute_id, order_id, reason_code, disputed_amount,
                 ground_truth_label, is_test_set, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            r["dispute_id"], r["order_id"], r["reason_code"],
            r["disputed_amount"], r["ground_truth_label"],
            r["is_test_set"], r["dispute_created_at"],
        ))

    conn.commit()
    conn.close()


def print_summary(records: list[dict]) -> None:
    """Print a summary of the generated data."""
    df = pd.DataFrame(records)
    total = len(df)
    train = len(df[df["is_test_set"] == 0])
    test = len(df[df["is_test_set"] == 1])
    valid = len(df[df["ground_truth_label"] == "valid_defense"])
    lost = len(df[df["ground_truth_label"] == "lost_cause"])

    print("\n" + "=" * 60)
    print("  AI RISK MANAGER — Synthetic Data Summary")
    print("=" * 60)
    print(f"  Total records generated : {total}")
    print(f"  Training set            : {train} ({train/total*100:.0f}%)")
    print(f"  Held-out test set       : {test} ({test/total*100:.0f}%)")
    print(f"  -------------------------------------")
    print(f"  valid_defense (merchant wins) : {valid} ({valid/total*100:.1f}%)")
    print(f"  lost_cause (merchant loses)   : {lost} ({lost/total*100:.1f}%)")
    print(f"  -------------------------------------")
    print(f"  Avg disputed amount     : INR {df['disputed_amount'].mean():,.2f}")
    print(f"  Total amount at risk    : INR {df['disputed_amount'].sum():,.2f}")
    print("=" * 60 + "\n")


def main():
    print("[generate_data] Initializing database schema...")
    init_db(drop_existing=True)

    print(f"[generate_data] Generating {NUM_RECORDS} synthetic records...")
    records = generate_records()

    print("[generate_data] Inserting into SQLite database...")
    insert_records(records)

    print_summary(records)
    print("[generate_data] Done! Database ready at risk_manager.db")


if __name__ == "__main__":
    main()
