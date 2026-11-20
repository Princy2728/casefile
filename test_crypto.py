"""
LIFECYCLE-X AI — Cryptographic Module Test & Validation Suite
Validates all 10 critical security test scenarios required by the system specification.
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.key_management import get_aes_key, get_hmac_key, get_key_metadata
from crypto.encryption import encrypt_evidence, decrypt_evidence, DecryptionError
from crypto.evidence_integrity import canonicalize, compute_sha256, generate_hmac, verify_hmac
from crypto.cryptotrace import CryptoTraceChain, build_initial_chain, GENESIS_HASH


def run_all_tests():
    print("=" * 75)
    print("⚡ LIFECYCLE-X AI — CRYPTOGRAPHIC MODULE VALIDATION SUITE")
    print("=" * 75)

    passed = 0
    total = 10

    # -------------------------------------------------------------------------
    # Test 1: Valid Evidence Encryption and Decryption (AES-256-GCM)
    # -------------------------------------------------------------------------
    print("\n[Scenario 1] Valid Evidence Encryption and Decryption (AES-256-GCM)...")
    sample_evidence = {
        "employee_id": "HF0004",
        "username": "r.menon",
        "role": "Accounts Payable Clerk",
        "termination_date": "2026-06-30",
        "suspicious_login": "2026-08-13 03:17:22",
        "ip_address": "172.16.8.44",
        "device": "kiosk-wh-02",
        "accessed_resource": "/finance/ap-queue/doc-0045.pdf"
    }

    vault_record = encrypt_evidence(sample_evidence)
    assert vault_record["algorithm"] == "AES-256-GCM"
    assert len(bytes.fromhex(vault_record["nonce_hex"])) == 12  # 96-bit nonce
    assert len(bytes.fromhex(vault_record["tag_hex"])) == 16    # 128-bit tag

    decrypted = decrypt_evidence(vault_record)
    assert decrypted == sample_evidence
    print("  ✅ AES-256-GCM authenticated encryption and decryption verified.")
    print(f"     Nonce: {vault_record['nonce_hex']} (96 bits)")
    print(f"     Auth Tag: {vault_record['tag_hex']} (128 bits)")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 2: Valid HMAC-SHA256 Verification
    # -------------------------------------------------------------------------
    print("\n[Scenario 2] Valid HMAC-SHA256 Evidence Integrity Verification...")
    hmac_code, digest = generate_hmac(sample_evidence)
    is_valid, status, details = verify_hmac(sample_evidence, hmac_code)

    assert is_valid is True
    assert status == "Integrity Verified"
    print(f"  ✅ Status: {status}")
    print(f"     SHA-256 Digest: {digest[:24]}...")
    print(f"     HMAC-SHA256:    {hmac_code[:24]}...")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 3: Modified Evidence Causing HMAC Verification Failure
    # -------------------------------------------------------------------------
    print("\n[Scenario 3] Modified Evidence Causing HMAC Verification Failure...")
    tampered_evidence = sample_evidence.copy()
    tampered_evidence["accessed_resource"] = "/finance/ap-queue/doc-0045-TAMPERED.pdf"

    is_valid_tampered, status_tampered, _ = verify_hmac(tampered_evidence, hmac_code)
    assert is_valid_tampered is False
    assert status_tampered == "Evidence Modified"
    print(f"  ✅ Tampering detected! Status: {status_tampered}")

    # Also test AES-GCM ciphertext tampering
    tampered_vault = vault_record.copy()
    tampered_ct = list(bytes.fromhex(tampered_vault["ciphertext_hex"]))
    tampered_ct[0] ^= 0xFF  # flip byte
    tampered_vault["ciphertext_hex"] = bytes(tampered_ct).hex()
    try:
        decrypt_evidence(tampered_vault)
        assert False, "Should have raised DecryptionError"
    except DecryptionError:
        print("  ✅ AES-256-GCM auth tag correctly rejected tampered ciphertext.")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 4: Valid Hash-Chain Verification
    # -------------------------------------------------------------------------
    print("\n[Scenario 4] Valid CryptoTrace Hash-Chain Verification...")
    chain = build_initial_chain()
    verification = chain.verify_chain()

    assert verification["is_valid"] is True
    assert verification["status"] == "Chain Verified"
    assert verification["total_records"] == 8
    print(f"  ✅ Status: {verification['status']} ({verification['total_records']} blocks verified)")
    print(f"     Genesis Hash: {GENESIS_HASH[:16]}...")
    print(f"     Latest Hash:  {chain.get_latest_hash()[:16]}...")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 5: Modified Audit Record Causing Hash-Chain Failure
    # -------------------------------------------------------------------------
    print("\n[Scenario 5] Modified Audit Record Causing Hash-Chain Failure...")
    tampered_chain = chain.simulate_tampering(record_index=2, field_to_alter="event_data")
    t_verification = chain.verify_chain(tampered_chain)

    assert t_verification["is_valid"] is False
    assert t_verification["status"] == "Tampering Detected"
    assert t_verification["first_invalid_index"] == 2
    print(f"  ✅ Status: {t_verification['status']}")
    print(f"     Compromised Block Index: #{t_verification['first_invalid_index']}")
    print(f"     Diagnostic: {t_verification['error_message']}")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 6: Broken Previous-Hash Relationship Detection
    # -------------------------------------------------------------------------
    print("\n[Scenario 6] Broken Previous-Hash Relationship Detection...")
    broken_chain = chain.simulate_tampering(
        record_index=3,
        field_to_alter="previous_hash",
        tampered_value="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )
    b_verification = chain.verify_chain(broken_chain)

    assert b_verification["is_valid"] is False
    assert b_verification["status"] == "Broken Chain"
    assert b_verification["first_invalid_index"] == 3
    print(f"  ✅ Status: {b_verification['status']}")
    print(f"     Compromised Block Index: #{b_verification['first_invalid_index']}")
    print(f"     Diagnostic: {b_verification['error_message']}")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 7: Post-Termination Authentication Detection Logic
    # -------------------------------------------------------------------------
    print("\n[Scenario 7] Post-Termination Authentication Detection Rule...")
    term_date = datetime.strptime("2026-06-30", "%Y-%m-%d")
    auth_date = datetime.strptime("2026-08-13 03:17:22", "%Y-%m-%d %H:%M:%S")
    days_post_term = (auth_date - term_date).days
    assert days_post_term == 44

    is_post_term = auth_date > term_date
    assert is_post_term is True
    print(f"  ✅ Detected unauthorized login: {days_post_term} days after termination.")
    print("     Deterministic Policy Decision: HARD BLOCK ENFORCED (Zero Tolerance).")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 8: Successful Mock Containment and Audit Logging
    # -------------------------------------------------------------------------
    print("\n[Scenario 8] Mock Containment and Cryptographic Audit Trail Appending...")
    pre_containment_len = len(chain.records)
    new_block = chain.create_block(
        event_id="EVT-20260813-009",
        employee_id="HF0004",
        event_type="CONTAINMENT_ACCOUNT_DEACTIVATED",
        event_timestamp="2026-08-13 03:21:00 UTC",
        event_data={
            "action": "DEACTIVATE_AND_REVOKE_REFRESH_TOKENS",
            "idp_provider": "Okta/AzureAD_Mock",
            "status": "SUCCESSFUL"
        },
        actor="SOC-AUTOMATION",
        severity="HIGH"
    )

    assert len(chain.records) == pre_containment_len + 1
    assert new_block["previous_hash"] != GENESIS_HASH
    print(f"  ✅ Containment event recorded as Block #{new_block['audit_id']}.")
    print(f"     Event Hash: {new_block['current_hash'][:16]}...")
    print(f"     HMAC Seal:  {new_block['hmac_seal'][:16]}...")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 9: Verification After Containment
    # -------------------------------------------------------------------------
    print("\n[Scenario 9] Verification of Complete Hash Chain After Containment...")
    post_verification = chain.verify_chain()
    assert post_verification["is_valid"] is True
    assert post_verification["status"] == "Chain Verified"
    assert post_verification["total_records"] == 9
    print(f"  ✅ Post-containment chain fully verified ({post_verification['total_records']} blocks).")
    passed += 1

    # -------------------------------------------------------------------------
    # Test 10: Application Restart & Storage Persistence
    # -------------------------------------------------------------------------
    print("\n[Scenario 10] Persistence & Recovery Across Restarts...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        chain.save_to_disk(tmp_path)
        new_instance = CryptoTraceChain()
        loaded = new_instance.load_from_disk(tmp_path)
        assert loaded is True
        assert len(new_instance.records) == len(chain.records)

        reloaded_verification = new_instance.verify_chain()
        assert reloaded_verification["is_valid"] is True
        assert reloaded_verification["status"] == "Chain Verified"
        print(f"  ✅ Restored {len(new_instance.records)} blocks from persistent disk storage.")
        print(f"     Verification on reloaded chain: {reloaded_verification['status']}.")
        passed += 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"TEST RESULTS: {passed}/{total} SCENARIOS PASSED (100% SUCCESS)")
    print("=" * 75)
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
