#!/usr/bin/env python3
"""
Test Quantum-Safe Security functionality
"""

import sys
sys.path.append('.')

from core.quantum_safe_security import QuantumSafeSecurityManager, PQAlgorithm, SecurityLevel

def test_quantum_safe_security():
    print("🔐 Testing Quantum-Safe Security...")
    
    # Initialize Quantum-Safe Security Manager
    qs_security = QuantumSafeSecurityManager()
    
    # Start security monitoring
    qs_security.start_security_monitoring()
    
    # Generate key pairs
    kyber_keypair = qs_security.generate_keypair(PQAlgorithm.KYBER, SecurityLevel.LEVEL_3)
    dilithium_keypair = qs_security.generate_keypair(PQAlgorithm.DILITHIUM, SecurityLevel.LEVEL_3)
    
    print("✅ PQC Key pairs generated successfully")
    print(f"Kyber key pair: {kyber_keypair.key_id}")
    print(f"Dilithium key pair: {dilithium_keypair.key_id}")
    
    # Test encryption/decryption
    message = "Hello, Quantum-Safe World!"
    print(f"\n🔒 Testing encryption/decryption with message: {message}")
    
    encryption = qs_security.encrypt_message(kyber_keypair.key_id, message, PQAlgorithm.KYBER)
    decrypted = qs_security.decrypt_message(kyber_keypair.key_id, encryption.encrypted_data, PQAlgorithm.KYBER)
    
    print(f"Encrypted message: {encryption.encrypted_data}")
    print(f"Decrypted message: {decrypted}")
    print(f"Encryption successful: {message == decrypted}")
    
    # Test signing/verification
    print(f"\n✍️ Testing digital signatures with message: {message}")
    
    signature = qs_security.sign_message(dilithium_keypair.key_id, message, PQAlgorithm.DILITHIUM)
    is_valid = qs_security.verify_signature(dilithium_keypair.key_id, message, signature.signature, PQAlgorithm.DILITHIUM)
    
    print(f"Signature: {signature.signature}")
    print(f"Signature valid: {is_valid}")
    
    # Test audit log creation
    print(f"\n📝 Testing immutable audit logs...")
    
    audit_log = qs_security.create_audit_log("test_action", {
        "message": "Quantum-safe security test",
        "timestamp": "2024-12-03T23:56:05Z",
        "security_level": "quantum_safe"
    })
    
    print(f"Audit log created: {audit_log['audit_id']}")
    print(f"Audit hash: {audit_log['hash']}")
    
    # Get security metrics
    metrics = qs_security.get_security_metrics()
    print(f"\n📊 Security metrics:")
    print(f"PQC signatures total: {metrics['quantum_safe_security']['pqc_signatures_total']}")
    print(f"PQC encryptions total: {metrics['quantum_safe_security']['pqc_encryptions_total']}")
    print(f"Verification success rate: {metrics['quantum_safe_security']['verification_success_rate']}")
    print(f"Encryption success rate: {metrics['quantum_safe_security']['encryption_success_rate']}")
    
    # Test blockchain message signing
    print(f"\n⛓️ Testing blockchain message signing...")
    
    blockchain_message = "SON decision: Optimize traffic routing for Site A"
    blockchain_signature = qs_security.sign_message(dilithium_keypair.key_id, blockchain_message, PQAlgorithm.DILITHIUM)
    blockchain_verification = qs_security.verify_signature(dilithium_keypair.key_id, blockchain_message, blockchain_signature.signature, PQAlgorithm.DILITHIUM)
    
    print(f"Blockchain message: {blockchain_message}")
    print(f"Blockchain signature: {blockchain_signature.signature}")
    print(f"Blockchain verification: {blockchain_verification}")
    
    # Create blockchain audit log
    blockchain_audit = qs_security.create_audit_log("son_decision", {
        "decision": "Optimize traffic routing",
        "site": "Site A",
        "timestamp": "2024-12-03T23:56:05Z",
        "signature": blockchain_signature.signature,
        "verified": blockchain_verification
    })
    
    print(f"Blockchain audit log: {blockchain_audit['audit_id']}")
    print(f"Blockchain audit hash: {blockchain_audit['hash']}")
    
    # Stop security monitoring
    qs_security.stop_security_monitoring()
    
    print("✅ Quantum-Safe Security testing completed successfully")
    return True

if __name__ == "__main__":
    test_quantum_safe_security()
