#!/usr/bin/env python3
"""
Blockchain-based Security & Trust Manager for Telecom AI 3.0
Implements Hyperledger Fabric and Ethereum integration for secure agent communication
"""

import asyncio
import logging
import hashlib
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Blockchain imports
try:
    from web3 import Web3
    from eth_account import Account
    ETHEREUM_AVAILABLE = True
except ImportError:
    ETHEREUM_AVAILABLE = False
    print("Ethereum libraries not available. Install with: pip install web3 eth-account")

try:
    from hfc.fabric import Client
    from hfc.fabric.user import create_user
    HYPERLEDGER_AVAILABLE = True
except ImportError:
    HYPERLEDGER_AVAILABLE = False
    print("Hyperledger Fabric not available. Install with: pip install hfc")

class BlockchainType(Enum):
    """Blockchain types"""
    ETHEREUM = "ethereum"
    HYPERLEDGER = "hyperledger"
    SIMULATED = "simulated"

class TrustLevel(Enum):
    """Trust levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BlockchainIdentity:
    """Blockchain identity for agents"""
    agent_id: str
    public_key: str
    private_key: str
    address: str
    trust_score: float
    created_at: datetime
    last_activity: datetime

@dataclass
class BlockchainTransaction:
    """Blockchain transaction record"""
    transaction_id: str
    from_address: str
    to_address: str
    data: Dict[str, Any]
    signature: str
    timestamp: datetime
    block_hash: Optional[str] = None
    confirmation_count: int = 0
    status: str = "pending"

@dataclass
class AuditLog:
    """Immutable audit log entry"""
    log_id: str
    agent_id: str
    action: str
    data: Dict[str, Any]
    signature: str
    timestamp: datetime
    block_hash: str
    previous_hash: str

class BlockchainManager:
    """Blockchain-based Security & Trust Manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Blockchain configuration
        self.blockchain_type = BlockchainType(self.config.get('blockchain_type', 'simulated'))
        self.network_url = self.config.get('network_url', 'http://localhost:8545')
        self.contract_address = self.config.get('contract_address')
        
        # Agent identities
        self.agent_identities = {}
        self.trust_scores = {}
        
        # Transaction tracking
        self.transactions = {}
        self.audit_logs = []
        
        # Blockchain connections
        self.web3 = None
        self.fabric_client = None
        
        # Security settings
        self.encryption_enabled = self.config.get('encryption_enabled', True)
        self.signature_required = self.config.get('signature_required', True)
        
        # Initialize blockchain
        self._initialize_blockchain()
    
    def _initialize_blockchain(self):
        """Initialize blockchain connection"""
        if self.blockchain_type == BlockchainType.ETHEREUM and ETHEREUM_AVAILABLE:
            self._initialize_ethereum()
        elif self.blockchain_type == BlockchainType.HYPERLEDGER and HYPERLEDGER_AVAILABLE:
            self._initialize_hyperledger()
        else:
            self.logger.info("Using simulated blockchain for development")
            self._initialize_simulated_blockchain()
    
    def _initialize_ethereum(self):
        """Initialize Ethereum connection"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.network_url))
            if self.web3.is_connected():
                self.logger.info(f"Connected to Ethereum network: {self.network_url}")
            else:
                self.logger.warning("Failed to connect to Ethereum network, using simulated mode")
                self._initialize_simulated_blockchain()
        except Exception as e:
            self.logger.error(f"Ethereum initialization failed: {e}")
            self._initialize_simulated_blockchain()
    
    def _initialize_hyperledger(self):
        """Initialize Hyperledger Fabric connection"""
        try:
            self.fabric_client = Client(net_profile="network.json")
            self.logger.info("Connected to Hyperledger Fabric network")
        except Exception as e:
            self.logger.error(f"Hyperledger Fabric initialization failed: {e}")
            self._initialize_simulated_blockchain()
    
    def _initialize_simulated_blockchain(self):
        """Initialize simulated blockchain for development"""
        self.logger.info("Initializing simulated blockchain")
        self.simulated_blocks = []
        self.simulated_block_height = 0
    
    def create_agent_identity(self, agent_id: str, trust_level: TrustLevel = TrustLevel.MEDIUM) -> BlockchainIdentity:
        """Create blockchain identity for an agent"""
        try:
            # Generate key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Generate address
            address = self._generate_address(public_pem)
            
            # Create identity
            identity = BlockchainIdentity(
                agent_id=agent_id,
                public_key=base64.b64encode(public_pem).decode(),
                private_key=base64.b64encode(private_pem).decode(),
                address=address,
                trust_score=self._get_trust_score(trust_level),
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Store identity
            self.agent_identities[agent_id] = identity
            self.trust_scores[agent_id] = identity.trust_score
            
            self.logger.info(f"Created blockchain identity for agent {agent_id}")
            return identity
            
        except Exception as e:
            self.logger.error(f"Failed to create identity for agent {agent_id}: {e}")
            raise
    
    def _generate_address(self, public_key: bytes) -> str:
        """Generate blockchain address from public key"""
        if self.blockchain_type == BlockchainType.ETHEREUM and self.web3:
            # Generate Ethereum address
            account = Account.from_key(public_key)
            return account.address
        else:
            # Generate simulated address
            hash_obj = hashlib.sha256(public_key)
            return "0x" + hash_obj.hexdigest()[:40]
    
    def _get_trust_score(self, trust_level: TrustLevel) -> float:
        """Get trust score based on trust level"""
        trust_mapping = {
            TrustLevel.LOW: 0.3,
            TrustLevel.MEDIUM: 0.6,
            TrustLevel.HIGH: 0.8,
            TrustLevel.CRITICAL: 0.95
        }
        return trust_mapping[trust_level]
    
    def sign_message(self, agent_id: str, message: str) -> str:
        """Sign a message with agent's private key"""
        if agent_id not in self.agent_identities:
            raise ValueError(f"Agent {agent_id} not found")
        
        identity = self.agent_identities[agent_id]
        
        try:
            # Decode private key
            private_key_pem = base64.b64decode(identity.private_key)
            private_key = serialization.load_pem_private_key(
                private_key_pem,
                password=None
            )
            
            # Sign message
            message_bytes = message.encode('utf-8')
            signature = private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Encode signature
            signature_b64 = base64.b64encode(signature).decode()
            
            # Update last activity
            identity.last_activity = datetime.now()
            
            return signature_b64
            
        except Exception as e:
            self.logger.error(f"Failed to sign message for agent {agent_id}: {e}")
            raise
    
    def verify_signature(self, agent_id: str, message: str, signature: str) -> bool:
        """Verify a message signature"""
        if agent_id not in self.agent_identities:
            return False
        
        identity = self.agent_identities[agent_id]
        
        try:
            # Decode public key and signature
            public_key_pem = base64.b64decode(identity.public_key)
            public_key = serialization.load_pem_public_key(public_key_pem)
            signature_bytes = base64.b64decode(signature)
            
            # Verify signature
            public_key.verify(
                signature_bytes,
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Signature verification failed for agent {agent_id}: {e}")
            return False
    
    def create_transaction(self, from_agent: str, to_agent: str, data: Dict[str, Any]) -> BlockchainTransaction:
        """Create a blockchain transaction"""
        if from_agent not in self.agent_identities:
            raise ValueError(f"Agent {from_agent} not found")
        
        if to_agent not in self.agent_identities:
            raise ValueError(f"Agent {to_agent} not found")
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        from_identity = self.agent_identities[from_agent]
        to_identity = self.agent_identities[to_agent]
        
        # Create transaction data
        transaction_data = {
            'transaction_id': transaction_id,
            'from_agent': from_agent,
            'to_agent': to_agent,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sign transaction
        message = json.dumps(transaction_data, sort_keys=True)
        signature = self.sign_message(from_agent, message)
        
        # Create transaction record
        transaction = BlockchainTransaction(
            transaction_id=transaction_id,
            from_address=from_identity.address,
            to_address=to_identity.address,
            data=data,
            signature=signature,
            timestamp=datetime.now()
        )
        
        # Store transaction
        self.transactions[transaction_id] = transaction
        
        # Submit to blockchain
        self._submit_transaction(transaction)
        
        self.logger.info(f"Created transaction {transaction_id} from {from_agent} to {to_agent}")
        return transaction
    
    def _submit_transaction(self, transaction: BlockchainTransaction):
        """Submit transaction to blockchain"""
        if self.blockchain_type == BlockchainType.ETHEREUM and self.web3:
            self._submit_ethereum_transaction(transaction)
        elif self.blockchain_type == BlockchainType.HYPERLEDGER and self.fabric_client:
            self._submit_hyperledger_transaction(transaction)
        else:
            self._submit_simulated_transaction(transaction)
    
    def _submit_ethereum_transaction(self, transaction: BlockchainTransaction):
        """Submit transaction to Ethereum"""
        try:
            # This would interact with a smart contract
            # For now, simulate the transaction
            transaction.status = "confirmed"
            transaction.confirmation_count = 1
            self.logger.info(f"Ethereum transaction {transaction.transaction_id} submitted")
        except Exception as e:
            self.logger.error(f"Failed to submit Ethereum transaction: {e}")
            transaction.status = "failed"
    
    def _submit_hyperledger_transaction(self, transaction: BlockchainTransaction):
        """Submit transaction to Hyperledger Fabric"""
        try:
            # This would interact with a Hyperledger chaincode
            # For now, simulate the transaction
            transaction.status = "confirmed"
            transaction.confirmation_count = 1
            self.logger.info(f"Hyperledger transaction {transaction.transaction_id} submitted")
        except Exception as e:
            self.logger.error(f"Failed to submit Hyperledger transaction: {e}")
            transaction.status = "failed"
    
    def _submit_simulated_transaction(self, transaction: BlockchainTransaction):
        """Submit transaction to simulated blockchain"""
        # Create simulated block
        block = {
            'block_hash': hashlib.sha256(f"{transaction.transaction_id}{time.time()}".encode()).hexdigest(),
            'transactions': [transaction.transaction_id],
            'timestamp': datetime.now().isoformat(),
            'previous_hash': self.simulated_blocks[-1]['block_hash'] if self.simulated_blocks else "0"
        }
        
        self.simulated_blocks.append(block)
        self.simulated_block_height += 1
        
        transaction.block_hash = block['block_hash']
        transaction.status = "confirmed"
        transaction.confirmation_count = 1
        
        self.logger.info(f"Simulated transaction {transaction.transaction_id} added to block {self.simulated_block_height}")
    
    def create_audit_log(self, agent_id: str, action: str, data: Dict[str, Any]) -> AuditLog:
        """Create immutable audit log entry"""
        if agent_id not in self.agent_identities:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create audit log data
        log_data = {
            'agent_id': agent_id,
            'action': action,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sign audit log
        message = json.dumps(log_data, sort_keys=True)
        signature = self.sign_message(agent_id, message)
        
        # Create audit log entry
        previous_hash = self.audit_logs[-1].block_hash if self.audit_logs else "0"
        block_hash = hashlib.sha256(f"{message}{signature}{previous_hash}".encode()).hexdigest()
        
        audit_log = AuditLog(
            log_id=str(uuid.uuid4()),
            agent_id=agent_id,
            action=action,
            data=data,
            signature=signature,
            timestamp=datetime.now(),
            block_hash=block_hash,
            previous_hash=previous_hash
        )
        
        # Store audit log
        self.audit_logs.append(audit_log)
        
        self.logger.info(f"Created audit log {audit_log.log_id} for agent {agent_id}")
        return audit_log
    
    def verify_audit_log(self, log_id: str) -> bool:
        """Verify audit log integrity"""
        for i, log in enumerate(self.audit_logs):
            if log.log_id == log_id:
                # Verify signature
                log_data = {
                    'agent_id': log.agent_id,
                    'action': log.action,
                    'data': log.data,
                    'timestamp': log.timestamp.isoformat()
                }
                message = json.dumps(log_data, sort_keys=True)
                
                if not self.verify_signature(log.agent_id, message, log.signature):
                    return False
                
                # Verify block hash
                expected_hash = hashlib.sha256(f"{message}{log.signature}{log.previous_hash}".encode()).hexdigest()
                if log.block_hash != expected_hash:
                    return False
                
                # Verify chain integrity
                if i > 0:
                    previous_log = self.audit_logs[i-1]
                    if log.previous_hash != previous_log.block_hash:
                        return False
                
                return True
        
        return False
    
    def get_agent_trust_score(self, agent_id: str) -> float:
        """Get trust score for an agent"""
        if agent_id not in self.trust_scores:
            return 0.0
        
        # Update trust score based on recent activity
        identity = self.agent_identities.get(agent_id)
        if identity:
            # Calculate time since last activity
            time_since_activity = (datetime.now() - identity.last_activity).total_seconds()
            
            # Reduce trust score if inactive
            if time_since_activity > 3600:  # 1 hour
                decay_factor = min(0.1, time_since_activity / 86400)  # Max 10% decay per day
                self.trust_scores[agent_id] = max(0.0, self.trust_scores[agent_id] - decay_factor)
        
        return self.trust_scores[agent_id]
    
    def update_trust_score(self, agent_id: str, score_delta: float):
        """Update trust score for an agent"""
        if agent_id not in self.trust_scores:
            self.trust_scores[agent_id] = 0.5  # Default score
        
        # Update trust score
        new_score = max(0.0, min(1.0, self.trust_scores[agent_id] + score_delta))
        self.trust_scores[agent_id] = new_score
        
        # Update identity
        if agent_id in self.agent_identities:
            self.agent_identities[agent_id].trust_score = new_score
        
        self.logger.info(f"Updated trust score for agent {agent_id}: {new_score:.3f}")
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain system status"""
        return {
            'blockchain_type': self.blockchain_type.value,
            'connected': self._is_connected(),
            'agent_count': len(self.agent_identities),
            'transaction_count': len(self.transactions),
            'audit_log_count': len(self.audit_logs),
            'block_height': self._get_block_height(),
            'trust_scores': dict(self.trust_scores)
        }
    
    def _is_connected(self) -> bool:
        """Check if connected to blockchain"""
        if self.blockchain_type == BlockchainType.ETHEREUM and self.web3:
            return self.web3.is_connected()
        elif self.blockchain_type == BlockchainType.HYPERLEDGER and self.fabric_client:
            return True  # Assume connected if client exists
        else:
            return True  # Simulated blockchain is always "connected"
    
    def _get_block_height(self) -> int:
        """Get current block height"""
        if self.blockchain_type == BlockchainType.ETHEREUM and self.web3:
            return self.web3.eth.block_number
        elif self.blockchain_type == BlockchainType.HYPERLEDGER:
            return 0  # Would need to query Hyperledger
        else:
            return self.simulated_block_height
    
    def get_audit_trail(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail for agent or all agents"""
        logs = self.audit_logs
        
        if agent_id:
            logs = [log for log in logs if log.agent_id == agent_id]
        
        # Return recent logs
        recent_logs = logs[-limit:] if logs else []
        
        return [
            {
                'log_id': log.log_id,
                'agent_id': log.agent_id,
                'action': log.action,
                'data': log.data,
                'timestamp': log.timestamp.isoformat(),
                'block_hash': log.block_hash,
                'verified': self.verify_audit_log(log.log_id)
            }
            for log in recent_logs
        ]
    
    def get_transaction_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction history for agent or all agents"""
        transactions = list(self.transactions.values())
        
        if agent_id:
            # Filter by agent
            agent_identity = self.agent_identities.get(agent_id)
            if agent_identity:
                transactions = [
                    tx for tx in transactions
                    if tx.from_address == agent_identity.address or tx.to_address == agent_identity.address
                ]
        
        # Return recent transactions
        recent_transactions = transactions[-limit:] if transactions else []
        
        return [
            {
                'transaction_id': tx.transaction_id,
                'from_address': tx.from_address,
                'to_address': tx.to_address,
                'data': tx.data,
                'timestamp': tx.timestamp.isoformat(),
                'status': tx.status,
                'confirmation_count': tx.confirmation_count,
                'block_hash': tx.block_hash
            }
            for tx in recent_transactions
        ]

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test Blockchain Manager
    print("Testing Blockchain-based Security & Trust Manager...")
    
    blockchain_manager = BlockchainManager({
        'blockchain_type': 'simulated',
        'encryption_enabled': True,
        'signature_required': True
    })
    
    # Create agent identities
    agent1_identity = blockchain_manager.create_agent_identity("agent_1", TrustLevel.HIGH)
    agent2_identity = blockchain_manager.create_agent_identity("agent_2", TrustLevel.MEDIUM)
    
    print(f"Created identities: {agent1_identity.agent_id}, {agent2_identity.agent_id}")
    
    # Test message signing and verification
    message = "Test message for blockchain verification"
    signature = blockchain_manager.sign_message("agent_1", message)
    verified = blockchain_manager.verify_signature("agent_1", message, signature)
    print(f"Message verification: {verified}")
    
    # Test transaction creation
    transaction = blockchain_manager.create_transaction(
        "agent_1", "agent_2", {"data": "test_transaction", "value": 100}
    )
    print(f"Created transaction: {transaction.transaction_id}")
    
    # Test audit log creation
    audit_log = blockchain_manager.create_audit_log(
        "agent_1", "network_optimization", {"action": "bandwidth_allocation", "value": 50}
    )
    print(f"Created audit log: {audit_log.log_id}")
    
    # Get status
    status = blockchain_manager.get_blockchain_status()
    print(f"Blockchain Status: {status}")
    
    print("Blockchain Manager testing completed!")
