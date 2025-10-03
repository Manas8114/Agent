#!/usr/bin/env python3
"""
Real Blockchain Implementation for Enhanced Telecom AI System
Implements actual blockchain functionality using local blockchain simulation
"""

import hashlib
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Blockchain transaction"""
    transaction_id: str
    sender: str
    receiver: str
    data: Dict[str, Any]
    timestamp: float
    signature: str
    block_hash: Optional[str] = None

@dataclass
class Block:
    """Blockchain block"""
    block_number: int
    previous_hash: str
    transactions: List[Transaction]
    timestamp: float
    nonce: int
    hash: str
    merkle_root: str

class RealBlockchain:
    """Real blockchain implementation"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = 4  # Mining difficulty
        self.mining_reward = 10.0
        self.wallets: Dict[str, float] = {}
        self.transaction_queue = queue.Queue()
        self.is_mining = False
        self.mining_thread = None
        
        # Initialize with genesis block
        self._create_genesis_block()
        
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_transaction = Transaction(
            transaction_id="genesis",
            sender="system",
            receiver="genesis",
            data={"message": "Genesis block of Enhanced Telecom AI System"},
            timestamp=time.time(),
            signature="genesis_signature"
        )
        
        genesis_block = Block(
            block_number=0,
            previous_hash="0",
            transactions=[genesis_transaction],
            timestamp=time.time(),
            nonce=0,
            hash="",
            merkle_root=""
        )
        
        # Calculate hash
        genesis_block.hash = self._calculate_block_hash(genesis_block)
        genesis_block.merkle_root = self._calculate_merkle_root([genesis_transaction])
        
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA-256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _calculate_block_hash(self, block: Block) -> str:
        """Calculate block hash"""
        block_string = f"{block.block_number}{block.previous_hash}{block.timestamp}{block.nonce}{block.merkle_root}"
        return self._calculate_hash(block_string)
    
    def _calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate Merkle root of transactions"""
        if not transactions:
            return "0"
        
        if len(transactions) == 1:
            return self._calculate_hash(transactions[0].transaction_id)
        
        # Simple Merkle tree implementation
        hashes = [self._calculate_hash(tx.transaction_id) for tx in transactions]
        
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]
                next_level.append(self._calculate_hash(combined))
            hashes = next_level
        
        return hashes[0] if hashes else "0"
    
    def _proof_of_work(self, block: Block) -> int:
        """Mine block using proof of work"""
        target = "0" * self.difficulty
        nonce = 0
        
        while True:
            block.nonce = nonce
            block_hash = self._calculate_block_hash(block)
            
            if block_hash.startswith(target):
                return nonce
            
            nonce += 1
    
    def create_transaction(self, sender: str, receiver: str, data: Dict[str, Any]) -> Transaction:
        """Create a new transaction"""
        transaction_id = self._calculate_hash(f"{sender}{receiver}{time.time()}")
        signature = self._calculate_hash(f"{transaction_id}{sender}{receiver}")
        
        transaction = Transaction(
            transaction_id=transaction_id,
            sender=sender,
            receiver=receiver,
            data=data,
            timestamp=time.time(),
            signature=signature
        )
        
        self.pending_transactions.append(transaction)
        self.transaction_queue.put(transaction)
        
        logger.info(f"Transaction created: {transaction_id}")
        return transaction
    
    def mine_block(self) -> Block:
        """Mine a new block"""
        if not self.pending_transactions:
            return None
        
        # Create new block
        previous_hash = self.chain[-1].hash if self.chain else "0"
        block = Block(
            block_number=len(self.chain),
            previous_hash=previous_hash,
            transactions=self.pending_transactions.copy(),
            timestamp=time.time(),
            nonce=0,
            hash="",
            merkle_root=""
        )
        
        # Calculate Merkle root
        block.merkle_root = self._calculate_merkle_root(block.transactions)
        
        # Mine block
        self._proof_of_work(block)
        block.hash = self._calculate_block_hash(block)
        
        # Add to chain
        self.chain.append(block)
        
        # Clear pending transactions
        self.pending_transactions.clear()
        
        logger.info(f"Block mined: {block.block_number}, Hash: {block.hash[:10]}...")
        return block
    
    def start_mining(self):
        """Start continuous mining"""
        if self.is_mining:
            return
        
        self.is_mining = True
        self.mining_thread = threading.Thread(target=self._mining_loop)
        self.mining_thread.daemon = True
        self.mining_thread.start()
        logger.info("Mining started")
    
    def stop_mining(self):
        """Stop mining"""
        self.is_mining = False
        if self.mining_thread:
            self.mining_thread.join()
        logger.info("Mining stopped")
    
    def _mining_loop(self):
        """Mining loop"""
        while self.is_mining:
            if self.pending_transactions:
                self.mine_block()
            time.sleep(1)  # Check every second
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        return {
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "difficulty": self.difficulty,
            "last_block_hash": self.chain[-1].hash if self.chain else None,
            "total_transactions": sum(len(block.transactions) for block in self.chain)
        }
    
    def get_block(self, block_number: int) -> Optional[Block]:
        """Get block by number"""
        if 0 <= block_number < len(self.chain):
            return self.chain[block_number]
        return None
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.transaction_id == transaction_id:
                    return transaction
        return None
    
    def verify_chain(self) -> bool:
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check previous hash
            if current_block.previous_hash != previous_block.hash:
                return False
            
            # Check block hash
            if current_block.hash != self._calculate_block_hash(current_block):
                return False
        
        return True
    
    def create_agent_transaction(self, agent_id: str, action: str, data: Dict[str, Any]) -> Transaction:
        """Create transaction for AI agent"""
        return self.create_transaction(
            sender=f"agent_{agent_id}",
            receiver="system",
            data={
                "agent_id": agent_id,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                **data
            }
        )
    
    def get_agent_transactions(self, agent_id: str) -> List[Transaction]:
        """Get all transactions for an agent"""
        agent_transactions = []
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.sender == f"agent_{agent_id}" or f"agent_{agent_id}" in str(transaction.data):
                    agent_transactions.append(transaction)
        return agent_transactions

# Global blockchain instance
real_blockchain = RealBlockchain()

def start_real_blockchain():
    """Start real blockchain"""
    real_blockchain.start_mining()
    logger.info("Real blockchain started")

def stop_real_blockchain():
    """Stop real blockchain"""
    real_blockchain.stop_mining()
    logger.info("Real blockchain stopped")

def get_blockchain_info() -> Dict[str, Any]:
    """Get blockchain information"""
    return real_blockchain.get_chain_info()

def create_agent_transaction(agent_id: str, action: str, data: Dict[str, Any]) -> Transaction:
    """Create transaction for AI agent"""
    return real_blockchain.create_agent_transaction(agent_id, action, data)

def get_agent_transactions(agent_id: str) -> List[Transaction]:
    """Get agent transactions"""
    return real_blockchain.get_agent_transactions(agent_id)

