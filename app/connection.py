# connection.py
from neo4j import GraphDatabase
from config import Config
from typing import List, Dict
from collections import defaultdict

class Neo4jConnection:
    def __init__(self):
        self.driver = None
        self.config = Config.NEO4J
        self.connect()
    
    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.URI,
                auth=(self.config.USER, self.config.PASSWORD)
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    print("✅ Successfully connected to Neo4j")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("🔒 Neo4j connection closed")
    
    def test_connection(self):
        """Test if connection is active"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except:
            return False
    
    def execute_query(self, query: str, parameters: Dict = None):
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return result.data()
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return []