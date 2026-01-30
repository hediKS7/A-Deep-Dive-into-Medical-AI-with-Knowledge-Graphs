# data_loader.py
from connection import Neo4jConnection
from config import Config
from embedding_model import EmbeddingModel
from typing import List, Dict, Tuple
from collections import defaultdict
import json
import numpy as np

class KGDataLoader:
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.entity_config = Config.ENTITY
        self.label_counts = defaultdict(int)
        self.type_counts = defaultdict(int)
        self.kg_nodes = []
        self.kg_texts = []
        self.kg_embeddings = None
        self.embedding_model = None
        self.messages = []  # Store messages for Streamlit
    
    def _add_message(self, message: str):
        """Add a message to the messages list"""
        self.messages.append(message)
        print(message)  # Keep original print for debugging
    
    def clear_messages(self):
        """Clear all stored messages"""
        self.messages = []
    
    def get_messages(self) -> List[str]:
        """Get all stored messages"""
        return self.messages
    
    def initialize_embedding_model(self, model_name: str = "all-MiniLM-L6-v2") -> Tuple[bool, List[str]]:
        """Initialize the embedding model"""
        try:
            self._add_message(f"🔄 Initializing embedding model: {model_name}")
            self.embedding_model = EmbeddingModel(model_name)
            self._add_message("✅ Embedding model initialized")
            return True, self.messages
        except Exception as e:
            self._add_message(f"❌ Failed to initialize embedding model: {e}")
            return False, self.messages
    
    def compute_embeddings(self) -> Tuple[bool, List[str]]:
        """Compute embeddings for loaded KG texts"""
        if not self.kg_texts:
            self._add_message("❌ No texts available for embedding. Load data first.")
            return False, self.messages
        
        if self.embedding_model is None:
            success, msgs = self.initialize_embedding_model()
            if not success:
                return False, self.messages + msgs
        
        try:
            # Compute embeddings
            self.kg_embeddings, emb_messages = self.embedding_model.encode_texts(
                self.kg_texts, 
                show_progress_bar=True
            )
            
            # Add embedding messages to our messages
            self.messages.extend(emb_messages)
            
            if len(self.kg_embeddings) > 0:
                self._add_message(f"✅ Computed embeddings for {len(self.kg_embeddings)} nodes")
                self._add_message(f"📏 Embedding dimension: {self.kg_embeddings.shape[1]}")
                return True, self.messages
            else:
                self._add_message("❌ Failed to compute embeddings")
                return False, self.messages
                
        except Exception as e:
            self._add_message(f"❌ Error computing embeddings: {e}")
            return False, self.messages
    
    def load_kg_nodes_with_properties(self) -> List[Dict]:
        """Load ONLY disease and drug nodes from the KG and combine all properties for embeddings."""
        
        if not self.neo4j_conn.test_connection():
            self._add_message("❌ No connection to Neo4j")
            return []
        
        with self.neo4j_conn.driver.session() as s:
            # Count disease and drug nodes
            count_result = s.run("""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            RETURN count(n) AS total_disease_drug_nodes
            """, {"entity_types": self.entity_config.TYPES}).data()
            
            total_disease_drug_nodes = count_result[0]["total_disease_drug_nodes"]
            self._add_message(f"📊 Total disease and drug nodes in KG: {total_disease_drug_nodes}")
            
            # Load ONLY disease and drug nodes with properties
            nodes = s.run("""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN $entity_types)
            RETURN elementId(n) AS node_id, labels(n) AS labels, properties(n) AS properties
            """, {"entity_types": self.entity_config.TYPES}).data()
        
        self._add_message(f"📥 Loaded {len(nodes)} disease/drug nodes from Neo4j query")
        
        enhanced_nodes = []
        label_counts_local = defaultdict(int)
        type_counts_local = defaultdict(int)
        
        for node in nodes:
            props = dict(node["properties"])
            labels = node["labels"]
            
            # Count label frequencies
            for label in labels:
                label_counts_local[label] += 1
            
            # Create combined text from ALL property values
            combined_text = " ".join(str(v) for v in props.values() if v is not None)
            
            # Determine best node name
            node_name = self._extract_node_name(props)
            
            # Determine entity type based on labels
            entity_type = self._determine_entity_type(labels, props)
            
            # Skip if not in configured entity types
            if entity_type not in self.entity_config.TYPES:
                continue
            
            # Count entity types
            type_counts_local[entity_type] += 1
            
            # Create enhanced property text
            property_text = self._create_property_text(node_name, entity_type, props)
            
            enhanced_nodes.append({
                "node_id": str(node["node_id"]),
                "node_name": node_name,
                "labels": labels,
                "entity_type": entity_type,
                "properties": props,
                "property_keys": list(props.keys()),
                "combined_text": combined_text.strip(),
                "property_text": property_text.strip(),
                "text_length": len(property_text.strip())
            })
        
        # Update statistics
        self.label_counts.update(label_counts_local)
        self.type_counts.update(type_counts_local)
        
        # Print statistics
        self._print_statistics(label_counts_local, type_counts_local, enhanced_nodes)
        
        return enhanced_nodes
    
    def _extract_node_name(self, props: Dict) -> str:
        """Extract node name from properties"""
        # First, check for node_name property
        if "node_name" in props and props["node_name"]:
            return str(props["node_name"])
        
        # Try other name candidates
        name_candidates = [
            "name", "Name", "NAME",
            "description", "Description",
            "SNOMEDCT_US_definition", "mondo_definitions",
            "indication", "mechanism_of_action",
            "title", "Title", "label", "Label"
        ]
        
        for prop in name_candidates:
            if prop in props and props[prop]:
                node_name = str(props[prop])
                if len(node_name) > 100:
                    node_name = node_name[:100] + "..."
                return node_name
        
        # Fallback: use first property value
        if props:
            for value in props.values():
                if value is not None:
                    node_name = str(value)
                    if len(node_name) > 100:
                        node_name = node_name[:100] + "..."
                    return node_name
        
        # Final fallback
        return f"Node_{len(self.kg_nodes)+1}"
    
    def _determine_entity_type(self, labels: List[str], props: Dict) -> str:
        """Determine entity type based on labels and properties"""
        # Check if node has disease or drug label
        for label in labels:
            if label.lower() in self.entity_config.TYPES:
                return label.lower()
        
        # If still unknown, try to infer from properties
        # Check for disease-specific properties
        disease_props = ["SNOMEDCT_US_definition", "mondo_definitions", "mayo_symptoms", "orphanet_definition"]
        if any(prop in props for prop in disease_props):
            return "disease"
        
        # Check for drug-specific properties
        drug_props = ["mechanism_of_action", "atc_4", "indication", "pharmacodynamics"]
        if any(prop in props for prop in drug_props):
            return "drug"
        
        # Fuzzy label matching
        label_str = " ".join(labels).lower()
        if "disease" in label_str or "disorder" in label_str or "syndrome" in label_str:
            return "disease"
        elif "drug" in label_str or "compound" in label_str or "medication" in label_str:
            return "drug"
        
        # Default to first label
        return labels[0] if labels else "Unknown"
    
    def _create_property_text(self, node_name: str, entity_type: str, props: Dict) -> str:
        """Create enhanced property text for better embeddings"""
        property_text = node_name
        
        # Use schema-based property enhancement
        schema = self.entity_config.get_schema(entity_type)
        if schema:
            # Add primary keys to property text
            for prop in schema.get("primary_keys", []):
                if prop in props and props[prop]:
                    prop_value = str(props[prop])
                    if prop_value and prop_value.lower() not in property_text.lower():
                        property_text += f" {prop_value}"
            
            # Add display properties to property text
            for prop in schema.get("display_properties", []):
                if prop in props and props[prop]:
                    prop_value = str(props[prop])
                    if prop_value and prop_value.lower() not in property_text.lower():
                        property_text += f" {prop_value}"
        else:
            # For unknown schema, add important properties
            important_props = ["description", "definition", "indication", "symptoms", "mechanism"]
            added_count = 0
            for prop in important_props:
                if prop in props and props[prop] and added_count < 3:
                    prop_value = str(props[prop])
                    if prop_value and prop_value.lower() not in property_text.lower():
                        property_text += f" {prop_value}"
                        added_count += 1
        
        # Limit property text length
        if len(property_text) > 500:
            property_text = property_text[:500] + "..."
        
        return property_text
    
    def _print_statistics(self, label_counts: Dict, type_counts: Dict, nodes: List[Dict]):
        """Print loading statistics"""
        # Print label distribution
        self._add_message(f"\n📊 Label Distribution (disease/drug):")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            self._add_message(f"   {label}: {count}")
        
        # Print entity type distribution
        self._add_message(f"\n📊 Entity Type Distribution:")
        
        # Count and display disease vs drug distribution
        disease_count = type_counts.get("disease", 0)
        drug_count = type_counts.get("drug", 0)
        total_filtered = disease_count + drug_count
        
        if total_filtered > 0:
            self._add_message(f"   Disease: {disease_count} ({disease_count/total_filtered*100:.1f}%)")
            self._add_message(f"   Drug: {drug_count} ({drug_count/total_filtered*100:.1f}%)")
        
        # Print other entity types if any
        for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            if entity_type not in self.entity_config.TYPES:
                self._add_message(f"   {entity_type}: {count}")
        
        # Print text length statistics
        if nodes:
            avg_length = sum(node["text_length"] for node in nodes) / len(nodes)
            self._add_message(f"\n📏 Average property text length: {avg_length:.0f} characters")
    
    def load_data(self, compute_embeddings: bool = True) -> Tuple[bool, List[str]]:
        """Main method to load all data.
        Returns: (success, messages)"""
        self.clear_messages()  # Clear previous messages
        
        self._add_message("🔄 Loading disease and drug nodes from knowledge graph...")
        self.kg_nodes = self.load_kg_nodes_with_properties()
        
        if not self.kg_nodes:
            self._add_message("❌ ERROR: No disease or drug nodes loaded!")
            return False, self.messages
        
        self.kg_texts = [node["property_text"] for node in self.kg_nodes]
        self._add_message(f"✅ Successfully loaded {len(self.kg_nodes)} disease/drug nodes")
        
        # Print sample nodes
        self._print_sample_nodes()
        
        # Save summary
        self._save_summary()
        
        # Compute embeddings if requested
        if compute_embeddings:
            self._add_message("\n🔄 Computing embeddings for loaded nodes...")
            success, emb_messages = self.compute_embeddings()
            self.messages.extend(emb_messages)
            
            if not success:
                self._add_message("⚠️ Data loaded but embeddings computation failed")
        
        return True, self.messages
    
    def _print_sample_nodes(self):
        """Print sample of loaded nodes"""
        self._add_message(f"\n📋 Sample disease/drug nodes loaded:")
        for i, node in enumerate(self.kg_nodes[:5]):
            entity_type = node.get('entity_type', 'Unknown')
            props_count = len(node['properties'])
            name_preview = node['node_name'][:60] + "..." if len(node['node_name']) > 60 else node['node_name']
            text_preview = node['property_text'][:80] + "..." if len(node['property_text']) > 80 else node['property_text']
            self._add_message(f"  {i+1}. {name_preview} ({entity_type}) - Props: {props_count}")
            self._add_message(f"     Preview: {text_preview}")
    
    def _save_summary(self):
        """Save summary to JSON file"""
        try:
            with open("disease_drug_nodes_summary.json", "w") as f:
                summary = {
                    "total_nodes": len(self.kg_nodes),
                    "entity_type_counts": dict(self.type_counts),
                    "label_counts": dict(self.label_counts),
                    "sample_nodes": self.kg_nodes[:10]
                }
                json.dump(summary, f, indent=2, default=str)
            self._add_message(f"\n💾 Summary saved to disease_drug_nodes_summary.json")
        except Exception as e:
            self._add_message(f"⚠️ Could not save summary: {e}")
    
    def get_nodes(self):
        """Get loaded nodes"""
        return self.kg_nodes
    
    def get_texts(self):
        """Get node texts"""
        return self.kg_texts
    
    def get_embeddings(self):
        """Get node embeddings"""
        return self.kg_embeddings
    
    def get_statistics(self):
        """Get loading statistics"""
        return {
            "total_nodes": len(self.kg_nodes),
            "entity_type_counts": dict(self.type_counts),
            "label_counts": dict(self.label_counts),
            "embeddings_computed": self.kg_embeddings is not None and len(self.kg_embeddings) > 0
        }
    
    def find_similar_nodes(self, query: str, top_k: int = 5) -> Tuple[List[Dict], List[str]]:
        """Find similar nodes based on query text"""
        if self.embedding_model is None or self.kg_embeddings is None:
            self._add_message("❌ Embeddings not available. Compute embeddings first.")
            return [], self.messages
        
        return self.embedding_model.find_similar(
            query=query,
            corpus_texts=self.kg_texts,
            corpus_embeddings=self.kg_embeddings,
            top_k=top_k)
    
    def find_similar_with_enhancement(
        self, 
        entity_name: str, 
        entity_type: str = "Unknown", 
        top_k: int = None
    ) -> Tuple[List[Dict], List[str]]:
        """
        Find similar nodes with enhanced similarity calculation.
        
        Args:
            entity_name: Name of entity to find matches for
            entity_type: Type of entity (disease/drug)
            top_k: Number of top matches to return
        
        Returns:
            Tuple of (enhanced_matches, messages)
        """
        if not self.kg_nodes or self.kg_embeddings is None:
            self._add_message("❌ KG data not loaded. Please load data first.")
            return [], self.messages
        
        if top_k is None:
            from config import Config
            top_k = Config.PIPELINE.TOP_K
        
        # Initialize similarity calculator if needed
        if not hasattr(self, 'similarity_calculator'):
            from similarity import SimilarityCalculator
            self.similarity_calculator = SimilarityCalculator(self.embedding_model)
        
        # Get enhanced candidates
        enhanced_candidates, sim_messages = self.similarity_calculator.get_enhanced_candidates(
            entity_name=entity_name,
            entity_type=entity_type,
            kg_nodes=self.kg_nodes,
            kg_embeddings=self.kg_embeddings,
            top_k=top_k
        )
        
        # Add similarity messages to our messages
        self.messages.extend(sim_messages)
        
        # Convert to match format
        enhanced_matches = []
        for candidate in enhanced_candidates:
            match = {
                "index": candidate["kg_index"],
                "node_id": candidate["node_id"],
                "node_name": candidate["node_name"],
                "entity_type": candidate["entity_type"],
                "similarity_score": candidate["similarity_score"],
                "enhanced_score": candidate["enhanced_score"],
                "match_type": candidate["match_type"],
                "property_info": candidate.get("property_info", {}),
                "confidence": candidate.get("property_info", {}).get("confidence", "medium")
            }
            enhanced_matches.append(match)
        
        return enhanced_matches, self.messages