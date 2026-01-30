# path_finder.py
from typing import List, Dict, Optional, Any
import json
from neo4j import GraphDatabase
from config import Config
from connection import Neo4jConnection
import numpy as np

class PathFinder:
    """Handles path finding and relationship discovery in the knowledge graph."""
    
    def __init__(self, connection: Optional[Neo4jConnection] = None):
        """
        Initialize the PathFinder.
        
        Args:
            connection: Optional Neo4jConnection instance
        """
        self.config = Config
        self.max_path_length = self.config.PIPELINE.MAX_PATH_LENGTH
        self.max_paths_per_pair = self.config.PIPELINE.MAX_PATHS_PER_PAIR
        self.llm_model = self.config.PIPELINE.LLM_MODEL
        
        # Initialize connection
        if connection:
            self.connection = connection
        else:
            self.connection = Neo4jConnection()
        
        self.messages = []
        self.llm_client = None
    
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
    
    def set_llm_client(self, client):
        """Set LLM client for path pruning"""
        self.llm_client = client
        self._add_message("✅ LLM client configured for path pruning")
    
    def safe_json_parse(self, text: str) -> Optional[Dict]:
        """Safely parse JSON, handling common issues."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return None
    
    def find_paths_between_nodes(self, source_node_id: str, target_node_id: str, 
                                max_length: Optional[int] = None) -> List[Dict]:
        """
        Find all paths between two nodes in the knowledge graph.
        Returns paths as sequences of nodes and relationships.
        """
        if max_length is None:
            max_length = self.max_path_length
        
        try:
            with self.connection.driver.session() as session:
                # Find all paths up to max_length hops
                result = session.run("""
                MATCH path = (a)-[*1..%d]-(b)
                WHERE elementId(a) = $source_id AND elementId(b) = $target_id
                RETURN path, length(path) as pathLength
                ORDER BY pathLength ASC
                LIMIT 30
                """ % max_length,
                source_id=source_node_id, target_id=target_node_id)
                
                paths = []
                seen_paths = set()
                
                for record in result:
                    path = record["path"]
                    path_length = record["pathLength"]
                    
                    # Extract nodes and relationships from path
                    nodes_in_path = []
                    relationships_in_path = []
                    
                    # Get nodes
                    for node in path.nodes:
                        node_props = dict(node)
                        node_labels = list(node.labels)
                        node_id = str(node.element_id)
                        
                        # Get node name for display - biomedical context
                        node_name = self._extract_node_name(node_props, node_labels)
                        
                        nodes_in_path.append({
                            "id": node_id,
                            "labels": node_labels,
                            "name": node_name or f"Node_{node_id[-8:]}",
                            "properties": node_props
                        })
                    
                    # Get relationships
                    for rel in path.relationships:
                        rel_type = rel.type
                        rel_props = dict(rel)
                        
                        relationships_in_path.append({
                            "type": rel_type,
                            "properties": rel_props,
                            "start_node": str(rel.start_node.element_id),
                            "end_node": str(rel.end_node.element_id)
                        })
                    
                    # Create path string representation
                    path_string = self._create_path_string(nodes_in_path, relationships_in_path)
                    
                    # Create unique signature to avoid duplicates
                    path_signature = path_string
                    
                    if path_signature not in seen_paths:
                        seen_paths.add(path_signature)
                        
                        # Generate detailed description
                        detailed_desc = self._generate_detailed_path_description(nodes_in_path, relationships_in_path)
                        
                        paths.append({
                            "path_string": path_string,
                            "path_length": path_length,
                            "nodes": nodes_in_path,
                            "relationships": relationships_in_path,
                            "source_node": nodes_in_path[0],
                            "target_node": nodes_in_path[-1],
                            "detailed_description": detailed_desc,
                            "path_signature": path_signature
                        })
                
                self._add_message(f"🔗 Found {len(paths)} paths between nodes")
                return paths
                
        except Exception as e:
            self._add_message(f"❌ Error finding paths: {e}")
            return []
    
    def _extract_node_name(self, node_props: Dict, node_labels: List[str]) -> str:
        """Extract a display name from node properties."""
        # Biomedical property priority list
        biomedical_properties = [
            # Disease-specific properties
            "SNOMEDCT_US_definition", "mayo_causes", "mayo_complications", "mayo_prevention",
            "mayo_risk_factors", "mayo_see_doc", "mayo_symptoms", "mondo_definitions",
            "orphanet_clinical_description", "orphanet_definition", "orphanet_epidemiology",
            "orphanet_management_and_treatment", "orphanet_prevalence", "umls_descriptions",
            
            # Drug-specific properties
            "atc_4", "category", "clogp", "description", "group", "half_life", "indication",
            "mechanism_of_action", "molecular_weight", "pathway", "pharmacodynamics",
            "protein_binding", "state", "tpsa",
            
            # Common properties
            "node_name", "name", "title", "label", "display_name", "preferred_name",
            "generic_name", "brand_name", "disease_name"
        ]
        
        # Try biomedical properties first
        for prop_name in biomedical_properties:
            if prop_name in node_props and node_props[prop_name]:
                name = str(node_props[prop_name])
                # Truncate if too long
                if len(name) > 100:
                    name = name[:100] + "..."
                return name
        
        # Fallback: use any property with content
        for prop_name, prop_value in node_props.items():
            if prop_value and isinstance(prop_value, (str, int, float)):
                name = str(prop_value)
                if len(name) > 100:
                    name = name[:100] + "..."
                return name
        
        # Use labels as last resort
        if node_labels:
            return f"{node_labels[0]} Node"
        
        return "Unknown Node"
    
    def _create_path_string(self, nodes: List[Dict], relationships: List[Dict]) -> str:
        """Create a string representation of a path."""
        path_parts = []
        for i, node in enumerate(nodes):
            path_parts.append(node["name"])
            if i < len(relationships):
                rel = relationships[i]
                path_parts.append(f"--[{rel['type']}]-->")
        
        return " ".join(path_parts)
    
    def _generate_detailed_path_description(self, nodes: List[Dict], relationships: List[Dict]) -> str:
        """Generate a natural language description of the path."""
        description_parts = []
        
        for i, (node, rel) in enumerate(zip(nodes[:-1], relationships)):
            # Node description
            labels_str = ', '.join(node['labels'][:2])  # Show up to 2 labels
            if not labels_str:
                labels_str = "Entity"
            node_desc = f"{node['name']} ({labels_str})"
            
            # Relationship description
            rel_desc = rel['type'].lower().replace('_', ' ')
            
            description_parts.append(f"{node_desc} {rel_desc}")
        
        # Add last node
        if nodes:
            last_node = nodes[-1]
            labels_str = ', '.join(last_node['labels'][:2])
            if not labels_str:
                labels_str = "Entity"
            last_node_desc = f"{last_node['name']} ({labels_str})"
            description_parts.append(last_node_desc)
        
        return " → ".join(description_parts)
    
    def find_shortest_paths_between_entities(self, mapped_entities: List[Dict], 
                                           max_paths_per_pair: Optional[int] = None) -> Dict[str, Any]:
        """
        Find shortest paths between all pairs of successfully mapped entities.
        
        Args:
            mapped_entities: List of entity mappings from similarity matching
            max_paths_per_pair: Maximum paths to return per pair
            
        Returns:
            Dictionary with path results
        """
        if max_paths_per_pair is None:
            max_paths_per_pair = self.max_paths_per_pair
        
        self.clear_messages()
        
        path_results = {
            "entity_pairs": [],
            "total_paths_found": 0,
            "all_paths": [],
            "statistics": {
                "total_pairs": 0,
                "pairs_with_paths": 0,
                "average_path_length": 0,
                "unique_relationship_types": set()
            }
        }
        
        # Get successfully mapped nodes
        successful_mappings = [
            mapping for mapping in mapped_entities 
            if mapping.get("mapped_node") and mapping["mapped_node"].get("node_id")
        ]
        
        if len(successful_mappings) < 2:
            self._add_message("❌ Need at least 2 mapped entities to find paths")
            return path_results
        
        self._add_message(f"🔍 Searching paths between {len(successful_mappings)} mapped entities...")
        
        total_path_lengths = []
        
        # Find paths for each pair
        for i in range(len(successful_mappings)):
            for j in range(i + 1, len(successful_mappings)):
                mapping1 = successful_mappings[i]
                mapping2 = successful_mappings[j]
                
                source_id = mapping1["mapped_node"]["node_id"]
                target_id = mapping2["mapped_node"]["node_id"]
                source_name = mapping1["query_entity"]["name"]
                target_name = mapping2["query_entity"]["name"]
                source_type = mapping1["query_entity"].get("type", "Unknown")
                target_type = mapping2["query_entity"].get("type", "Unknown")
                
                self._add_message(f"  🔗 {source_name} ({source_type}) → {target_name} ({target_type})")
                
                paths = self.find_paths_between_nodes(source_id, target_id)
                
                if paths:
                    # Sort by path length and take top N
                    paths.sort(key=lambda x: x["path_length"])
                    top_paths = paths[:max_paths_per_pair]
                    
                    # Collect relationship types for statistics
                    for path in top_paths:
                        for rel in path["relationships"]:
                            path_results["statistics"]["unique_relationship_types"].add(rel["type"])
                    
                    # Calculate average path length
                    pair_avg_length = np.mean([p["path_length"] for p in paths]) if paths else 0
                    total_path_lengths.extend([p["path_length"] for p in paths])
                    
                    pair_result = {
                        "source_entity": source_name,
                        "target_entity": target_name,
                        "source_type": source_type,
                        "target_type": target_type,
                        "source_node": mapping1["mapped_node"]["node_name"],
                        "target_node": mapping2["mapped_node"]["node_name"],
                        "paths_found": len(paths),
                        "shortest_path_length": paths[0]["path_length"] if paths else None,
                        "average_path_length": round(pair_avg_length, 2),
                        "top_paths": top_paths,
                        "entity_mapping_confidence": {
                            "source": mapping1.get("confidence_score", 0),
                            "target": mapping2.get("confidence_score", 0)
                        }
                    }
                    
                    path_results["entity_pairs"].append(pair_result)
                    path_results["total_paths_found"] += len(paths)
                    path_results["all_paths"].extend(top_paths)
                    path_results["statistics"]["pairs_with_paths"] += 1
                    
                    self._add_message(f"    ✅ Found {len(paths)} paths (showing {len(top_paths)})")
                    
                    # Log sample paths
                    for path_idx, path in enumerate(top_paths[:2]):  # Show first 2
                        self._add_message(f"       {path_idx+1}. {path['path_string']}")
                else:
                    self._add_message(f"    ❌ No paths found")
                
                path_results["statistics"]["total_pairs"] += 1
        
        # Calculate statistics
        if total_path_lengths:
            path_results["statistics"]["average_path_length"] = round(np.mean(total_path_lengths), 2)
        
        # Convert set to list for JSON serialization
        path_results["statistics"]["unique_relationship_types"] = list(
            path_results["statistics"]["unique_relationship_types"]
        )
        
        self._add_message(f"📊 Path finding complete: {path_results['statistics']['pairs_with_paths']}/{path_results['statistics']['total_pairs']} pairs have paths")
        
        return path_results
    
    def prune_paths_with_llm(self, question: str, all_paths: List[Dict], 
                           entity_mappings: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to select the most relevant path for answering the question.
        
        Args:
            question: The original question
            all_paths: All found paths
            entity_mappings: Entity mappings from similarity matching
            
        Returns:
            Dictionary with selected path and reasoning
        """
        if not self.llm_client:
            self._add_message("⚠️ LLM client not configured, using fallback selection")
            return self._select_best_path_fallback(all_paths, entity_mappings)
        
        if not all_paths:
            return {"selected_path": None, "reasoning": "No paths found"}
        
        # Prepare entity mapping information
        entity_info = []
        for mapping in entity_mappings:
            if mapping.get("mapped_node"):
                entity = mapping["query_entity"]
                mapped = mapping["mapped_node"]
                entity_info.append({
                    "query_entity": entity["name"],
                    "entity_type": entity.get("type", "Unknown"),
                    "mapped_to": mapped["node_name"],
                    "mapping_confidence": mapped.get("enhanced_score", 0),
                    "mapping_stage": mapped.get("mapping_stage", "unknown")
                })
        
        # Prepare path candidates for LLM evaluation
        path_candidates = []
        for i, path in enumerate(all_paths[:10]):  # Limit to 10 for LLM
            # Calculate path relevance score
            entity_coverage = 0
            for mapping in entity_mappings:
                if mapping.get("mapped_node"):
                    mapped_name = mapping["mapped_node"]["node_name"]
                    # Check if this entity appears in the path
                    for node in path["nodes"]:
                        if node["name"] == mapped_name:
                            entity_coverage += 1
                            break
            
            path_relevance_score = entity_coverage / max(len(entity_mappings), 1)
            
            # Calculate biomedical relevance score
            biomedical_relevance = 0
            fertility_keywords = ["disease", "drug", "gene", "protein", "pathway", 
                                "treatment", "therapy", "infertility", "fertility"]
            
            for node in path["nodes"]:
                labels_lower = [label.lower() for label in node["labels"]]
                for keyword in fertility_keywords:
                    if any(keyword in label for label in labels_lower):
                        biomedical_relevance += 1
                        break
            
            path_candidates.append({
                "path_id": i + 1,
                "path_string": path["path_string"],
                "path_length": path["path_length"],
                "detailed_description": path.get("detailed_description", path["path_string"]),
                "entity_coverage": entity_coverage,
                "relevance_score": round(path_relevance_score, 2),
                "biomedical_relevance": biomedical_relevance
            })
        
        # Create LLM prompt for path pruning
        pruning_prompt = f"""
        You are a biomedical knowledge graph expert specializing in fertility and reproductive medicine. 
        Select the most relevant path for answering the given clinical question.

        CLINICAL QUESTION: "{question}"

        ENTITY MAPPINGS (from question to knowledge graph):
        {json.dumps(entity_info, indent=2)}

        AVAILABLE PATHS in knowledge graph:
        {json.dumps(path_candidates, indent=2)}

        SELECTION CRITERIA FOR BIOMEDICAL CONTEXT:
        1. CLINICAL RELEVANCE: Does the path directly address the clinical question about fertility?
        2. ENTITY COVERAGE: Does the path include all/most mapped clinical entities?
        3. BIOMEDICAL COHERENCE: Does the path make biological/medical sense?
        4. PATH LENGTH: Shorter paths are preferred for clarity
        5. FERTILITY CONTEXT: Does the path involve fertility-related concepts?

        OUTPUT FORMAT:
        {{
          "selected_path": {{
            "path_id": selected_path_number,
            "path_string": "selected_path_string",
            "reasoning": "detailed_explanation_focusing_on_biomedical_relevance_and_fertility_context",
            "confidence_score": 0.0_to_1.0
          }}
        }}

        If no path is clinically relevant, set "path_id" to -1.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You select the most relevant biomedical knowledge graph path for answering fertility-related clinical questions."},
                    {"role": "user", "content": pruning_prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            content = response.choices[0].message.content
            parsed = self.safe_json_parse(content)
            
            if not parsed:
                self._add_message("❌ Failed to parse LLM response, using fallback")
                return self._select_best_path_fallback(path_candidates, entity_mappings)
            
            selected_path = parsed.get("selected_path", {})
            path_id = selected_path.get("path_id", -1)
            
            if 1 <= path_id <= len(all_paths):
                selected_path_data = all_paths[path_id - 1]
                
                # Add LLM reasoning to the path
                selected_path_data["llm_selected"] = True
                selected_path_data["selection_reasoning"] = selected_path.get("reasoning", "LLM selected")
                selected_path_data["selection_confidence"] = selected_path.get("confidence_score", 0.8)
                
                result = {
                    "selected_path": selected_path_data,
                    "reasoning": selected_path.get("reasoning", "Path selected by LLM for clinical relevance"),
                    "confidence": selected_path.get("confidence_score", 0.8),
                    "all_considered_paths": path_candidates,
                    "llm_used": True
                }
                
                self._add_message(f"✅ LLM selected path {path_id} with confidence {result['confidence']:.2f}")
                return result
            else:
                self._add_message("⚠️ LLM found no relevant path, using fallback")
                return self._select_best_path_fallback(path_candidates, entity_mappings)
                
        except Exception as e:
            self._add_message(f"❌ Error in LLM path pruning: {e}")
            return self._select_best_path_fallback(path_candidates, entity_mappings)
    
    def _select_best_path_fallback(self, path_candidates: List[Dict], 
                                 entity_mappings: List[Dict]) -> Dict[str, Any]:
        """Fallback method to select best path when LLM fails."""
        if not path_candidates:
            return {"selected_path": None, "reasoning": "No paths available", "llm_used": False}
        
        # Score each path
        scored_paths = []
        for path in path_candidates:
            score = 0
            
            # Preference for shorter paths
            length_score = 1.0 / (path["path_length"] + 1)
            
            # Preference for higher entity coverage
            coverage_score = path["entity_coverage"] / max(len(entity_mappings), 1)
            
            # Preference for biomedical relevance
            biomedical_score = path.get("biomedical_relevance", 0) / max(len(entity_mappings), 1)
            
            # Combined score - weighted
            total_score = (length_score * 0.3) + (coverage_score * 0.4) + (biomedical_score * 0.3)
            
            scored_paths.append((total_score, path))
        
        # Select best path
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = scored_paths[0]
        
        self._add_message(f"🔧 Fallback: Selected path with score {best_score:.2f}")
        
        return {
            "selected_path": best_path,
            "reasoning": f"Fallback: Selected path with best entity coverage ({best_path['entity_coverage']}/{len(entity_mappings)}), biomedical relevance ({best_path.get('biomedical_relevance', 0)}), and length {best_path['path_length']}",
            "confidence": round(best_score, 2),
            "all_considered_paths": path_candidates,
            "llm_used": False
        }
    
    def get_node_neighbors(self, node_id: str, depth: int = 1, 
                          relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get neighbors of a node up to specified depth.
        
        Args:
            node_id: Node ID
            depth: Depth to explore (1=direct neighbors)
            relationship_types: Optional list of relationship types to filter
            
        Returns:
            Dictionary with neighbor information
        """
        try:
            with self.connection.driver.session() as session:
                rel_filter = ""
                params = {"node_id": node_id, "depth": depth}
                
                if relationship_types:
                    rel_types_str = "|".join([f":`{rt}`" for rt in relationship_types])
                    rel_filter = f"[:{rel_types_str}*1..{depth}]"
                
                query = f"""
                MATCH (n)
                WHERE elementId(n) = $node_id
                MATCH (n)-{rel_filter if rel_filter else f'[*1..$depth]'}-(neighbor)
                RETURN DISTINCT neighbor, 
                       elementId(neighbor) as neighbor_id,
                       labels(neighbor) as neighbor_labels,
                       [r in relationships(path) | type(r)] as rel_types,
                       length(shortestPath((n)-[*]-(neighbor))) as distance
                ORDER BY distance ASC
                LIMIT 50
                """
                
                result = session.run(query, params)
                
                neighbors = []
                for record in result:
                    neighbor = record["neighbor"]
                    neighbor_id = record["neighbor_id"]
                    neighbor_labels = record["neighbor_labels"]
                    rel_types = record["rel_types"]
                    distance = record["distance"]
                    
                    # Extract node properties
                    neighbor_props = dict(neighbor)
                    neighbor_name = self._extract_node_name(neighbor_props, neighbor_labels)
                    
                    neighbors.append({
                        "id": neighbor_id,
                        "labels": neighbor_labels,
                        "name": neighbor_name,
                        "distance": distance,
                        "relationship_types": list(set(rel_types)),  # Unique types
                        "properties": neighbor_props
                    })
                
                self._add_message(f"🔍 Found {len(neighbors)} neighbors for node")
                
                return {
                    "node_id": node_id,
                    "depth": depth,
                    "neighbors": neighbors,
                    "total_neighbors": len(neighbors),
                    "unique_relationship_types": list(set([rt for n in neighbors for rt in n["relationship_types"]]))
                }
                
        except Exception as e:
            self._add_message(f"❌ Error getting node neighbors: {e}")
            return {"node_id": node_id, "neighbors": [], "error": str(e)}
    
    def visualize_path(self, path: Dict) -> str:
        """
        Create a visualization-friendly representation of a path.
        
        Args:
            path: Path dictionary
            
        Returns:
            String representation suitable for visualization
        """
        if not path:
            return ""
        
        nodes = path.get("nodes", [])
        relationships = path.get("relationships", [])
        
        # Create a simple text visualization
        lines = []
        lines.append("=" * 80)
        lines.append(f"PATH VISUALIZATION (Length: {path.get('path_length', 0)})")
        lines.append("=" * 80)
        
        for i, node in enumerate(nodes):
            # Node info
            node_line = f"[{i+1}] {node['name']}"
            if node.get('labels'):
                node_line += f" ({', '.join(node['labels'][:2])})"
            lines.append(node_line)
            
            # Relationship info (if not last node)
            if i < len(relationships):
                rel = relationships[i]
                rel_line = f"    ║ {rel['type']} ║"
                if rel.get('properties'):
                    # Show important properties if available
                    important_props = ['weight', 'confidence', 'source']
                    for prop in important_props:
                        if prop in rel['properties']:
                            rel_line += f" [{prop}: {rel['properties'][prop]}]"
                            break
                lines.append(rel_line)
                lines.append("    ↓")
        
        # Add path description if available
        if path.get('detailed_description'):
            lines.append("\n📝 Description:")
            lines.append(path['detailed_description'])
        
        return "\n".join(lines)
    
    def validate_path(self, path: Dict) -> Dict[str, Any]:
        """
        Validate a path structure.
        
        Args:
            path: Path dictionary
            
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check required fields
        required_fields = ["path_string", "path_length", "nodes", "relationships"]
        for field in required_fields:
            if field not in path:
                validation["is_valid"] = False
                validation["issues"].append(f"Missing required field: {field}")
        
        # Check node-relationship consistency
        nodes = path.get("nodes", [])
        relationships = path.get("relationships", [])
        
        if len(nodes) != len(relationships) + 1:
            validation["warnings"].append(f"Node-relationship count mismatch: {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Check for valid node IDs
        for i, node in enumerate(nodes):
            if not node.get("id"):
                validation["warnings"].append(f"Node {i} missing ID")
            if not node.get("name"):
                validation["warnings"].append(f"Node {i} missing name")
        
        # Check for valid relationship types
        for i, rel in enumerate(relationships):
            if not rel.get("type"):
                validation["warnings"].append(f"Relationship {i} missing type")
        
        # Collect statistics
        validation["statistics"] = {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "unique_node_labels": list(set([label for node in nodes for label in node.get("labels", [])])),
            "unique_relationship_types": list(set([rel.get("type", "") for rel in relationships]))
        }
        
        return validation