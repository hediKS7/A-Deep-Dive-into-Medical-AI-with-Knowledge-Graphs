# entity_mapping.py
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from config import Config
from embedding_model import EmbeddingModel
from similarity import SimilarityCalculator
import numpy as np

class EntityMapper:
    """Handles multi-stage entity mapping to knowledge graph with property-aware matching"""
    
    def __init__(self, similarity_calculator: Optional[SimilarityCalculator] = None, 
                 llm_client = None):
        """
        Initialize the EntityMapper
        
        Args:
            similarity_calculator: SimilarityCalculator instance
            llm_client: Optional LLM client for advanced matching
        """
        self.similarity_calculator = similarity_calculator
        self.llm_client = llm_client
        self.llm_model = Config.PIPELINE.LLM_MODEL
        self.entity_config = Config.ENTITY
        self.pipeline_config = Config.PIPELINE
        self.messages = []  
        self._initialize_thresholds()
    
    def _initialize_thresholds(self):
        """Initialize matching thresholds"""
        self.similarity_threshold = self.pipeline_config.SIM_THRESHOLD
        self.exact_match_threshold = self.pipeline_config.EXACT_MATCH_THRESHOLD
        self.top_k = self.pipeline_config.TOP_K
    
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
    
    def set_similarity_calculator(self, similarity_calculator: SimilarityCalculator):
        """Set similarity calculator"""
        self.similarity_calculator = similarity_calculator
        self._add_message("✅ Similarity calculator configured for entity mapping")
    
    def set_llm_client(self, client):
        """Set LLM client for advanced matching"""
        self.llm_client = client
        self._add_message("✅ LLM client configured for entity mapping")
    
    def stage1_property_exact_match(
        self, 
        entity_name: str, 
        entity_type: str, 
        candidates: List[Dict]
    ) -> Optional[Dict]:
        """
        Stage 1: Property-based exact matching.
        
        Args:
            entity_name: Name of the entity to match
            entity_type: Type of entity (disease/drug)
            candidates: List of candidate KG nodes
            
        Returns:
            Best exact match candidate or None
        """
        entity_lower = entity_name.lower().strip()
        
        for candidate in candidates:
            candidate_props = candidate.get("properties", {})
            
            # Get entity schema
            schema = self.entity_config.PROPERTY_SCHEMAS.get(entity_type, {})
            primary_keys = schema.get("primary_keys", [])
            
            # Check if entity matches any primary key exactly
            for prop in primary_keys:
                if prop in candidate_props:
                    prop_value = str(candidate_props[prop]).lower()
                    if prop_value == entity_lower:
                        candidate["property_match_type"] = f"exact_primary_key:{prop}"
                        candidate["match_type"] = f"exact_primary_key:{prop}"
                        self._add_message(f"🎯 Stage 1: Found exact primary key match on '{prop}'")
                        return candidate
            
            # Check if entity matches any display property exactly
            display_props = schema.get("display_properties", [])
            for prop in display_props:
                if prop in candidate_props:
                    prop_value = str(candidate_props[prop]).lower()
                    if prop_value == entity_lower:
                        candidate["property_match_type"] = f"exact_display_property:{prop}"
                        candidate["match_type"] = f"exact_display_property:{prop}"
                        self._add_message(f"🎯 Stage 1: Found exact display property match on '{prop}'")
                        return candidate
            
            # Check name property exactly
            if candidate["node_name"].lower().strip() == entity_lower:
                candidate["property_match_type"] = "exact_name_match"
                candidate["match_type"] = "exact_name_match"
                self._add_message(f"🎯 Stage 1: Found exact name match: {candidate['node_name']}")
                return candidate
        
        self._add_message("⚠️ Stage 1: No exact property matches found")
        return None
    
    def stage2_property_enhanced_similarity(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Stage 2: Property-enhanced similarity matching.
        
        Args:
            candidates: List of candidate KG nodes with enhanced scores
            
        Returns:
            Best enhanced similarity candidate or None
        """
        if not candidates:
            return None
        
        # Sort candidates by enhanced score
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get("enhanced_score", 0), 
            reverse=True
        )
        
        best_candidate = sorted_candidates[0]
        best_score = best_candidate.get("enhanced_score", 0)
        
        # Adjust threshold based on match type
        threshold = self.similarity_threshold
        match_type = best_candidate.get("match_type", "")
        
        if any(x in match_type for x in ["exact_primary_key", "exact_display_property", "exact_name_match"]):
            threshold = 0.8  # Lower threshold for exact matches
            self._add_message(f"📊 Stage 2: Adjusted threshold to {threshold} for exact match type: {match_type}")
        elif "contains" in match_type:
            threshold = 0.6
            self._add_message(f"📊 Stage 2: Adjusted threshold to {threshold} for contains match type: {match_type}")
        elif best_score > 0.7:
            threshold = 0.65  # Adjust for high scores
            self._add_message(f"📊 Stage 2: Adjusted threshold to {threshold} for high score: {best_score:.3f}")
        
        if best_score >= threshold:
            self._add_message(f"✅ Stage 2: Found strong enhanced similarity match "
                            f"(score: {best_score:.3f}, type: {match_type})")
            return best_candidate
        else:
            self._add_message(f"⚠️ Stage 2: Best enhanced score {best_score:.3f} "
                            f"below threshold {threshold}")
            return None
    
    def stage3_property_contextual_llm_match(
        self, 
        entity: Dict, 
        candidates: List[Dict], 
        context: str = ""
    ) -> Optional[Dict]:
        """
        Stage 3: LLM-based matching with biomedical property context.
        
        Args:
            entity: Query entity dictionary
            candidates: List of candidate KG nodes
            context: Optional context/question for better matching
            
        Returns:
            LLM-selected candidate or None
        """
        if not candidates or not self.llm_client:
            return None
        
        self._add_message("🧠 Stage 3: Using LLM for contextual property matching")
        
        # Prepare detailed candidate information with biomedical properties
        candidates_info = []
        for i, candidate in enumerate(candidates[:5]):  # Limit to top 5 for LLM
            candidate_info = {
                "index": i,
                "node_name": candidate["node_name"],
                "labels": candidate["labels"],
                "entity_type": candidate.get("entity_type", "Unknown"),
                "similarity_score": round(candidate["similarity_score"], 4),
                "enhanced_score": round(candidate.get("enhanced_score", 0), 4),
                "match_type": candidate.get("match_type", "unknown")
            }
            
            # Add relevant biomedical properties
            candidate_props = candidate.get("properties", {})
            schema = self.entity_config.PROPERTY_SCHEMAS.get(candidate_info["entity_type"], {})
            
            # Include important biomedical properties
            important_props = {}
            
            # For disease entities
            if candidate_info["entity_type"] == "disease":
                disease_props = [
                    "mondo_definitions", "mayo_symptoms", "orphanet_prevalence", 
                    "SNOMEDCT_US_definition", "orphanet_definition", "mayo_causes"
                ]
                for prop in disease_props:
                    if prop in candidate_props and candidate_props[prop]:
                        # Truncate long text for display
                        prop_value = str(candidate_props[prop])
                        if len(prop_value) > 150:
                            prop_value = prop_value[:150] + "..."
                        important_props[prop] = prop_value
                        if len(important_props) >= 3:
                            break
            
            # For drug entities  
            elif candidate_info["entity_type"] == "drug":
                drug_props = [
                    "description", "indication", "mechanism_of_action", 
                    "category", "atc_4", "pharmacodynamics"
                ]
                for prop in drug_props:
                    if prop in candidate_props and candidate_props[prop]:
                        prop_value = str(candidate_props[prop])
                        if len(prop_value) > 150:
                            prop_value = prop_value[:150] + "..."
                        important_props[prop] = prop_value
                        if len(important_props) >= 3:
                            break
            
            # Include primary keys
            for prop in schema.get("primary_keys", [])[:2]:
                if prop in candidate_props and prop not in important_props:
                    important_props[prop] = candidate_props[prop]
            
            candidate_info["key_properties"] = important_props
            candidates_info.append(candidate_info)
        
        # Create biomedical property-based reasoning prompt
        llm_prompt = self._create_llm_prompt(entity, candidates_info, context)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You select the most relevant biomedical knowledge graph entity based on property matching and fertility context. Focus on reproductive medicine relevance."
                    },
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            parsed = EmbeddingModel.safe_json_parse(content)
            
            # Extract selected entity from LLM response
            selected_candidate = self._parse_llm_response(parsed, candidates)
            
            if selected_candidate:
                self._add_message(f"✅ Stage 3: LLM selected candidate: {selected_candidate['node_name']}")
                return selected_candidate
            else:
                self._add_message("⚠️ Stage 3: LLM did not select a valid candidate")
                
        except Exception as e:
            self._add_message(f"❌ Error in LLM property matching: {e}")
        
        return None
    
    def _create_llm_prompt(self, entity: Dict, candidates_info: List[Dict], context: str) -> str:
        """Create LLM prompt for biomedical entity matching"""
        return f"""
        You are a biomedical knowledge graph expert specializing in fertility medicine. 
        Help map a query entity to the most relevant knowledge graph node.
        
        CONTEXT:
        - Context: "{context}"
        - Query Entity: "{entity['name']}" (Type: {entity.get('type', 'Unknown')})
        - Entity Confidence: {entity.get('confidence', 0.7):.2f}
        - Domain: Fertility/Reproductive Medicine
        
        CANDIDATE NODES (with key biomedical properties):
        {json.dumps(candidates_info, indent=2)}
        
        IMPORTANT PROPERTY MAPPINGS FOR FERTILITY ENTITIES:
        
        1. DISEASE NODES:
           - Primary keys: node_id, node_name
           - Key properties to check: 
             * mondo_definitions: Medical definitions from MONDO ontology
             * mayo_symptoms: Clinical symptoms from Mayo Clinic
             * orphanet_prevalence: Prevalence in population
             * SNOMEDCT_US_definition: Standard medical terminology definitions
             * orphanet_definition: Rare disease definitions
        
        2. DRUG NODES:
           - Primary keys: node_id, node_name  
           - Key properties to check:
             * description: General description of the drug
             * indication: What conditions it treats
             * mechanism_of_action: How it works biologically
             * category: Drug classification/category
             * atc_4: Anatomical Therapeutic Chemical classification
        
        ANALYSIS INSTRUCTIONS FOR FERTILITY CONTEXT:
        1. Check if query entity matches any PRIMARY KEY values exactly
        2. Check if query entity appears in DISPLAY PROPERTY values
        3. Evaluate FERTILITY RELEVANCE - does the entity relate to reproduction?
        4. Consider if properties mention fertility-related terms:
           - For diseases: infertility, pregnancy, menstrual, ovarian, sperm, embryo
           - For drugs: fertility treatment, ovulation induction, hormone therapy
        5. Evaluate the enhanced score (combines similarity + property matching)
        6. If NO candidate is suitable, select NONE
        
        OUTPUT FORMAT:
        {{
          "selected_entity": {{
            "index": selected_index_or_-1_for_none,
            "node_name": "selected_node_name_or_NONE",
            "matching_properties": ["property1", "property2"],
            "reason": "brief_explanation_focusing_on_property_matches_and_fertility_relevance"
          }}
        }}
        """
    
    def _parse_llm_response(self, parsed: Dict, candidates: List[Dict]) -> Optional[Dict]:
        """Parse LLM response and extract selected candidate"""
        if not parsed or not isinstance(parsed, dict):
            return None
        
        selected = parsed.get("selected_entity", {})
        if not isinstance(selected, dict):
            return None
        
        selected_index = selected.get("index", -1)
        if 0 <= selected_index < len(candidates):
            candidate = candidates[selected_index]
            
            # Add LLM metadata to candidate
            candidate["llm_property_matches"] = selected.get("matching_properties", [])
            candidate["llm_reason"] = selected.get("reason", "LLM selected based on biomedical property context")
            
            return candidate
        
        return None
    
    def map_entity_to_kg(
        self, 
        entity: Dict, 
        kg_nodes: List[Dict], 
        kg_embeddings: np.ndarray,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Enhanced entity mapping using property-aware pipeline.
        
        Args:
            entity: Query entity dictionary
            kg_nodes: List of KG node dictionaries
            kg_embeddings: Numpy array of KG node embeddings
            context: Optional context/question for better matching
            
        Returns:
            Mapping result dictionary
        """
        self.clear_messages()
        
        entity_name = entity["name"]
        entity_type = entity.get("type", "Unknown")
        
        self._add_message(f"🔍 Starting entity mapping for: '{entity_name}' (type: {entity_type})")
        
        # Check if similarity calculator is available
        if not self.similarity_calculator:
            self._add_message("❌ Similarity calculator not available for entity mapping")
            return self._create_mapping_result(entity, None, "error", "Similarity calculator not initialized")
        
        # Stage 0: Get property-enhanced candidates
        candidates, messages = self.similarity_calculator.get_enhanced_candidates(
            entity_name=entity_name,
            entity_type=entity_type,
            kg_nodes=kg_nodes,
            kg_embeddings=kg_embeddings,
            top_k=self.top_k
        )
        
        self.messages.extend(messages)
        
        if not candidates:
            self._add_message("❌ No similar nodes found in knowledge graph")
            return self._create_mapping_result(
                entity, None, "no_candidates", "No similar nodes found in knowledge graph"
            )
        
        self._add_message(f"📊 Found {len(candidates)} candidate nodes")
        
        # Stage 1: Property-based exact match
        exact_match = self.stage1_property_exact_match(entity_name, entity_type, candidates)
        if exact_match:
            return self._create_mapping_result(
                entity, exact_match, "property_exact", 
                f"Property-based exact match: {exact_match.get('property_match_type', 'unknown')}"
            )
        
        # Stage 2: Property-enhanced similarity
        similarity_match = self.stage2_property_enhanced_similarity(candidates)
        if similarity_match:
            return self._create_mapping_result(
                entity, similarity_match, "property_enhanced",
                f"Property-enhanced similarity ({similarity_match.get('match_type', 'unknown')}, "
                f"score: {similarity_match.get('enhanced_score', 0):.4f})"
            )
        
        # Stage 3: LLM with property context
        llm_match = self.stage3_property_contextual_llm_match(entity, candidates, context)
        if llm_match:
            return self._create_mapping_result(
                entity, llm_match, "property_llm",
                llm_match.get("llm_reason", "LLM selected based on property context"),
                llm_metadata={
                    "property_matches": llm_match.get("llm_property_matches", []),
                    "reason": llm_match.get("llm_reason", "")
                }
            )
        
        # Fallback: Return best candidate
        best_candidate = candidates[0]
        self._add_message(f"⚠️ No strong matches found. Using best candidate: {best_candidate['node_name']}")
        
        return self._create_mapping_result(
            entity, best_candidate, "property_fallback",
            f"No strong property match found. Returning best enhanced match: {best_candidate['node_name']}"
        )
    
    def _create_mapping_result(
        self, 
        entity: Dict, 
        candidate: Optional[Dict], 
        mapping_stage: str, 
        reason: str,
        llm_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a standardized mapping result dictionary"""
        result = {
            "query_entity": entity,
            "mapping_stage": mapping_stage,
            "reason": reason,
            "mapping_timestamp": self._get_timestamp(),
            "messages": self.messages.copy()
        }
        
        if candidate:
            result["mapped_node"] = {
                "node_id": candidate["node_id"],
                "node_name": candidate["node_name"],
                "entity_type": candidate.get("entity_type", "Unknown"),
                "labels": candidate.get("labels", []),
                "similarity_score": candidate.get("similarity_score", 0),
                "enhanced_score": candidate.get("enhanced_score", 0),
                "match_type": candidate.get("match_type", "unknown"),
                "property_match_type": candidate.get("property_match_type", ""),
                "mapping_stage": mapping_stage
            }
            
            # Add property info if available
            if "property_info" in candidate:
                result["property_info"] = candidate["property_info"]
            
            # Add LLM metadata if available
            if llm_metadata:
                result["llm_metadata"] = llm_metadata
                result["mapped_node"]["llm_property_matches"] = llm_metadata.get("property_matches", [])
                result["mapped_node"]["llm_reason"] = llm_metadata.get("reason", "")
        else:
            result["mapped_node"] = None
            result["property_info"] = {}
        
        return result
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_map_entities(
        self, 
        entities: List[Dict], 
        kg_nodes: List[Dict], 
        kg_embeddings: np.ndarray,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Map multiple entities to KG in batch.
        
        Args:
            entities: List of query entity dictionaries
            kg_nodes: List of KG node dictionaries
            kg_embeddings: Numpy array of KG node embeddings
            context: Optional context/question for better matching
            
        Returns:
            Batch mapping results dictionary
        """
        self.clear_messages()
        
        batch_results = {
            "mappings": [],
            "statistics": {
                "total_entities": len(entities),
                "successful_mappings": 0,
                "failed_mappings": 0,
                "mapping_stages": {},
                "average_score": 0.0
            },
            "context": context,
            "timestamp": self._get_timestamp()
        }
        
        all_scores = []
        
        for i, entity in enumerate(entities):
            self._add_message(f"Mapping entity {i+1}/{len(entities)}: '{entity['name']}'")
            
            mapping_result = self.map_entity_to_kg(
                entity=entity,
                kg_nodes=kg_nodes,
                kg_embeddings=kg_embeddings,
                context=context
            )
            
            batch_results["mappings"].append(mapping_result)
            
            # Update statistics
            if mapping_result["mapped_node"]:
                batch_results["statistics"]["successful_mappings"] += 1
                all_scores.append(mapping_result["mapped_node"].get("enhanced_score", 0))
                
                # Track mapping stages
                stage = mapping_result["mapping_stage"]
                batch_results["statistics"]["mapping_stages"][stage] = \
                    batch_results["statistics"]["mapping_stages"].get(stage, 0) + 1
            else:
                batch_results["statistics"]["failed_mappings"] += 1
        
        # Calculate average score
        if all_scores:
            batch_results["statistics"]["average_score"] = sum(all_scores) / len(all_scores)
        
        self._add_message(f"✅ Batch mapping completed: "
                         f"{batch_results['statistics']['successful_mappings']}/"
                         f"{batch_results['statistics']['total_entities']} successful mappings")
        
        batch_results["messages"] = self.messages.copy()
        
        return batch_results
    
    def validate_mapping_result(self, mapping_result: Dict) -> Dict[str, Any]:
        """
        Validate a mapping result.
        
        Args:
            mapping_result: Mapping result dictionary
            
        Returns:
            Validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "scores": {}
        }
        
        # Check required fields
        required_fields = ["query_entity", "mapping_stage", "reason"]
        for field in required_fields:
            if field not in mapping_result:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Missing required field: {field}")
        
        # Check query entity
        query_entity = mapping_result.get("query_entity", {})
        if not query_entity or "name" not in query_entity:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Invalid or missing query entity")
        
        # Check mapped node if present
        mapped_node = mapping_result.get("mapped_node")
        if mapped_node:
            required_node_fields = ["node_id", "node_name", "entity_type"]
            for field in required_node_fields:
                if field not in mapped_node:
                    validation_results["warnings"].append(f"Mapped node missing field: {field}")
            
            # Check scores
            enhanced_score = mapped_node.get("enhanced_score", -1)
            if not (0 <= enhanced_score <= 1):
                validation_results["warnings"].append(f"Enhanced score out of range: {enhanced_score}")
            
            validation_results["scores"]["enhanced_score"] = enhanced_score
            validation_results["scores"]["similarity_score"] = mapped_node.get("similarity_score", 0)
        
        # Check mapping stage
        valid_stages = [
            "property_exact", "property_enhanced", "property_llm", 
            "property_fallback", "no_candidates", "error"
        ]
        mapping_stage = mapping_result.get("mapping_stage", "")
        if mapping_stage not in valid_stages:
            validation_results["warnings"].append(f"Unknown mapping stage: {mapping_stage}")
        
        return validation_results
    
    def get_mapping_statistics(self, mapping_results: List[Dict]) -> Dict[str, Any]:
        """
        Get statistics from mapping results.
        
        Args:
            mapping_results: List of mapping result dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not mapping_results:
            return {"total_mappings": 0, "success_rate": 0.0}
        
        total = len(mapping_results)
        successful = sum(1 for r in mapping_results if r.get("mapped_node"))
        
        # Collect scores
        scores = []
        for result in mapping_results:
            if result.get("mapped_node"):
                scores.append(result["mapped_node"].get("enhanced_score", 0))
        
        # Count mapping stages
        stage_counts = {}
        for result in mapping_results:
            stage = result.get("mapping_stage", "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            "total_mappings": total,
            "successful_mappings": successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "stage_distribution": stage_counts,
            "score_range": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "average": sum(scores) / len(scores) if scores else 0
            }
        }