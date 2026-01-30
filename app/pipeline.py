# pipeline.py

import time
import json
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from config import Config
from extraction import EntityExtractor
from similarity import SimilarityCalculator
from path_finder import PathFinder
from cot import ChainOfThoughtGenerator

class BiomedicalPipeline:
    """
    Complete biomedical pipeline with property-aware entity mapping, path pruning, and CoT generation.
    Fertility-focused clinical reasoning system.
    """
    
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        similarity_calculator: Optional[SimilarityCalculator] = None,
        path_finder: Optional[PathFinder] = None,
        cot_generator: Optional[ChainOfThoughtGenerator] = None
    ):
        """
        Initialize the biomedical pipeline.
        
        Args:
            entity_extractor: Entity extraction component
            similarity_calculator: Similarity calculation component
            path_finder: Path finding component
            cot_generator: Chain-of-Thought generation component
        """
        self.config = Config
        self.messages = []
        
        # Initialize components
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        self.path_finder = path_finder or PathFinder()
        self.cot_generator = cot_generator or ChainOfThoughtGenerator()
        
        # Pipeline configuration
        self.top_k = self.config.PIPELINE.TOP_K
        self.sim_threshold = self.config.PIPELINE.SIM_THRESHOLD
        self.exact_match_threshold = self.config.PIPELINE.EXACT_MATCH_THRESHOLD
        self.max_path_length = self.config.PIPELINE.MAX_PATH_LENGTH
        self.max_paths_per_pair = self.config.PIPELINE.MAX_PATHS_PER_PAIR
        self.llm_model = self.config.PIPELINE.LLM_MODEL
        
        # Data store (to be populated externally)
        self.kg_nodes = []
        self.kg_embeddings = None
        
        # Pipeline state
        self.current_question = ""
        self.current_results = {}
    
    def _add_message(self, message: str):
        """Add a message to the messages list"""
        self.messages.append(message)
        print(message)
    
    def clear_messages(self):
        """Clear all stored messages"""
        self.messages = []
    
    def get_messages(self) -> List[str]:
        """Get all stored messages"""
        return self.messages
    
    def set_kg_data(self, kg_nodes: List[Dict], kg_embeddings: Optional[np.ndarray] = None):
        """
        Set knowledge graph data for the pipeline.
        
        Args:
            kg_nodes: List of KG node dictionaries
            kg_embeddings: Optional KG embeddings (numpy array)
        """
        self.kg_nodes = kg_nodes
        self.kg_embeddings = kg_embeddings
        
        # Debug info
        if kg_embeddings is not None:
            self._add_message(f"✅ KG data set: {len(kg_nodes)} nodes, embeddings shape: {kg_embeddings.shape}")
        else:
            self._add_message(f"✅ KG data set: {len(kg_nodes)} nodes, no embeddings")
        
        # Configure similarity calculator with embeddings if available
        if hasattr(self.similarity_calculator, 'set_kg_embeddings') and kg_embeddings is not None:
            self.similarity_calculator.set_kg_embeddings(kg_embeddings)
    
    def set_llm_client(self, client):
        """
        Set LLM client for all components that need it.
        
        Args:
            client: LLM client instance
        """
        # Set for entity extractor
        if hasattr(self.entity_extractor, 'set_llm_client'):
            self.entity_extractor.set_llm_client(client)
        
        # Set for path finder
        if hasattr(self.path_finder, 'set_llm_client'):
            self.path_finder.set_llm_client(client)
        
        # Set for CoT generator
        if hasattr(self.cot_generator, 'set_llm_client'):
            self.cot_generator.set_llm_client(client)
        
        self._add_message("✅ LLM client configured for all pipeline components")
    
    def set_llm_model(self, model_name: str):
        """
        Set LLM model for all components.
        
        Args:
            model_name: Name of the LLM model
        """
        self.llm_model = model_name
        
        # Set for entity extractor
        if hasattr(self.entity_extractor, 'set_llm_model'):
            self.entity_extractor.set_llm_model(model_name)
        
        # Set for path finder
        if hasattr(self.path_finder, 'set_llm_model'):
            self.path_finder.set_llm_model(model_name)
        
        # Set for CoT generator
        if hasattr(self.cot_generator, 'set_llm_model'):
            self.cot_generator.set_llm_model(model_name)
        
        self._add_message(f"✅ LLM model set to: {model_name}")
    
    def is_kg_ready(self) -> bool:
        """
        Check if KG data is properly loaded and ready for use.
        
        Returns:
            True if KG data is ready, False otherwise
        """
        try:
            # Check nodes
            if self.kg_nodes is None or len(self.kg_nodes) == 0:
                return False
            
            # Check embeddings
            if self.kg_embeddings is None:
                return False
            
            # Check if embeddings is a numpy array
            if not isinstance(self.kg_embeddings, np.ndarray):
                return False
            
            # Check shape
            if len(self.kg_embeddings.shape) != 2:
                return False
            
            # Check dimensions match
            if self.kg_embeddings.shape[0] != len(self.kg_nodes):
                self._add_message(f"⚠️ Warning: Embeddings shape {self.kg_embeddings.shape[0]} doesn't match nodes count {len(self.kg_nodes)}")
                # Continue anyway but warn
            
            return True
            
        except Exception as e:
            self._add_message(f"[DEBUG] Error in is_kg_ready: {e}")
            return False
    
    def extract_and_map_entities_with_cot(self, text: str) -> Dict[str, Any]:
        """
        Complete enhanced pipeline with property-aware entity mapping, 
        path pruning, and CoT generation.
        
        Args:
            text: Clinical question text
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.clear_messages()
        self.current_question = text
        
        self._add_message(f"\n🔍 Processing clinical question: '{text[:100]}...'" 
                         if len(text) > 100 else f"\n🔍 Processing clinical question: '{text}'")
        
        # Step 1: Extract fertility-related biomedical entities from text
        self._add_message("📝 Extracting fertility-related biomedical entities from text...")
        extracted_entities = self._extract_entities(text)
        
        if not extracted_entities:
            self._add_message("❌ No entities extracted from text")
            return self._create_empty_result(text, "No entities extracted")
        
        # Step 2: Map each entity to KG with property enhancement
        self._add_message("\n🗺️  Mapping clinical entities to biomedical knowledge graph...")
        mapped_results = self._map_entities_to_kg(extracted_entities)
        
        # Step 3: Perform biomedical path search between mapped entities
        self._add_message("\n🛤️  Searching biomedical pathways between mapped entities...")
        path_results = self._find_paths_between_entities(mapped_results)
        
        # Step 4: Prune paths using LLM to select the most clinically relevant one
        if path_results.get("all_paths"):
            self._add_message("\n✂️  Selecting most clinically relevant path...")
            path_pruning_result = self._prune_paths(text, path_results["all_paths"], mapped_results)
        else:
            self._add_message("\n⚠️  Skipping path pruning - no paths available")
            path_pruning_result = {"selected_path": None, "reasoning": "No paths found"}
        
        # Step 5: Generate clinical Chain-of-Thought reasoning
        self._add_message("\n🧠 Generating clinical Chain-of-Thought reasoning...")
        selected_path = path_pruning_result.get("selected_path")
        cot_result = self._generate_chain_of_thought(text, selected_path, mapped_results)
        
        # Step 6: Generate comprehensive output
        output = self._generate_comprehensive_output(
            text, extracted_entities, mapped_results, 
            path_results, path_pruning_result, cot_result
        )
        
        self.current_results = output
        self._display_results_summary(output)
        
        return output
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from clinical text."""
        results = self.entity_extractor.extract_with_details(text)
        entities = results.get("entities", [])
        
        self._add_message(f"✅ Extracted {len(entities)} clinical entities:")
        for entity in entities:
            conf = entity.get('confidence', 0.7)
            entity_type = entity.get('type', 'Unknown')
            self._add_message(f"   - {entity['name']} ({entity_type}) [Confidence: {conf:.2f}]")
        
        return entities
    
    def _map_entities_to_kg(self, extracted_entities: List[Dict]) -> List[Dict]:
        """Map extracted entities to knowledge graph nodes."""
        
        # CRITICAL FIX: Properly check for None instead of using "not" on numpy array
        if not self.kg_nodes or self.kg_embeddings is None:
            self._add_message("⚠️ KG data not available for mapping")
            self._add_message(f"   kg_nodes: {'Available' if self.kg_nodes else 'None'}, "
                           f"kg_embeddings: {'Available' if self.kg_embeddings is not None else 'None'}")
            return []
        
        # Additional safety check for numpy array
        if not isinstance(self.kg_embeddings, np.ndarray):
            self._add_message("⚠️ KG embeddings not properly initialized (not a numpy array)")
            return []
        
        # Check dimensions
        if len(self.kg_embeddings.shape) != 2:
            self._add_message(f"⚠️ KG embeddings have incorrect shape: {self.kg_embeddings.shape}")
            return []
        
        if len(self.kg_embeddings) == 0:
            self._add_message("⚠️ KG embeddings array is empty")
            return []
        
        # Ensure similarity calculator has access to embeddings
        if hasattr(self.similarity_calculator, 'set_kg_embeddings'):
            self.similarity_calculator.set_kg_embeddings(self.kg_embeddings)
        
        mapped_results = []
        
        for entity in extracted_entities:
            entity_name = entity['name']
            entity_type = entity.get('type', 'Unknown')
            
            self._add_message(f"  🔄 Mapping clinical entity: {entity_name} ({entity_type})...")
            
            try:
                # Get enhanced candidates from KG
                candidates, _ = self.similarity_calculator.get_enhanced_candidates(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    kg_nodes=self.kg_nodes,
                    kg_embeddings=self.kg_embeddings,
                    top_k=self.top_k
                )
                
                mapping_result = {
                    "query_entity": entity,
                    "candidates": candidates,
                    "mapped_node": None,
                    "mapping_stage": "no_candidates",
                    "confidence_score": 0.0,
                    "mapping_details": {}
                }
                
                if candidates:
                    # Select best candidate
                    best_candidate = candidates[0]
                    mapping_result["mapped_node"] = best_candidate
                    mapping_result["confidence_score"] = best_candidate.get('enhanced_score', 0)
                    mapping_result["mapping_stage"] = "property_enhanced"
                    mapping_result["mapping_details"] = {
                        "match_type": best_candidate.get('match_type', 'unknown'),
                        "property_info": best_candidate.get('property_info', {})
                    }
                    
                    # Determine stage emoji and color
                    score = best_candidate.get('enhanced_score', 0)
                    if score >= 0.9:
                        score_color = "\033[92m"  # Green
                    elif score >= 0.7:
                        score_color = "\033[93m"  # Yellow
                    else:
                        score_color = "\033[91m"  # Red
                    
                    stage = mapping_result["mapping_stage"]
                    stage_emoji = {
                        "property_exact": "🎯",
                        "property_enhanced": "📊",
                        "property_llm": "🤖",
                        "property_fallback": "⚠️",
                        "no_candidates": "❌"
                    }.get(stage, "🔍")
                    
                    self._add_message(f"    {stage_emoji} {stage}: {entity_name} → "
                                    f"\033[1m{best_candidate['node_name']}\033[0m "
                                    f"(Score: {score_color}{score:.3f}\033[0m, "
                                    f"Match: {best_candidate.get('match_type', 'unknown')})")
                else:
                    self._add_message(f"    ❌ No biomedical mapping found for {entity_name}")
                
                mapped_results.append(mapping_result)
                
            except Exception as e:
                self._add_message(f"    ❌ Error mapping {entity_name}: {str(e)}")
                mapping_result = {
                    "query_entity": entity,
                    "candidates": [],
                    "mapped_node": None,
                    "mapping_stage": "error",
                    "confidence_score": 0.0,
                    "mapping_details": {"error": str(e)},
                    "error": str(e)
                }
                mapped_results.append(mapping_result)
        
        # Summary of mapping results
        successful = [m for m in mapped_results if m.get("mapped_node")]
        self._add_message(f"📊 Mapping summary: {len(successful)}/{len(extracted_entities)} entities successfully mapped")
        
        return mapped_results
    
    def _find_paths_between_entities(self, mapped_results: List[Dict]) -> Dict[str, Any]:
        """Find paths between mapped entities."""
        # Filter to successfully mapped nodes
        successful_mappings = [
            m for m in mapped_results 
            if m.get("mapped_node") and m["mapped_node"].get("node_id")
        ]
        
        if len(successful_mappings) < 2:
            self._add_message("❌ Need at least 2 mapped entities to find paths")
            return {
                "entity_pairs": [],
                "total_paths_found": 0,
                "all_paths": []
            }
        
        # Use path finder to get paths
        try:
            path_results = self.path_finder.find_shortest_paths_between_entities(
                successful_mappings, 
                self.max_paths_per_pair
            )
            
            # Display summary
            if path_results["entity_pairs"]:
                self._add_message(f"✅ Found paths for {len(path_results['entity_pairs'])} entity pairs")
                for pair in path_results["entity_pairs"][:2]:  # Show first 2 pairs
                    self._add_message(f"   • {pair['source_entity']} → {pair['target_entity']}: "
                                    f"{pair['paths_found']} paths")
            else:
                self._add_message("❌ No biomedical pathways found between mapped entities")
            
            return path_results
            
        except Exception as e:
            self._add_message(f"❌ Error finding paths: {str(e)}")
            return {
                "entity_pairs": [],
                "total_paths_found": 0,
                "all_paths": [],
                "error": str(e)
            }
    
    def _prune_paths(
        self, 
        question: str, 
        all_paths: List[Dict], 
        entity_mappings: List[Dict]
    ) -> Dict[str, Any]:
        """Prune paths to select the most clinically relevant one."""
        if not all_paths:
            return {"selected_path": None, "reasoning": "No paths available"}
        
        # Use path finder's LLM pruning if available
        if hasattr(self.path_finder, 'prune_paths_with_llm'):
            try:
                pruning_result = self.path_finder.prune_paths_with_llm(
                    question, all_paths, entity_mappings
                )
                return pruning_result
            except Exception as e:
                self._add_message(f"⚠️ LLM path pruning failed: {str(e)}")
                # Fall through to fallback
        
        # Fallback: select shortest path
        try:
            shortest_path = min(all_paths, key=lambda x: x.get('path_length', 100))
            pruning_result = {
                "selected_path": shortest_path,
                "reasoning": "Selected shortest path as fallback",
                "confidence": 0.5
            }
            return pruning_result
        except Exception as e:
            self._add_message(f"⚠️ Fallback path selection failed: {str(e)}")
            return {"selected_path": None, "reasoning": f"Error selecting path: {str(e)}"}
    
    def _generate_chain_of_thought(
        self, 
        question: str, 
        selected_path: Dict, 
        entity_mappings: List[Dict]
    ) -> Dict[str, Any]:
        """Generate Chain-of-Thought reasoning."""
        if not selected_path:
            return {
                "chain_of_thought": {
                    "reasoning_steps": ["No relevant path found for reasoning"],
                    "detailed_reasoning": "Cannot generate clinical reasoning without a valid biomedical knowledge graph path.",
                    "final_answer": "Cannot answer based on available biomedical knowledge graph information.",
                    "confidence_score": 0.0
                }
            }
        
        # Use CoT generator
        try:
            cot_result = self.cot_generator.generate_chain_of_thought(
                question, selected_path, entity_mappings
            )
            return cot_result
        except Exception as e:
            self._add_message(f"❌ Error generating CoT: {str(e)}")
            return {
                "chain_of_thought": {
                    "reasoning_steps": [f"Error generating reasoning: {str(e)}"],
                    "detailed_reasoning": f"Failed to generate clinical reasoning due to error: {str(e)}",
                    "final_answer": "Unable to generate clinical answer due to system error.",
                    "confidence_score": 0.0
                }
            }
    
    def _generate_comprehensive_output(
        self,
        question: str,
        extracted_entities: List[Dict],
        mapped_results: List[Dict],
        path_results: Dict,
        path_pruning_result: Dict,
        cot_result: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive output with all pipeline results."""
        
        # Calculate statistics
        successful_mappings = [m for m in mapped_results if m.get("mapped_node")]
        total_entities = len(extracted_entities)
        success_rate = (len(successful_mappings) / total_entities * 100) if total_entities > 0 else 0
        
        # Calculate confidence levels
        confidence_levels = {"high": 0, "medium": 0, "low": 0}
        for mapping in successful_mappings:
            score = mapping["mapped_node"].get("enhanced_score", 0)
            if score >= 0.8:
                confidence_levels["high"] += 1
            elif score >= 0.6:
                confidence_levels["medium"] += 1
            else:
                confidence_levels["low"] += 1
        
        # Determine fertility relevance
        fertility_keywords = ["fertility", "infertility", "reproductive", "pregnancy", 
                            "ovarian", "uterine", "sperm", "egg", "embryo"]
        question_lower = question.lower()
        fertility_relevant = any(kw in question_lower for kw in fertility_keywords)
        
        # Create comprehensive output
        output = {
            "clinical_question": question,
            "timestamp": datetime.now().isoformat(),
            
            # Entity extraction results
            "entity_extraction": {
                "extracted_entities": extracted_entities,
                "total_entities": total_entities,
                "statistics": {
                    "disease_count": sum(1 for e in extracted_entities if e.get('type') == 'disease'),
                    "drug_count": sum(1 for e in extracted_entities if e.get('type') == 'drug'),
                    "average_confidence": sum(e.get('confidence', 0) for e in extracted_entities) / total_entities if total_entities > 0 else 0
                }
            },
            
            # Entity mapping results
            "entity_mapping": {
                "mapped_results": mapped_results,
                "successful_mappings": successful_mappings,
                "total_mapped": len(successful_mappings),
                "success_rate": success_rate,
                "confidence_distribution": confidence_levels,
                "statistics": {
                    "high_confidence": confidence_levels["high"],
                    "medium_confidence": confidence_levels["medium"],
                    "low_confidence": confidence_levels["low"]
                }
            },
            
            # Path discovery results
            "path_discovery": {
                "path_results": path_results,
                "path_pruning_result": path_pruning_result,
                "selected_path": path_pruning_result.get("selected_path"),
                "selected_path_confidence": path_pruning_result.get("confidence", 0),
                "statistics": {
                    "entity_pairs_analyzed": len(path_results.get("entity_pairs", [])),
                    "total_paths_found": path_results.get("total_paths_found", 0),
                    "pairs_with_paths": sum(1 for p in path_results.get("entity_pairs", []) if p.get("paths_found", 0) > 0)
                }
            },
            
            # Chain-of-Thought results
            "chain_of_thought": cot_result,
            
            # Clinical context
            "clinical_context": {
                "domain": "Biomedical/Fertility",
                "fertility_relevance": {
                    "is_fertility_question": fertility_relevant,
                    "fertility_keywords_present": [kw for kw in fertility_keywords if kw in question_lower],
                    "extracted_fertility_entities": sum(1 for e in extracted_entities 
                                                      if any(kw in e['name'].lower() for kw in 
                                                            ["pcos", "endometriosis", "infertility", "clomiphene", "ivf"]))
                },
                "biomedical_kg_info": {
                    "total_nodes": len(self.kg_nodes),
                    "entity_types_focused": self.config.ENTITY.TYPES,
                    "property_schemas_used": list(self.config.ENTITY.PROPERTY_SCHEMAS.keys())
                }
            },
            
            # Pipeline configuration
            "pipeline_configuration": {
                "top_k": self.top_k,
                "similarity_threshold": self.sim_threshold,
                "exact_match_threshold": self.exact_match_threshold,
                "max_path_length": self.max_path_length,
                "max_paths_per_pair": self.max_paths_per_pair,
                "llm_model": self.llm_model,
                "property_weights": self.config.ENTITY.PROPERTY_WEIGHTS
            },
            
            # Summary statistics
            "summary": {
                "overall_status": "success" if len(successful_mappings) > 0 else "partial" if extracted_entities else "failed",
                "entities_processed": total_entities,
                "entities_successfully_mapped": len(successful_mappings),
                "paths_found": path_results.get("total_paths_found", 0),
                "cot_generated": bool(cot_result),
                "cot_confidence": cot_result.get("chain_of_thought", {}).get("confidence_score", 0) if cot_result else 0,
                "final_answer_available": bool(cot_result and cot_result.get("chain_of_thought", {}).get("final_answer"))
            }
        }
        
        return output
    
    def _display_results_summary(self, output: Dict[str, Any]):
        """Display results summary in formatted way."""
        
        print("\n" + "="*80)
        print("CLINICAL ANALYSIS RESULTS")
        print("="*80)
        
        # Entity mapping summary
        entity_mapping = output["entity_mapping"]
        print(f"\n📊 Clinical Entity Mapping Summary:")
        print(f"   Successfully mapped: {entity_mapping['total_mapped']}/{len(output['entity_extraction']['extracted_entities'])} "
              f"entities ({entity_mapping['success_rate']:.1f}%)")
        
        if entity_mapping['successful_mappings']:
            stats = entity_mapping["statistics"]
            print(f"   Mapping confidence levels:")
            print(f"     • High (≥0.8): {stats['high_confidence']} entities")
            print(f"     • Medium (0.6-0.8): {stats['medium_confidence']} entities")
            print(f"     • Low (<0.6): {stats['low_confidence']} entities")
        
        # Path analysis
        print(f"\n🛤️  Biomedical Pathway Analysis:")
        selected_path = output["path_discovery"]["selected_path"]
        if selected_path:
            path_desc = selected_path.get('detailed_description', selected_path.get('path_string', 'N/A'))
            print(f"   Selected clinical pathway: {path_desc}")
            print(f"   Pathway length: {selected_path.get('path_length', 'N/A')} biological relationships")
            
            confidence = output["path_discovery"]["selected_path_confidence"]
            confidence_color = "\033[92m" if confidence >= 0.8 else "\033[93m" if confidence >= 0.6 else "\033[91m"
            print(f"   Clinical relevance confidence: {confidence_color}{confidence:.2f}\033[0m")
            
            reasoning = output["path_discovery"]["path_pruning_result"].get('reasoning', 'N/A')
            if len(reasoning) > 120:
                reasoning = reasoning[:120] + "..."
            print(f"   Selection rationale: {reasoning}")
        else:
            print("   ⚠️  No clinically relevant pathway selected")
        
        # Clinical reasoning
        print(f"\n🧠 Clinical Reasoning Chain:")
        cot_result = output.get("chain_of_thought", {})
        if cot_result and "chain_of_thought" in cot_result:
            cot = cot_result["chain_of_thought"]
            
            # Show reasoning steps
            steps = cot.get('reasoning_steps', [])
            if steps:
                print(f"   Reasoning steps ({len(steps)} total):")
                for i, step in enumerate(steps[:3]):
                    step_text = step.replace("Step X:", "").replace("Step X: ", "").strip()
                    if step_text:
                        print(f"     {i+1}. {step_text[:80]}...")
                if len(steps) > 3:
                    print(f"     ... and {len(steps) - 3} more steps")
            
            # Show final clinical answer
            final_answer = cot.get('final_answer', 'N/A')
            print(f"\n   📋 Clinical Answer:")
            print(f"   \"{final_answer}\"")
        else:
            print("   ⚠️  No clinical reasoning generated")
        
        # Biomedical knowledge used
        print(f"\n📚 Biomedical Knowledge Used:")
        print(f"   • Knowledge graph nodes: {len(self.kg_nodes)}")
        print(f"   • Entity types focused: {', '.join(self.config.ENTITY.TYPES)}")
        
        # Fertility relevance
        fertility_info = output["clinical_context"]["fertility_relevance"]
        if fertility_info["is_fertility_question"]:
            print(f"\n🎯 Fertility Relevance:")
            print(f"   • Fertility keywords detected: {', '.join(fertility_info['fertility_keywords_present'][:3])}")
            print(f"   • Fertility entities extracted: {fertility_info['extracted_fertility_entities']}")
    
    def export_results(self, format: str = "json", include_details: bool = True) -> Any:
        """
        Export pipeline results in specified format.
        
        Args:
            format: Export format ("json", "markdown", "summary")
            include_details: Whether to include detailed information
            
        Returns:
            Exported results in requested format
        """
        if not self.current_results:
            return None
        
        if format == "json":
            if include_details:
                return json.dumps(self.current_results, indent=2, default=str)
            else:
                # Return summary only
                summary = {
                    "question": self.current_results.get("clinical_question"),
                    "summary": self.current_results.get("summary", {}),
                    "final_answer": self.current_results.get("chain_of_thought", {}).get("chain_of_thought", {}).get("final_answer", "")
                }
                return json.dumps(summary, indent=2)
        
        elif format == "markdown":
            if hasattr(self.cot_generator, 'export_cot_to_markdown'):
                return self.cot_generator.export_cot_to_markdown(self.current_results)
            else:
                # Generate basic markdown
                lines = []
                lines.append("# Biomedical Pipeline Results")
                lines.append(f"**Question:** {self.current_results.get('clinical_question', 'N/A')}")
                lines.append(f"**Timestamp:** {self.current_results.get('timestamp', 'N/A')}")
                
                summary = self.current_results.get("summary", {})
                lines.append(f"\n## Summary")
                lines.append(f"- Status: {summary.get('overall_status', 'unknown')}")
                lines.append(f"- Entities Processed: {summary.get('entities_processed', 0)}")
                lines.append(f"- Entities Mapped: {summary.get('entities_successfully_mapped', 0)}")
                lines.append(f"- Paths Found: {summary.get('paths_found', 0)}")
                
                cot = self.current_results.get("chain_of_thought", {}).get("chain_of_thought", {})
                if cot.get("final_answer"):
                    lines.append(f"\n## Final Answer")
                    lines.append(f"{cot['final_answer']}")
                
                return "\n".join(lines)
        
        elif format == "summary":
            summary = self.current_results.get("summary", {})
            return {
                "question": self.current_results.get("clinical_question"),
                "status": summary.get("overall_status"),
                "entities_mapped": f"{summary.get('entities_successfully_mapped', 0)}/{summary.get('entities_processed', 0)}",
                "paths_found": summary.get("paths_found", 0),
                "has_final_answer": summary.get("final_answer_available", False),
                "cot_confidence": summary.get("cot_confidence", 0)
            }
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate pipeline components and configuration.
        
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "components": {},
            "issues": [],
            "warnings": []
        }
        
        # Check KG data
        if not self.kg_nodes:
            validation["is_valid"] = False
            validation["issues"].append("No KG nodes loaded")
        
        # Check entity extractor
        if self.entity_extractor:
            validation["components"]["entity_extractor"] = "✅ Available"
        else:
            validation["warnings"].append("Entity extractor not initialized")
        
        # Check similarity calculator
        if self.similarity_calculator:
            validation["components"]["similarity_calculator"] = "✅ Available"
        else:
            validation["warnings"].append("Similarity calculator not initialized")
        
        # Check path finder
        if self.path_finder:
            validation["components"]["path_finder"] = "✅ Available"
        else:
            validation["warnings"].append("Path finder not initialized")
        
        # Check CoT generator
        if self.cot_generator:
            validation["components"]["cot_generator"] = "✅ Available"
        else:
            validation["warnings"].append("CoT generator not initialized")
        
        # Check embeddings
        if not self.kg_embeddings:
            validation["warnings"].append("KG embeddings not loaded (may affect similarity)")
        elif not isinstance(self.kg_embeddings, np.ndarray):
            validation["warnings"].append("KG embeddings is not a numpy array")
        elif len(self.kg_embeddings.shape) != 2:
            validation["warnings"].append(f"KG embeddings has incorrect shape: {self.kg_embeddings.shape}")
        
        return validation
    
    def run_batch(
        self, 
        questions: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on multiple questions.
        
        Args:
            questions: List of clinical questions
            show_progress: Whether to show progress messages
            
        Returns:
            List of pipeline results for each question
        """
        all_results = []
        
        for i, question in enumerate(questions):
            if show_progress:
                print(f"\n📋 Processing question {i+1}/{len(questions)}: '{question[:50]}...'")
            
            try:
                result = self.extract_and_map_entities_with_cot(question)
                all_results.append(result)
                
                if show_progress:
                    summary = result.get("summary", {})
                    print(f"   ✅ Completed: {summary.get('entities_successfully_mapped', 0)} entities mapped, "
                          f"{summary.get('paths_found', 0)} paths found")
            
            except Exception as e:
                print(f"   ❌ Error processing question: {e}")
                all_results.append({
                    "clinical_question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return all_results
    
    def _create_empty_result(self, text: str, reason: str) -> Dict[str, Any]:
        """Create empty result structure."""
        return {
            "clinical_question": text,
            "timestamp": datetime.now().isoformat(),
            "error": reason,
            "entity_extraction": {
                "extracted_entities": [],
                "total_entities": 0,
                "statistics": {}
            },
            "entity_mapping": {
                "mapped_results": [],
                "successful_mappings": [],
                "total_mapped": 0,
                "success_rate": 0,
                "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
                "statistics": {}
            },
            "path_discovery": {
                "path_results": {"entity_pairs": [], "total_paths_found": 0, "all_paths": []},
                "path_pruning_result": {"selected_path": None, "reasoning": reason},
                "selected_path": None,
                "selected_path_confidence": 0,
                "statistics": {}
            },
            "chain_of_thought": {
                "chain_of_thought": {
                    "reasoning_steps": [f"No reasoning generated: {reason}"],
                    "detailed_reasoning": f"Cannot generate clinical reasoning: {reason}",
                    "final_answer": f"Cannot answer: {reason}",
                    "confidence_score": 0.0
                }
            },
            "summary": {
                "overall_status": "failed",
                "entities_processed": 0,
                "entities_successfully_mapped": 0,
                "paths_found": 0,
                "cot_generated": False,
                "cot_confidence": 0,
                "final_answer_available": False
            }
        }