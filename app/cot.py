# cot.py
import json
from typing import List, Dict, Optional, Any
from config import Config

class ChainOfThoughtGenerator:
    """
    Generate Chain-of-Thought reasoning based on biomedical knowledge graph paths.
    Supports custom LLM models like Llama.
    """
    
    def __init__(self, llm_model: Optional[str] = None):
        """
        Initialize the Chain-of-Thought Generator.
        
        Args:
            llm_model: Name of the LLM model to use (defaults to config)
        """
        self.llm_model = llm_model or Config.PIPELINE.LLM_MODEL
        self.client = None
        self.messages = []
        
        # Relationship translations for biomedical context
        self.relationship_translations = {
            "ASSOCIATED_WITH": "is clinically associated with",
            "TREATS": "is used to treat",
            "CAUSES": "can cause or contribute to",
            "INTERACTS_WITH": "biologically interacts with",
            "REGULATES": "regulates or modulates",
            "TARGETS": "targets or affects",
            "INDUCES": "induces or triggers",
            "INHIBITS": "inhibits or blocks",
            "ENHANCES": "enhances or increases",
            "PREDISPOSES": "predisposes to or increases risk of",
            "PART_OF": "is part of biological pathway",
            "ENCODES": "encodes or produces",
            "METABOLIZES": "metabolizes or breaks down",
            "BINDS_TO": "binds to or interacts with",
            "UPREGULATES": "upregulates or increases expression of",
            "DOWNREGULATES": "downregulates or decreases expression of"
        }
        
        # Biomedical property categories
        self.disease_properties = [
            "mondo_definitions", "mayo_symptoms", "orphanet_definition",
            "SNOMEDCT_US_definition", "mayo_causes", "mayo_complications",
            "orphanet_clinical_description", "umls_descriptions"
        ]
        
        self.drug_properties = [
            "description", "indication", "mechanism_of_action",
            "pharmacodynamics", "category", "group", "pathway"
        ]
        
        self.fertility_keywords = [
            "fertility", "infertility", "ovarian", "uterine", "sperm", "egg",
            "embryo", "pregnancy", "reproductive", "menstrual", "ovulation",
            "conception", "implantation", "gestation"
        ]
    
    def set_llm_client(self, client):
        """Set the LLM client for CoT generation"""
        self.client = client
        self._add_message(f"✅ LLM client configured for Chain-of-Thought generation (model: {self.llm_model})")
    
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
    
    def translate_biomedical_relationship(self, rel_type: str) -> str:
        """Translate relationship type to biomedical meaning."""
        return self.relationship_translations.get(
            rel_type, 
            f"has biological relationship: {rel_type.lower().replace('_', ' ')}"
        )
    
    def generate_chain_of_thought(
        self, 
        question: str, 
        selected_path: Dict, 
        entity_mappings: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate Chain-of-Thought reasoning based on the selected biomedical path.
        
        Args:
            question: Clinical question to answer
            selected_path: Selected knowledge graph path
            entity_mappings: Entity mappings from question to KG
            
        Returns:
            Dictionary with CoT reasoning and final answer
        """
        if not self.client:
            self._add_message("⚠️ LLM client not configured, using fallback CoT")
            return self._generate_fallback_biomedical_cot(question, selected_path, entity_mappings)
        
        if not selected_path:
            return {
                "chain_of_thought": {
                    "reasoning_steps": ["No relevant path found for reasoning"],
                    "detailed_reasoning": "Cannot generate clinical reasoning without a valid biomedical knowledge graph path.",
                    "final_answer": "Cannot answer based on available biomedical knowledge graph information.",
                    "confidence_score": 0.0
                }
            }
        
        # Extract path information
        path_description = selected_path.get("detailed_description", selected_path.get("path_string", ""))
        nodes = selected_path.get("nodes", [])
        relationships = selected_path.get("relationships", [])
        
        # Prepare entity context
        entity_context = self._prepare_entity_context(entity_mappings)
        
        # Prepare path nodes information for biomedical context
        path_nodes_info = self._prepare_path_nodes_info(nodes)
        
        # Prepare relationships information
        relationships_info = self._prepare_relationships_info(relationships, nodes)
        
        # Create CoT generation prompt
        cot_prompt = self._create_cot_prompt(
            question, entity_context, path_description, path_nodes_info, relationships_info
        )
        
        try:
            # Check if using OpenAI format or custom model
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a clinical researcher that generates detailed Chain-of-Thought reasoning based on biomedical knowledge graph paths. Specialize in fertility and reproductive medicine. Be precise and reference specific biological entities and relationships."},
                        {"role": "user", "content": cot_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                content = response.choices[0].message.content
                
            elif hasattr(self.client, 'completions') and hasattr(self.client.completions, 'create'):
                # OpenAI legacy completions API
                response = self.client.completions.create(
                    model=self.llm_model,
                    prompt=cot_prompt,
                    max_tokens=800,
                    temperature=0.2
                )
                content = response.choices[0].text
                
            else:
                # Try generic call method
                try:
                    response = self.client(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are a clinical researcher that generates detailed Chain-of-Thought reasoning based on biomedical knowledge graph paths. Specialize in fertility and reproductive medicine. Be precise and reference specific biological entities and relationships."},
                            {"role": "user", "content": cot_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=800
                    )
                    content = response["choices"][0]["message"]["content"] if isinstance(response, dict) else response
                except:
                    # Fallback: use the client directly with default parameters
                    import inspect
                    if callable(self.client):
                        sig = inspect.signature(self.client)
                        if "messages" in sig.parameters:
                            response = self.client(
                                messages=[
                                    {"role": "system", "content": "You are a clinical researcher that generates detailed Chain-of-Thought reasoning based on biomedical knowledge graph paths."},
                                    {"role": "user", "content": cot_prompt}
                                ],
                                temperature=0.2,
                                max_tokens=800
                            )
                            content = response if isinstance(response, str) else str(response)
                        else:
                            response = self.client(cot_prompt)
                            content = response if isinstance(response, str) else str(response)
                    else:
                        raise ValueError("LLM client has unknown interface")
            
            parsed = self.safe_json_parse(content)
            
            if not parsed:
                self._add_message("❌ Failed to parse LLM CoT response, using fallback")
                return self._generate_fallback_biomedical_cot(question, selected_path, entity_mappings)
            
            cot_data = parsed.get("chain_of_thought", {})
            
            # Ensure we have the required structure
            cot_data = self._validate_cot_structure(cot_data, selected_path, question)
            
            self._add_message("✅ Generated Chain-of-Thought reasoning")
            return cot_data
            
        except Exception as e:
            self._add_message(f"❌ Error in CoT generation: {e}")
            return self._generate_fallback_biomedical_cot(question, selected_path, entity_mappings)
    
    def _prepare_entity_context(self, entity_mappings: List[Dict]) -> List[Dict]:
        """Prepare entity mapping context for CoT."""
        entity_context = []
        for mapping in entity_mappings:
            if mapping.get("mapped_node"):
                entity = mapping["query_entity"]
                mapped = mapping["mapped_node"]
                entity_context.append({
                    "mentioned_in_question": entity["name"],
                    "mapped_to_knowledge_graph": mapped["node_name"],
                    "entity_type": entity.get("type", "Unknown"),
                    "mapping_confidence": round(mapped.get("enhanced_score", 0), 3),
                    "mapping_stage": mapped.get("mapping_stage", "unknown")
                })
        return entity_context
    
    def _prepare_path_nodes_info(self, nodes: List[Dict]) -> List[Dict]:
        """Prepare detailed information about path nodes for biomedical context."""
        path_nodes_info = []
        
        for i, node in enumerate(nodes):
            node_info = {
                "step": i + 1,
                "node_name": node["name"],
                "node_labels": node.get("labels", []),
                "role_in_path": "Source" if i == 0 else "Target" if i == len(nodes)-1 else "Intermediate"
            }
            
            # Add biomedical context based on labels and properties
            props = node.get("properties", {})
            
            # Check for disease properties
            for prop in self.disease_properties:
                if prop in props and props[prop]:
                    node_info["disease_info"] = f"Has {prop}: {str(props[prop])[:100]}..."
                    break
            
            # Check for drug properties
            for prop in self.drug_properties:
                if prop in props and props[prop]:
                    node_info["drug_info"] = f"Has {prop}: {str(props[prop])[:100]}..."
                    break
            
            # Add fertility relevance
            node_text = json.dumps(node).lower()
            fertility_mentions = [kw for kw in self.fertility_keywords if kw in node_text]
            if fertility_mentions:
                node_info["fertility_relevance"] = f"Mentions: {', '.join(fertility_mentions[:3])}"
            
            # Add key properties for context
            key_props = ["node_id", "node_name", "description"]
            for prop in key_props:
                if prop in props and props[prop]:
                    node_info[prop] = str(props[prop])[:50] + "..." if len(str(props[prop])) > 50 else str(props[prop])
            
            path_nodes_info.append(node_info)
        
        return path_nodes_info
    
    def _prepare_relationships_info(self, relationships: List[Dict], nodes: List[Dict]) -> List[Dict]:
        """Prepare detailed information about relationships."""
        relationships_info = []
        
        for i, rel in enumerate(relationships):
            rel_info = {
                "step": i + 1,
                "relationship_type": rel["type"],
                "biomedical_meaning": self.translate_biomedical_relationship(rel["type"]),
                "connects": f"{nodes[i]['name']} → {nodes[i+1]['name']}"
            }
            
            # Check if relationship has fertility relevance
            rel_text = json.dumps(rel).lower()
            fertility_rel_keywords = ["treats", "causes", "associated", "targets", "interacts", "regulates"]
            if any(kw in rel_text for kw in fertility_rel_keywords):
                rel_info["fertility_context"] = "May relate to fertility mechanisms"
            
            # Add relationship properties if available
            if rel.get("properties"):
                props = rel["properties"]
                if props:
                    rel_info["properties"] = {k: v for k, v in list(props.items())[:3]}  # Show first 3
            
            relationships_info.append(rel_info)
        
        return relationships_info
    
    def _create_cot_prompt(
        self, 
        question: str, 
        entity_context: List[Dict], 
        path_description: str,
        path_nodes_info: List[Dict],
        relationships_info: List[Dict]
    ) -> str:
        """Create the CoT generation prompt optimized for Llama models."""
        
        # Create a more concise prompt for Llama
        entity_context_str = "\n".join([
            f"- {ec['mentioned_in_question']} → {ec['mapped_to_knowledge_graph']} ({ec['entity_type']}, confidence: {ec['mapping_confidence']})"
            for ec in entity_context
        ])
        
        nodes_str = "\n".join([
            f"- {node['node_name']} ({node['role_in_path']}, labels: {', '.join(node['node_labels'][:2])})"
            for node in path_nodes_info
        ])
        
        relationships_str = "\n".join([
            f"- {rel['connects']} ({rel['biomedical_meaning']})"
            for rel in relationships_info
        ])
        
        return f"""Generate a detailed Chain-of-Thought reasoning to answer this clinical question:

CLINICAL QUESTION: "{question}"

ENTITY MAPPINGS:
{entity_context_str}

KNOWLEDGE GRAPH PATH:
{path_description}

PATH DETAILS:
Nodes in path:
{nodes_str}

Biological relationships:
{relationships_str}

Please provide a step-by-step clinical reasoning following this structure:

1. CLINICAL ENTITY IDENTIFICATION: Identify which clinical entities from the question are present in the path
2. PATH BIOLOGICAL INTERPRETATION: Explain what this path means in biological/medical terms
3. RELATIONSHIP ANALYSIS: Analyze each biological relationship in medical context
4. CLINICAL INFERENCE: Draw logical clinical conclusions from the complete path
5. FINAL ANSWER: Provide a concise, clinically relevant answer to the original question

Focus on:
- Fertility and reproductive health implications when relevant
- Disease-drug relationships
- Biological mechanisms
- Clinical significance

OUTPUT FORMAT (JSON):
{{
  "chain_of_thought": {{
    "reasoning_steps": [
      "Step 1: ...",
      "Step 2: ...",
      "Step 3: ...",
      "Step 4: ...", 
      "Step 5: ..."
    ],
    "detailed_reasoning": "Multi-paragraph explanation...",
    "final_answer": "Concise clinical answer...",
    "confidence_score": 0.85
  }}
}}

IMPORTANT: Base your reasoning ONLY on the specific path information provided above."""
    
    def _validate_cot_structure(self, cot_data: Dict, selected_path: Dict, question: str) -> Dict:
        """Validate and complete the CoT structure."""
        if not cot_data.get("reasoning_steps"):
            nodes = selected_path.get("nodes", [])
            source_name = nodes[0].get("name", "Unknown") if nodes else "Unknown"
            target_name = nodes[-1].get("name", "Unknown") if nodes else "Unknown"
            
            relationships = selected_path.get("relationships", [])
            rel_text = ""
            if relationships:
                rel_type = relationships[0].get("type", "connected")
                rel_meaning = self.translate_biomedical_relationship(rel_type)
                rel_text = f" through {rel_meaning}"
            
            cot_data["reasoning_steps"] = [
                f"Step 1: Identified clinical entities in the question",
                f"Step 2: Found biomedical path connecting {source_name} to {target_name}{rel_text}",
                f"Step 3: Analyzed biological relationships in the path",
                f"Step 4: Drew clinical conclusions from the biological connections",
                f"Step 5: Formulated clinical answer based on path analysis"
            ]
        
        if not cot_data.get("detailed_reasoning"):
            cot_data["detailed_reasoning"] = self._generate_biomedical_path_based_reasoning(selected_path, question)
        
        if not cot_data.get("final_answer"):
            cot_data["final_answer"] = self._generate_biomedical_final_answer(selected_path, question)
        
        if "confidence_score" not in cot_data:
            # Calculate confidence based on path quality
            path_length = selected_path.get("path_length", 0)
            confidence = 0.8 - (0.1 * path_length)  # Shorter paths are more reliable
            confidence = max(0.3, min(0.95, confidence))
            cot_data["confidence_score"] = round(confidence, 2)
        
        return {"chain_of_thought": cot_data}
    
    def _generate_biomedical_path_based_reasoning(self, path: Dict, question: str) -> str:
        """Generate clinical reasoning based on path structure."""
        if not path:
            return "No path available for clinical reasoning."
        
        path_desc = path.get("detailed_description", path.get("path_string", ""))
        nodes = path.get("nodes", [])
        
        reasoning = f"**Path Analysis:** {path_desc}\n\n"
        
        if len(nodes) >= 2:
            # Extract actual node names
            node_names = []
            for node in nodes:
                name = node.get("name", "Unknown")
                # Clean up the name
                name = name.split(" --[")[0] if " --[" in name else name
                node_names.append(name)
            
            reasoning += f"**Clinical Connection:** {node_names[0]} "
            
            for i, rel in enumerate(path.get("relationships", [])):
                rel_meaning = self.translate_biomedical_relationship(rel['type'])
                reasoning += f"→ {rel_meaning} → {node_names[i+1]} "
            
            reasoning += f"\n\n**Clinical Interpretation:** This biomedical pathway demonstrates a direct relationship between the clinical entities."
            
            # Add fertility context if relevant
            question_lower = question.lower()
            path_text = json.dumps(path).lower()
            
            if any(term in question_lower for term in self.fertility_keywords) or any(term in path_text for term in self.fertility_keywords):
                reasoning += " The connection is particularly relevant to fertility and reproductive health, suggesting potential implications for treatment or understanding of reproductive conditions."
        
        return reasoning
    
    def _generate_biomedical_final_answer(self, path: Dict, question: str) -> str:
        """Generate clinical final answer based on path."""
        if not path:
            return "Based on the available biomedical knowledge graph, no direct clinical relationship was found to answer the question."
        
        # Get the path string and nodes
        nodes = path.get("nodes", [])
        
        # Extract source and target node names
        source_name = "Unknown"
        target_name = "Unknown"
        
        if nodes:
            source_name = nodes[0].get("name", "Unknown") if nodes else "Unknown"
            target_name = nodes[-1].get("name", "Unknown") if nodes else "Unknown"
        
        # Clean up node names
        source_name = source_name.split(" --[")[0] if " --[" in source_name else source_name
        target_name = target_name.split(" --[")[0] if " --[" in target_name else target_name
        
        # Get relationship information
        relationships = path.get("relationships", [])
        rel_info = ""
        if relationships:
            rel_type = relationships[0].get("type", "connected")
            rel_meaning = self.translate_biomedical_relationship(rel_type)
            rel_info = f" {rel_meaning} "
        
        # Generate appropriate answer based on question
        question_lower = question.lower()
        
        # Fertility-specific patterns
        if any(term in question_lower for term in ["fertility", "infertility", "reproductive"]):
            if "drug" in question_lower or "treatment" in question_lower or "treat" in question_lower:
                return f"Yes, {source_name}{rel_info}{target_name} in the context of fertility treatment."
            elif "cause" in question_lower or "risk" in question_lower:
                return f"Yes, {source_name}{rel_info}{target_name}, which may impact fertility."
            else:
                return f"Yes, there is a biomedical relationship between {source_name} and {target_name} relevant to fertility."
        
        # Disease-drug patterns
        elif "disease" in question_lower and "drug" in question_lower:
            if "treat" in question_lower or "effective" in question_lower:
                return f"Yes, {target_name}{rel_info}{source_name} for treatment."
            elif "side effect" in question_lower or "risk" in question_lower:
                return f"Yes, {source_name}{rel_info}{target_name}, which may have clinical implications."
            else:
                return f"Yes, {source_name}{rel_info}{target_name} in the knowledge graph."
        
        # General biomedical relationship
        elif "relationship" in question_lower or "connect" in question_lower or "associate" in question_lower:
            return f"Yes, {source_name}{rel_info}{target_name} in the biomedical knowledge graph."
        
        else:
            # Generic biomedical answer
            return f"Yes, {source_name} is biologically connected to {target_name} in the knowledge graph."
    
    def _generate_fallback_biomedical_cot(
        self, 
        question: str, 
        selected_path: Dict, 
        entity_mappings: List[Dict]
    ) -> Dict[str, Any]:
        """Generate fallback biomedical Chain-of-Thought when LLM fails."""
        if not selected_path:
            return {
                "chain_of_thought": {
                    "reasoning_steps": ["No clinical path available for reasoning"],
                    "detailed_reasoning": "Cannot generate clinical reasoning without a valid biomedical knowledge graph path.",
                    "final_answer": "Insufficient information in biomedical knowledge graph to answer the clinical question.",
                    "confidence_score": 0.0
                }
            }
        
        # Get node names from the path
        nodes = selected_path.get("nodes", [])
        source_name = "Unknown"
        target_name = "Unknown"
        
        if nodes:
            source_name = nodes[0].get("name", "Unknown")
            target_name = nodes[-1].get("name", "Unknown")
            # Clean names
            source_name = source_name.split(" --[")[0] if " --[" in source_name else source_name
            target_name = target_name.split(" --[")[0] if " --[" in target_name else target_name
        
        # Get entity names from mappings
        entity_names = []
        for mapping in entity_mappings:
            if mapping.get("query_entity"):
                entity_names.append(mapping["query_entity"]["name"])
        
        path_desc = selected_path.get("detailed_description", selected_path.get("path_string", ""))
        
        # Get relationship information for biomedical context
        relationships = selected_path.get("relationships", [])
        rel_text = ""
        if relationships:
            rel_type = relationships[0].get("type", "connected")
            rel_meaning = self.translate_biomedical_relationship(rel_type)
            rel_text = f" through {rel_meaning}"
        
        reasoning_steps = [
            f"Step 1: Identified clinical entities in question: {', '.join(entity_names)}",
            f"Step 2: Found biomedical knowledge graph path connecting {source_name} to {target_name}{rel_text}",
            f"Step 3: Path length: {selected_path.get('path_length', 'N/A')} biological relationships",
            f"Step 4: The path shows that {source_name} is biologically connected to {target_name}",
            f"Step 5: Based on this biomedical connection, we can answer the clinical question"
        ]
        
        detailed_reasoning = self._generate_biomedical_path_based_reasoning(selected_path, question)
        final_answer = self._generate_biomedical_final_answer(selected_path, question)
        
        # Calculate confidence
        confidence = 0.7
        if selected_path.get("path_length", 0) <= 2:
            confidence = 0.8
        elif selected_path.get("path_length", 0) >= 4:
            confidence = 0.5
        
        confidence_score = selected_path.get("selection_confidence", confidence)
        
        return {
            "chain_of_thought": {
                "reasoning_steps": reasoning_steps,
                "detailed_reasoning": detailed_reasoning,
                "final_answer": final_answer,
                "confidence_score": confidence_score
            }
        }
    
    def format_biomedical_paths_for_display(self, path_results: Dict) -> str:
        """
        Format biomedical path results in a readable way.
        
        Args:
            path_results: Dictionary containing path search results
            
        Returns:
            Formatted string for display
        """
        if not path_results.get("entity_pairs"):
            return "No clinical paths found between mapped entities."
        
        output_lines = []
        output_lines.append("\n" + "="*80)
        output_lines.append("BIOMEDICAL PATH SEARCH RESULTS")
        output_lines.append("="*80)
        
        output_lines.append(f"\nTotal clinical entity pairs analyzed: {len(path_results['entity_pairs'])}")
        output_lines.append(f"Total biomedical paths found: {path_results.get('total_paths_found', 0)}")
        
        for pair_idx, pair in enumerate(path_results["entity_pairs"], 1):
            output_lines.append(f"\n{pair_idx}. {pair.get('source_entity', 'Unknown')} ({pair.get('source_type', 'Unknown')}) → {pair.get('target_entity', 'Unknown')} ({pair.get('target_type', 'Unknown')})")
            output_lines.append(f"   Mapped to KG: {pair.get('source_node', 'Unknown')} → {pair.get('target_node', 'Unknown')}")
            output_lines.append(f"   Paths found: {pair.get('paths_found', 0)}")
            output_lines.append(f"   Shortest path length: {pair.get('shortest_path_length', 'N/A')} biological relationships")
            
            top_paths = pair.get("top_paths", [])
            for path_idx, path in enumerate(top_paths[:2], 1):  # Show first 2 paths
                output_lines.append(f"\n   Path {path_idx} ({path.get('path_length', 'N/A')} hops):")
                output_lines.append(f"   {path.get('path_string', 'No path string')}")
                
                # Add biomedical context
                nodes = path.get("nodes", [])
                if nodes:
                    source_labels = nodes[0].get("labels", [])
                    target_labels = nodes[-1].get("labels", [])
                    if source_labels or target_labels:
                        output_lines.append(f"   Context: {source_labels[0] if source_labels else 'Unknown'} → {target_labels[0] if target_labels else 'Unknown'}")
        
        return "\n".join(output_lines)
    
    def generate_qa_workflow(
        self,
        question: str,
        entity_mappings: List[Dict],
        path_results: Dict,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Complete Q&A workflow: entity mapping → path finding → CoT reasoning.
        
        Args:
            question: Clinical question
            entity_mappings: Entity mappings from question to KG
            path_results: Path finding results
            use_llm: Whether to use LLM for CoT generation
            
        Returns:
            Complete Q&A workflow results
        """
        workflow_results = {
            "question": question,
            "entity_mappings": entity_mappings,
            "path_results": path_results,
            "cot_results": None,
            "summary": {},
            "timestamp": None
        }
        
        try:
            # Import datetime for timestamp
            from datetime import datetime
            workflow_results["timestamp"] = datetime.now().isoformat()
            
            self._add_message(f"🔍 Starting Q&A workflow for question: '{question}'")
            
            # Check if we have paths
            if not path_results.get("entity_pairs"):
                workflow_results["summary"] = {
                    "status": "no_paths",
                    "message": "No paths found between mapped entities",
                    "entities_mapped": len([m for m in entity_mappings if m.get("mapped_node")])
                }
                workflow_results["cot_results"] = {
                    "chain_of_thought": {
                        "reasoning_steps": ["No paths found between mapped clinical entities"],
                        "detailed_reasoning": "Cannot generate clinical reasoning without connecting paths in the knowledge graph.",
                        "final_answer": "The biomedical knowledge graph does not contain connecting paths between the identified clinical entities.",
                        "confidence_score": 0.0
                    }
                }
                return workflow_results
            
            # Select the best path (for simplicity, take the first path of the first pair)
            best_path = None
            if path_results["entity_pairs"]:
                first_pair = path_results["entity_pairs"][0]
                if first_pair.get("top_paths"):
                    best_path = first_pair["top_paths"][0]
            
            if not best_path:
                workflow_results["summary"] = {
                    "status": "no_valid_path",
                    "message": "No valid path selected for reasoning",
                    "paths_available": path_results.get("total_paths_found", 0)
                }
                return workflow_results
            
            # Generate Chain-of-Thought
            if use_llm and self.client:
                cot_results = self.generate_chain_of_thought(question, best_path, entity_mappings)
            else:
                cot_results = self._generate_fallback_biomedical_cot(question, best_path, entity_mappings)
            
            workflow_results["cot_results"] = cot_results
            
            # Generate summary
            entities_mapped = len([m for m in entity_mappings if m.get("mapped_node")])
            paths_found = path_results.get("total_paths_found", 0)
            confidence = cot_results.get("chain_of_thought", {}).get("confidence_score", 0)
            
            workflow_results["summary"] = {
                "status": "success",
                "entities_mapped": entities_mapped,
                "paths_found": paths_found,
                "selected_path_length": best_path.get("path_length", "N/A"),
                "cot_confidence": confidence,
                "has_final_answer": bool(cot_results.get("chain_of_thought", {}).get("final_answer"))
            }
            
            self._add_message(f"✅ Q&A workflow completed successfully")
            self._add_message(f"   Entities mapped: {entities_mapped}")
            self._add_message(f"   Paths found: {paths_found}")
            self._add_message(f"   CoT confidence: {confidence:.2f}")
            
        except Exception as e:
            self._add_message(f"❌ Error in Q&A workflow: {e}")
            workflow_results["summary"] = {
                "status": "error",
                "message": str(e)
            }
        
        return workflow_results
    
    def validate_cot(self, cot_data: Dict) -> Dict[str, Any]:
        """
        Validate a Chain-of-Thought reasoning structure.
        
        Args:
            cot_data: CoT data to validate
            
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "completeness_score": 0.0
        }
        
        if not cot_data:
            validation["is_valid"] = False
            validation["issues"].append("CoT data is empty")
            return validation
        
        chain_of_thought = cot_data.get("chain_of_thought", {})
        
        # Check required fields
        required_fields = ["reasoning_steps", "final_answer"]
        for field in required_fields:
            if field not in chain_of_thought:
                validation["is_valid"] = False
                validation["issues"].append(f"Missing required field: {field}")
        
        # Check reasoning steps
        reasoning_steps = chain_of_thought.get("reasoning_steps", [])
        if not isinstance(reasoning_steps, list):
            validation["is_valid"] = False
            validation["issues"].append("reasoning_steps should be a list")
        elif len(reasoning_steps) < 3:
            validation["warnings"].append(f"Only {len(reasoning_steps)} reasoning steps, should have at least 3")
        
        # Check detailed reasoning
        if not chain_of_thought.get("detailed_reasoning"):
            validation["warnings"].append("Missing detailed_reasoning field")
        
        # Check final answer
        final_answer = chain_of_thought.get("final_answer", "")
        if not final_answer or len(final_answer.strip()) < 10:
            validation["warnings"].append("Final answer is very short or empty")
        
        # Calculate completeness score
        score = 0.0
        if reasoning_steps:
            score += 0.3
        if chain_of_thought.get("detailed_reasoning"):
            score += 0.3
        if final_answer and len(final_answer.strip()) >= 20:
            score += 0.4
        
        validation["completeness_score"] = round(score, 2)
        
        return validation
    
    def export_cot_to_markdown(self, workflow_results: Dict) -> str:
        """
        Export the complete Q&A workflow results to Markdown format.
        
        Args:
            workflow_results: Complete Q&A workflow results
            
        Returns:
            Markdown formatted string
        """
        markdown_lines = []
        
        # Header
        markdown_lines.append("# Biomedical Q&A Workflow Report")
        markdown_lines.append("")
        
        # Question
        markdown_lines.append(f"## Question")
        markdown_lines.append(f"**{workflow_results.get('question', 'No question')}**")
        markdown_lines.append("")
        
        # Entity Mappings
        markdown_lines.append("## Entity Mappings")
        entity_mappings = workflow_results.get("entity_mappings", [])
        if entity_mappings:
            markdown_lines.append("| Question Entity | KG Entity | Type | Confidence |")
            markdown_lines.append("|----------------|-----------|------|------------|")
            for mapping in entity_mappings:
                query_entity = mapping.get("query_entity", {})
                mapped_node = mapping.get("mapped_node", {})
                markdown_lines.append(f"| {query_entity.get('name', 'N/A')} | {mapped_node.get('node_name', 'N/A')} | {query_entity.get('type', 'Unknown')} | {mapped_node.get('enhanced_score', 0):.3f} |")
        else:
            markdown_lines.append("*No entities mapped*")
        markdown_lines.append("")
        
        # Path Results
        path_results = workflow_results.get("path_results", {})
        if path_results.get("entity_pairs"):
            markdown_lines.append("## Path Discovery")
            markdown_lines.append(f"**Total paths found:** {path_results.get('total_paths_found', 0)}")
            markdown_lines.append("")
            
            for pair in path_results["entity_pairs"][:3]:  # Show first 3 pairs
                markdown_lines.append(f"### {pair.get('source_entity', 'Unknown')} → {pair.get('target_entity', 'Unknown')}")
                markdown_lines.append(f"- **Paths found:** {pair.get('paths_found', 0)}")
                markdown_lines.append(f"- **Shortest path:** {pair.get('shortest_path_length', 'N/A')} hops")
                
                # Show first path if available
                if pair.get("top_paths"):
                    first_path = pair["top_paths"][0]
                    markdown_lines.append(f"- **Example path:** {first_path.get('path_string', 'No path')}")
        markdown_lines.append("")
        
        # Chain-of-Thought
        cot_results = workflow_results.get("cot_results", {})
        if cot_results:
            markdown_lines.append("## Chain-of-Thought Reasoning")
            
            chain_of_thought = cot_results.get("chain_of_thought", {})
            
            # Reasoning Steps
            markdown_lines.append("### Reasoning Steps")
            for i, step in enumerate(chain_of_thought.get("reasoning_steps", []), 1):
                markdown_lines.append(f"{i}. {step}")
            markdown_lines.append("")
            
            # Detailed Reasoning
            detailed = chain_of_thought.get("detailed_reasoning", "")
            if detailed:
                markdown_lines.append("### Detailed Analysis")
                markdown_lines.append(detailed)
                markdown_lines.append("")
            
            # Final Answer
            final_answer = chain_of_thought.get("final_answer", "")
            if final_answer:
                markdown_lines.append("### Final Answer")
                markdown_lines.append(f"> {final_answer}")
                markdown_lines.append("")
            
            # Confidence
            confidence = chain_of_thought.get("confidence_score", 0)
            markdown_lines.append(f"**Confidence Score:** {confidence:.2f}")
        
        # Summary
        summary = workflow_results.get("summary", {})
        if summary:
            markdown_lines.append("## Summary")
            markdown_lines.append(f"- **Status:** {summary.get('status', 'unknown')}")
            markdown_lines.append(f"- **Entities Mapped:** {summary.get('entities_mapped', 0)}")
            markdown_lines.append(f"- **Paths Found:** {summary.get('paths_found', 0)}")
            if summary.get('cot_confidence') is not None:
                markdown_lines.append(f"- **CoT Confidence:** {summary.get('cot_confidence', 0):.2f}")
        
        # Timestamp
        timestamp = workflow_results.get("timestamp")
        if timestamp:
            markdown_lines.append("")
            markdown_lines.append(f"*Report generated: {timestamp}*")
        
        return "\n".join(markdown_lines)