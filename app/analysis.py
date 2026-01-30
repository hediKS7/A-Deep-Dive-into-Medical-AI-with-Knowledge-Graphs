# analysis.py
from typing import List, Dict, Optional, Any, DefaultDict
from collections import defaultdict
import pandas as pd
import numpy as np
import json
from datetime import datetime

class PipelineAnalyzer:
    """
    Analyze biomedical pipeline results with detailed property matching analysis.
    """
    
    def __init__(self):
        """Initialize the pipeline analyzer."""
        self.results = []
        self.analysis_summary = {}
        
        # Emoji mapping for visualization
        self.stage_emojis = {
            "property_exact": "🎯",
            "property_enhanced": "📊",
            "property_llm": "🤖",
            "property_fallback": "⚠️",
            "no_candidates": "❌",
            "unknown": "🔍"
        }
        
        self.entity_type_emojis = {
            "disease": "🦠",
            "drug": "💊",
            "Unknown": "❓",
            "unknown": "❓"
        }
        
        self.confidence_emojis = {
            "high": "🟢",
            "medium": "🟡",
            "low": "🔴"
        }
    
    def add_results(self, results: List[Dict]):
        """
        Add pipeline results for analysis.
        
        Args:
            results: List of pipeline results dictionaries
        """
        self.results.extend(results)
        self._update_summary()
    
    def clear_results(self):
        """Clear all stored results."""
        self.results = []
        self.analysis_summary = {}
    
    def analyze_property_matches(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Analyze property-based matching performance for biomedical entities.
        
        Args:
            detailed: Whether to include detailed statistics
            
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*70)
        print("BIOMEDICAL PROPERTY MATCHING ANALYSIS")
        print("="*70)
        
        analysis = {
            "summary": {},
            "stage_distribution": defaultdict(int),
            "match_type_distribution": defaultdict(int),
            "entity_type_analysis": defaultdict(lambda: defaultdict(int)),
            "confidence_analysis": defaultdict(int),
            "fertility_analysis": defaultdict(int)
        }
        
        total_mapped = 0
        property_based_matches = 0
        exact_property_matches = 0
        
        for result in self.results:
            mapped_entities = result.get("mapped_entities", result.get("entity_mapping", {}).get("mapped_results", []))
            
            for mapping in mapped_entities:
                if mapping.get("mapped_node"):
                    total_mapped += 1
                    stage = mapping.get("mapping_stage", "unknown")
                    analysis["stage_distribution"][stage] += 1
                    
                    # Get entity type
                    entity_type = mapping.get("query_entity", {}).get("type", "Unknown")
                    entity_type_lower = entity_type.lower()
                    analysis["entity_type_analysis"][entity_type_lower]["total"] += 1
                    analysis["entity_type_analysis"][entity_type_lower][stage] += 1
                    
                    # Check property matching
                    mapped_node = mapping.get("mapped_node", {})
                    match_type = mapped_node.get("match_type", "")
                    if match_type:
                        analysis["match_type_distribution"][match_type] += 1
                    
                    # Determine if property-based
                    is_property_based = False
                    if match_type and ("property" in match_type.lower() or 
                                      "exact_" in match_type or 
                                      "contains_" in match_type):
                        is_property_based = True
                        property_based_matches += 1
                        analysis["entity_type_analysis"][entity_type_lower]["property_based"] += 1
                    
                    # Check for exact matches
                    if "exact_" in match_type:
                        exact_property_matches += 1
                    
                    # Analyze confidence
                    score = mapped_node.get('enhanced_score', 0)
                    if score >= 0.8:
                        confidence_level = "high"
                    elif score >= 0.6:
                        confidence_level = "medium"
                    else:
                        confidence_level = "low"
                    analysis["confidence_analysis"][confidence_level] += 1
                    
                    # Fertility relevance
                    entity_name = mapping.get("query_entity", {}).get("name", "").lower()
                    fertility_keywords = ["pcos", "endometriosis", "infertility", "fertility", 
                                         "clomiphene", "letrozole", "ivf", "ovarian", "uterine"]
                    if any(kw in entity_name for kw in fertility_keywords):
                        analysis["fertility_analysis"]["fertility_entities"] += 1
        
        # Calculate statistics
        analysis["summary"]["total_mapped"] = total_mapped
        analysis["summary"]["property_based_matches"] = property_based_matches
        analysis["summary"]["exact_property_matches"] = exact_property_matches
        
        if total_mapped > 0:
            prop_percentage = property_based_matches / total_mapped * 100
            exact_percentage = exact_property_matches / total_mapped * 100
            analysis["summary"]["property_match_percentage"] = prop_percentage
            analysis["summary"]["exact_match_percentage"] = exact_percentage
            
            print(f"Total clinical entities mapped: {total_mapped}")
            print(f"Biomedical property-based matches: {property_based_matches} ({prop_percentage:.1f}%)")
            print(f"Exact property matches: {exact_property_matches} ({exact_percentage:.1f}%)")
        
        # Display stage distribution
        print(f"\n📊 Mapping Stage Distribution:")
        for stage, count in sorted(analysis["stage_distribution"].items()):
            emoji = self.stage_emojis.get(stage, "🔍")
            percentage = count / total_mapped * 100 if total_mapped > 0 else 0
            print(f"  {emoji} {stage}: {count} ({percentage:.1f}%)")
        
        # Display match type distribution
        print(f"\n🎯 Match Type Distribution:")
        grouped_matches = defaultdict(int)
        for match_type, count in analysis["match_type_distribution"].items():
            if "exact_primary_key" in match_type or "exact_display_property" in match_type:
                grouped_matches["exact_property"] += count
            elif "contains_" in match_type:
                grouped_matches["contains_property"] += count
            elif "exact_name" in match_type:
                grouped_matches["exact_name"] += count
            elif "contains_name" in match_type:
                grouped_matches["contains_name"] += count
            elif "property_enhanced" in match_type:
                grouped_matches["property_enhanced"] += count
            else:
                grouped_matches["other"] += count
        
        match_display = {
            "exact_property": "🎯 Exact property matches",
            "contains_property": "🔍 Contains property matches", 
            "exact_name": "🏷️ Exact name matches",
            "contains_name": "📄 Contains name matches",
            "property_enhanced": "⚡ Property-enhanced similarity",
            "other": "❓ Other matches"
        }
        
        for match_type, count in sorted(grouped_matches.items()):
            if count > 0:
                display = match_display.get(match_type, match_type)
                percentage = count / total_mapped * 100 if total_mapped > 0 else 0
                print(f"  {display}: {count} ({percentage:.1f}%)")
        
        # Display entity type distribution
        print(f"\n🏥 Entity Type Distribution:")
        if analysis["entity_type_analysis"]:
            for entity_type, stats in analysis["entity_type_analysis"].items():
                emoji = self.entity_type_emojis.get(entity_type, "❓")
                total = stats.get("total", 0)
                prop_based = stats.get("property_based", 0)
                
                if total > 0:
                    prop_pct = prop_based / total * 100
                    print(f"  {emoji} {entity_type.capitalize()}: {total} entities")
                    print(f"     Property-based matches: {prop_based} ({prop_pct:.1f}%)")
                    
                    # Show stage distribution for this entity type
                    for stage, count in stats.items():
                        if stage not in ["total", "property_based"] and count > 0:
                            stage_emoji = self.stage_emojis.get(stage, "🔍")
                            stage_pct = count / total * 100
                            print(f"     {stage_emoji} {stage}: {count} ({stage_pct:.1f}%)")
        
        # Display confidence analysis
        print(f"\n📈 Confidence Level Analysis:")
        for level, count in sorted(analysis["confidence_analysis"].items()):
            emoji = self.confidence_emojis.get(level, "⚪")
            percentage = count / total_mapped * 100 if total_mapped > 0 else 0
            print(f"  {emoji} {level.capitalize()} (≥{self._get_confidence_threshold(level)}): "
                  f"{count} ({percentage:.1f}%)")
        
        # Fertility analysis
        if analysis["fertility_analysis"].get("fertility_entities", 0) > 0:
            print(f"\n🎯 Fertility Relevance:")
            fertility_count = analysis["fertility_analysis"]["fertility_entities"]
            fertility_pct = fertility_count / total_mapped * 100 if total_mapped > 0 else 0
            print(f"  🏥 Fertility-related entities: {fertility_count} ({fertility_pct:.1f}%)")
        
        return dict(analysis)
    
    def generate_property_mapping_report(self, result: Dict, show_details: bool = True) -> Dict[str, Any]:
        """
        Generate detailed clinical property mapping report for a single result.
        
        Args:
            result: Single pipeline result
            show_details: Whether to print detailed report
            
        Returns:
            Detailed mapping report
        """
        report = {
            "question": result.get('clinical_question', result.get('text', '')),
            "timestamp": result.get('timestamp', datetime.now().isoformat()),
            "entities": [],
            "path_analysis": {},
            "cot_summary": {}
        }
        
        if show_details:
            print("\n" + "="*70)
            print("CLINICAL PROPERTY MAPPING REPORT")
            print("="*70)
            
            question = result.get('clinical_question', result.get('text', ''))
            print(f"Clinical Question: {question[:120]}{'...' if len(question) > 120 else ''}")
            
            entity_extraction = result.get("entity_extraction", {})
            print(f"Extracted biomedical entities: {entity_extraction.get('total_entities', 0)}")
            
            # Check fertility relevance
            clinical_context = result.get("clinical_context", {})
            fertility_context = clinical_context.get("fertility_relevance", {})
            if fertility_context.get("is_fertility_question", False):
                print(f"Fertility relevance: ✅ (Keywords detected in question)")
            if fertility_context.get("extracted_fertility_entities", 0) > 0:
                print(f"Fertility entities extracted: {fertility_context['extracted_fertility_entities']}")
        
        # Get mapped entities
        mapped_entities = result.get("mapped_entities", [])
        entity_mapping = result.get("entity_mapping", {})
        if not mapped_entities and entity_mapping:
            mapped_entities = entity_mapping.get("mapped_results", [])
        
        if show_details:
            print(f"\n🏥 Biomedical Entity Mapping Details:")
        
        for i, mapping in enumerate(mapped_entities, 1):
            entity_report = self._analyze_entity_mapping(mapping, i, show_details)
            report["entities"].append(entity_report)
        
        # Path analysis
        path_discovery = result.get("path_discovery", {})
        path_pruning = path_discovery.get("path_pruning_result", {})
        selected_path = path_pruning.get("selected_path")
        
        if selected_path and show_details:
            print(f"\n🛤️  Selected Biomedical Pathway:")
            print(f"   Pathway: {selected_path.get('detailed_description', selected_path.get('path_string', ''))}")
            print(f"   Length: {selected_path.get('path_length', 'N/A')} biological relationships")
            print(f"   Selection confidence: {path_pruning.get('confidence', 0):.2f}")
            
            # Show entities in path
            nodes = selected_path.get("nodes", [])
            if nodes:
                print(f"   Entities in pathway:")
                for j, node in enumerate(nodes[:3]):
                    node_labels = node.get("labels", [])
                    label_str = node_labels[0] if node_labels else "Unknown"
                    print(f"     {j+1}. {node.get('name', 'Unknown')} ({label_str})")
                if len(nodes) > 3:
                    print(f"     ... and {len(nodes) - 3} more")
        
        report["path_analysis"] = {
            "selected_path": bool(selected_path),
            "path_length": selected_path.get("path_length") if selected_path else None,
            "selection_confidence": path_pruning.get("confidence", 0),
            "nodes_in_path": len(selected_path.get("nodes", [])) if selected_path else 0
        }
        
        # Clinical reasoning summary
        cot_result = result.get("chain_of_thought", {})
        if cot_result and "chain_of_thought" in cot_result:
            cot = cot_result["chain_of_thought"]
            
            if show_details:
                print(f"\n🧠 Clinical Reasoning Summary:")
                print(f"   Final clinical answer: {cot.get('final_answer', 'N/A')}")
                
                steps = cot.get('reasoning_steps', [])
                if steps and len(steps) > 0:
                    print(f"   First reasoning step: {steps[0][:80]}...")
            
            report["cot_summary"] = {
                "has_final_answer": bool(cot.get("final_answer")),
                "final_answer": cot.get("final_answer", ""),
                "reasoning_steps_count": len(cot.get("reasoning_steps", [])),
                "confidence_score": cot.get("confidence_score", 0),
                "has_detailed_reasoning": bool(cot.get("detailed_reasoning"))
            }
        
        return report
    
    def generate_comprehensive_report(self, output_format: str = "text") -> Any:
        """
        Generate comprehensive analysis report across all results.
        
        Args:
            output_format: "text", "json", or "dataframe"
            
        Returns:
            Analysis report in requested format
        """
        if not self.results:
            return "No results available for analysis"
        
        # Collect statistics
        stats = {
            "total_questions": len(self.results),
            "total_entities_extracted": 0,
            "total_entities_mapped": 0,
            "total_paths_found": 0,
            "fertility_questions": 0,
            "successful_pipelines": 0
        }
        
        entity_type_stats = defaultdict(int)
        confidence_stats = defaultdict(int)
        stage_stats = defaultdict(int)
        
        for result in self.results:
            # Count extracted entities
            entity_extraction = result.get("entity_extraction", {})
            stats["total_entities_extracted"] += entity_extraction.get("total_entities", 0)
            
            # Count mapped entities
            entity_mapping = result.get("entity_mapping", {})
            stats["total_entities_mapped"] += entity_mapping.get("total_mapped", 0)
            
            # Count paths
            path_discovery = result.get("path_discovery", {})
            path_stats = path_discovery.get("statistics", {})
            stats["total_paths_found"] += path_stats.get("total_paths_found", 0)
            
            # Check fertility relevance
            clinical_context = result.get("clinical_context", {})
            fertility_context = clinical_context.get("fertility_relevance", {})
            if fertility_context.get("is_fertility_question", False):
                stats["fertility_questions"] += 1
            
            # Check pipeline success
            summary = result.get("summary", {})
            if summary.get("overall_status") == "success":
                stats["successful_pipelines"] += 1
            
            # Collect entity type distribution
            mapped_results = entity_mapping.get("mapped_results", [])
            for mapping in mapped_results:
                entity_type = mapping.get("query_entity", {}).get("type", "Unknown")
                entity_type_stats[entity_type] += 1
                
                # Confidence distribution
                mapped_node = mapping.get("mapped_node", {})
                score = mapped_node.get('enhanced_score', 0)
                if score >= 0.8:
                    confidence_stats["high"] += 1
                elif score >= 0.6:
                    confidence_stats["medium"] += 1
                else:
                    confidence_stats["low"] += 1
                
                # Stage distribution
                stage = mapping.get("mapping_stage", "unknown")
                stage_stats[stage] += 1
        
        # Calculate percentages
        if stats["total_questions"] > 0:
            stats["success_rate"] = stats["successful_pipelines"] / stats["total_questions"] * 100
            stats["fertility_question_percentage"] = stats["fertility_questions"] / stats["total_questions"] * 100
        
        if stats["total_entities_extracted"] > 0:
            stats["mapping_success_rate"] = stats["total_entities_mapped"] / stats["total_entities_extracted"] * 100
        
        # Prepare report based on format
        if output_format == "text":
            return self._generate_text_report(stats, entity_type_stats, confidence_stats, stage_stats)
        elif output_format == "json":
            return {
                "statistics": stats,
                "entity_type_distribution": dict(entity_type_stats),
                "confidence_distribution": dict(confidence_stats),
                "stage_distribution": dict(stage_stats),
                "timestamp": datetime.now().isoformat()
            }
        elif output_format == "dataframe":
            # Create DataFrames for different aspects
            df_stats = pd.DataFrame([stats])
            df_entity_types = pd.DataFrame(list(entity_type_stats.items()), 
                                          columns=["Entity Type", "Count"])
            df_confidence = pd.DataFrame(list(confidence_stats.items()),
                                        columns=["Confidence Level", "Count"])
            df_stages = pd.DataFrame(list(stage_stats.items()),
                                    columns=["Mapping Stage", "Count"])
            
            return {
                "statistics": df_stats,
                "entity_types": df_entity_types,
                "confidence": df_confidence,
                "stages": df_stages
            }
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _analyze_entity_mapping(self, mapping: Dict, index: int, show_details: bool) -> Dict:
        """Analyze a single entity mapping."""
        entity_report = {
            "index": index,
            "query_entity": mapping.get("query_entity", {}),
            "is_mapped": False,
            "mapping_details": {}
        }
        
        query_entity = mapping.get("query_entity", {})
        entity_type = query_entity.get("type", "Unknown")
        entity_emoji = self.entity_type_emojis.get(entity_type.lower(), "❓")
        
        if show_details:
            print(f"\n{index}. {entity_emoji} Clinical Entity: {query_entity.get('name')} ({entity_type})")
            print(f"   Extraction confidence: {query_entity.get('confidence', 0.7):.2f}")
        
        mapped_node = mapping.get("mapped_node")
        if mapped_node:
            entity_report["is_mapped"] = True
            
            score = mapped_node.get('enhanced_score', 0)
            stage = mapped_node.get('mapping_stage', 'unknown')
            match_type = mapped_node.get('match_type', 'unknown')
            
            entity_report["mapping_details"] = {
                "kg_node_name": mapped_node.get('node_name'),
                "enhanced_score": score,
                "mapping_stage": stage,
                "match_type": match_type,
                "kg_labels": mapped_node.get('labels', [])
            }
            
            if show_details:
                # Score color coding
                if score >= 0.9:
                    score_color = "\033[92m"  # Green
                elif score >= 0.7:
                    score_color = "\033[93m"  # Yellow
                else:
                    score_color = "\033[91m"  # Red
                
                print(f"   ✅ Mapped to KG: {mapped_node.get('node_name')}")
                print(f"   Stage: {stage}")
                print(f"   Enhanced score: {score_color}{score:.4f}\033[0m")
                print(f"   Match type: {match_type}")
                
                # Show fertility relevance
                entity_name = query_entity.get('name', '').lower()
                fertility_keywords = ["pcos", "endometriosis", "infertility", "fertility", 
                                     "clomiphene", "letrozole", "ivf"]
                if any(kw in entity_name for kw in fertility_keywords):
                    print(f"   🏥 Fertility relevance: High")
                
                # Show property match details
                property_info = mapping.get("property_info", {})
                if property_info:
                    print(f"\n   🔬 Biomedical Property Analysis:")
                    
                    match_type = property_info.get("match_type", "")
                    if match_type:
                        if "exact_primary_key" in match_type:
                            print(f"     🎯 {match_type}")
                        elif "exact_display_property" in match_type:
                            print(f"     📊 {match_type}")
                        elif "contains" in match_type:
                            print(f"     🔍 {match_type}")
                    
                    matched_props = property_info.get("matched_properties", [])
                    if matched_props:
                        print(f"     Matched biomedical properties: {', '.join(matched_props[:3])}")
                        if len(matched_props) > 3:
                            print(f"     ... and {len(matched_props) - 3} more")
        else:
            entity_report["mapping_details"] = {
                "reason": mapping.get('reason', 'No match found'),
                "mapping_stage": mapping.get('mapping_stage', 'unknown')
            }
            
            if show_details:
                print(f"   ❌ Not mapped to knowledge graph")
                print(f"   Stage: {mapping.get('mapping_stage', 'unknown')}")
                print(f"   Reason: {mapping.get('reason', 'No match found')}")
        
        return entity_report
    
    def _generate_text_report(self, stats: Dict, entity_types: Dict, 
                            confidence: Dict, stages: Dict) -> str:
        """Generate text report."""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("COMPREHENSIVE PIPELINE ANALYSIS REPORT")
        lines.append("="*70)
        
        lines.append(f"\n📊 Overall Statistics:")
        lines.append(f"  • Total questions processed: {stats['total_questions']}")
        lines.append(f"  • Successful pipelines: {stats.get('successful_pipelines', 0)} "
                    f"({stats.get('success_rate', 0):.1f}%)")
        lines.append(f"  • Fertility questions: {stats.get('fertility_questions', 0)} "
                    f"({stats.get('fertility_question_percentage', 0):.1f}%)")
        lines.append(f"  • Total entities extracted: {stats['total_entities_extracted']}")
        lines.append(f"  • Total entities mapped: {stats['total_entities_mapped']}")
        if stats['total_entities_extracted'] > 0:
            lines.append(f"  • Mapping success rate: {stats.get('mapping_success_rate', 0):.1f}%")
        lines.append(f"  • Total paths found: {stats['total_paths_found']}")
        
        lines.append(f"\n🏥 Entity Type Distribution:")
        for entity_type, count in sorted(entity_types.items()):
            emoji = self.entity_type_emojis.get(entity_type.lower(), "❓")
            lines.append(f"  {emoji} {entity_type}: {count}")
        
        lines.append(f"\n📈 Confidence Distribution:")
        total_confidence = sum(confidence.values())
        for level, count in sorted(confidence.items()):
            emoji = self.confidence_emojis.get(level, "⚪")
            percentage = count / total_confidence * 100 if total_confidence > 0 else 0
            threshold = self._get_confidence_threshold(level)
            lines.append(f"  {emoji} {level.capitalize()} (≥{threshold}): {count} ({percentage:.1f}%)")
        
        lines.append(f"\n🔧 Mapping Stage Distribution:")
        total_stages = sum(stages.values())
        for stage, count in sorted(stages.items()):
            emoji = self.stage_emojis.get(stage, "🔍")
            percentage = count / total_stages * 100 if total_stages > 0 else 0
            lines.append(f"  {emoji} {stage}: {count} ({percentage:.1f}%)")
        
        # Add summary insights
        lines.append(f"\n💡 Key Insights:")
        
        if stats.get('mapping_success_rate', 0) >= 70:
            lines.append("  ✅ High entity mapping success rate")
        elif stats.get('mapping_success_rate', 0) >= 40:
            lines.append("  ⚠️ Moderate entity mapping success rate")
        else:
            lines.append("  ❌ Low entity mapping success rate")
        
        if stats.get('fertility_question_percentage', 0) >= 50:
            lines.append("  🎯 Strong focus on fertility-related questions")
        
        high_confidence_pct = confidence.get('high', 0) / total_confidence * 100 if total_confidence > 0 else 0
        if high_confidence_pct >= 60:
            lines.append("  📈 High confidence mappings predominant")
        
        return "\n".join(lines)
    
    def _get_confidence_threshold(self, level: str) -> float:
        """Get confidence threshold for a level."""
        thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.0
        }
        return thresholds.get(level, 0.0)
    
    def _update_summary(self):
        """Update internal summary statistics."""
        if not self.results:
            self.analysis_summary = {}
            return
        
        # Calculate basic statistics
        self.analysis_summary = {
            "total_results": len(self.results),
            "last_updated": datetime.now().isoformat()
        }
    
    def export_analysis(self, filename: str, format: str = "json"):
        """
        Export analysis to file.
        
        Args:
            filename: Output filename
            format: Export format ("json", "csv", "txt")
        """
        if not self.results:
            print("❌ No results to export")
            return
        
        analysis = self.generate_comprehensive_report("json")
        
        try:
            if format == "json":
                with open(f"{filename}.json", 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"✅ Analysis exported to {filename}.json")
            
            elif format == "csv":
                # Convert to DataFrames and export
                df_stats = pd.DataFrame([analysis["statistics"]])
                df_entity_types = pd.DataFrame(list(analysis["entity_type_distribution"].items()),
                                              columns=["Entity Type", "Count"])
                df_confidence = pd.DataFrame(list(analysis["confidence_distribution"].items()),
                                            columns=["Confidence Level", "Count"])
                
                with pd.ExcelWriter(f"{filename}.xlsx") as writer:
                    df_stats.to_excel(writer, sheet_name='Statistics', index=False)
                    df_entity_types.to_excel(writer, sheet_name='Entity Types', index=False)
                    df_confidence.to_excel(writer, sheet_name='Confidence', index=False)
                
                print(f"✅ Analysis exported to {filename}.xlsx")
            
            elif format == "txt":
                text_report = self.generate_comprehensive_report("text")
                with open(f"{filename}.txt", 'w') as f:
                    f.write(text_report)
                print(f"✅ Analysis exported to {filename}.txt")
            
            else:
                print(f"❌ Unsupported format: {format}")
                
        except Exception as e:
            print(f"❌ Error exporting analysis: {e}")
    
    def plot_distributions(self, save_path: Optional[str] = None):
        """
        Create visualizations of analysis distributions.
        
        Args:
            save_path: Optional path to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get data
            analysis = self.analyze_property_matches(detailed=False)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Biomedical Pipeline Analysis', fontsize=16, fontweight='bold')
            
            # Entity Type Distribution
            entity_types = analysis.get("entity_type_analysis", {})
            if entity_types:
                labels = []
                sizes = []
                for entity_type, stats in entity_types.items():
                    if stats.get("total", 0) > 0:
                        labels.append(entity_type.capitalize())
                        sizes.append(stats["total"])
                
                axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Entity Type Distribution')
            
            # Confidence Distribution
            confidence = analysis.get("confidence_analysis", {})
            if confidence:
                labels = [f"{k.capitalize()}\n(≥{self._get_confidence_threshold(k)})" 
                         for k in confidence.keys()]
                sizes = list(confidence.values())
                
                # Use confidence emojis as colors
                colors = ['#4CAF50' if 'high' in k.lower() else 
                         '#FFC107' if 'medium' in k.lower() else 
                         '#F44336' for k in confidence.keys()]
                
                axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Mapping Confidence Distribution')
            
            # Stage Distribution
            stages = analysis.get("stage_distribution", {})
            if stages:
                labels = []
                sizes = []
                colors = []
                
                stage_colors = {
                    "property_exact": "#4CAF50",
                    "property_enhanced": "#2196F3", 
                    "property_llm": "#9C27B0",
                    "property_fallback": "#FF9800",
                    "no_candidates": "#F44336"
                }
                
                for stage, count in stages.items():
                    if count > 0:
                        labels.append(stage.replace('_', ' ').title())
                        sizes.append(count)
                        colors.append(stage_colors.get(stage, "#9E9E9E"))
                
                axes[1, 0].bar(labels, sizes, color=colors)
                axes[1, 0].set_title('Mapping Stage Distribution')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].set_ylabel('Count')
            
            # Match Type Distribution (simplified)
            match_types = analysis.get("match_type_distribution", {})
            if match_types:
                # Group match types
                grouped = {
                    "Exact Property": sum(v for k, v in match_types.items() if "exact_" in k),
                    "Contains Property": sum(v for k, v in match_types.items() if "contains_" in k),
                    "Name Match": sum(v for k, v in match_types.items() if "name" in k),
                    "Other": sum(v for k, v in match_types.items() if "exact_" not in k and "contains_" not in k and "name" not in k)
                }
                
                labels = list(grouped.keys())
                sizes = list(grouped.values())
                colors = ['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']
                
                axes[1, 1].bar(labels, sizes, color=colors)
                axes[1, 1].set_title('Match Type Distribution')
                axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("❌ Matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")