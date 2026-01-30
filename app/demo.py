# demo.py
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from analysis import PipelineAnalyzer

class BiomedicalPipelineDemo:
    """
    Complete clinical pipeline demonstration with entity extraction, 
    mapping, path pruning, and CoT generation.
    """
    
    def __init__(self, pipeline, cot_generator, analyzer=None):
        """
        Initialize the pipeline demo.
        
        Args:
            pipeline: BiomedicalPipeline instance
            cot_generator: ChainOfThoughtGenerator instance
            analyzer: PipelineAnalyzer instance
        """
        self.pipeline = pipeline
        self.cot_generator = cot_generator
        self.analyzer = analyzer or PipelineAnalyzer()
        self.results = []
        
        # Clinical test cases focused on fertility/reproductive medicine
        self.test_cases = [
            {
                "question": "Is Clomiphene citrate effective for endometriosis-related infertility?",
                "category": "fertility_treatment",
                "expected_entities": ["Clomiphene citrate", "endometriosis", "infertility"]
            },
            {
                "question": "How does Metformin help with infertility in PCOS patients?",
                "category": "fertility_treatment",
                "expected_entities": ["Metformin", "PCOS", "infertility"]
            },
            {
                "question": "What is the relationship between Liothyronine and fertility issues?",
                "category": "fertility_relationship",
                "expected_entities": ["Liothyronine", "fertility"]
            },
            {
                "question": "Can Hydrocortisone be used to treat autoimmune infertility conditions?",
                "category": "fertility_treatment",
                "expected_entities": ["Hydrocortisone", "autoimmune", "infertility"]
            },
            {
                "question": "What is the mechanism of Letrozole for ovulation induction?",
                "category": "fertility_mechanism",
                "expected_entities": ["Letrozole", "ovulation"]
            }
        ]
    
    def run_complete_pipeline_demo(self, use_llm: bool = True, detailed_output: bool = True) -> List[Dict[str, Any]]:
        """
        Run complete clinical pipeline with entity extraction, mapping, path pruning, and CoT generation.
        
        Args:
            use_llm: Whether to use LLM for CoT generation
            detailed_output: Whether to show detailed output
            
        Returns:
            List of pipeline results
        """
        if detailed_output:
            print("\n" + "="*80)
            print("CLINICAL BIOMEDICAL PIPELINE WITH PATH PRUNING & COT GENERATION")
            print("="*80)
            print(f"🔬 Running {len(self.test_cases)} clinical test cases\n")
        
        all_results = []
        start_time = time.time()
        
        for i, test_case in enumerate(self.test_cases, 1):
            question = test_case["question"]
            
            if detailed_output:
                print(f"\n📋 CLINICAL TEST CASE {i}: {test_case['category'].replace('_', ' ').title()}")
                print("-" * 80)
                print(f"❓ Clinical Question: {question}")
            
            try:
                # Run the complete pipeline
                result = self.pipeline.extract_and_map_entities_with_cot(question)
                all_results.append(result)
                
                # Display clinical summary
                self._display_case_summary(i, result, test_case, detailed_output)
                
            except Exception as e:
                error_result = {
                    "clinical_question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(error_result)
                
                if detailed_output:
                    print(f"❌ Error processing case {i}: {e}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if detailed_output:
            self._display_overall_summary(all_results, execution_time)
        
        # Store results in analyzer
        self.analyzer.add_results([r for r in all_results if "error" not in r])
        self.results = all_results
        
        return all_results
    
    def _display_case_summary(self, case_num: int, result: Dict, test_case: Dict, detailed: bool):
        """Display summary for a single test case."""
        if "error" in result:
            if detailed:
                print(f"   ❌ Error: {result['error']}")
            return
        
        if detailed:
            # Show entity extraction
            entity_extraction = result.get("entity_extraction", {})
            extracted_entities = entity_extraction.get("extracted_entities", [])
            total_entities = entity_extraction.get("total_entities", 0)
            
            print(f"   📝 Extracted {total_entities} biomedical entities:")
            for entity in extracted_entities[:3]:  # Show first 3
                entity_type = entity.get('type', 'Unknown')
                entity_emoji = "🦠" if entity_type.lower() == "disease" else "💊" if entity_type.lower() == "drug" else "❓"
                confidence = entity.get('confidence', 0.7)
                print(f"     {entity_emoji} {entity['name']} ({entity_type}, confidence: {confidence:.2f})")
            if len(extracted_entities) > 3:
                print(f"     ... and {len(extracted_entities) - 3} more")
            
            # Show entity mapping
            entity_mapping = result.get("entity_mapping", {})
            mapped_count = entity_mapping.get("total_mapped", 0)
            success_rate = entity_mapping.get("success_rate", 0)
            
            print(f"   🗺️  Mapped {mapped_count} entities ({success_rate:.1f}% success rate)")
            
            # Show high confidence mappings
            successful_mappings = entity_mapping.get("successful_mappings", [])
            high_conf_mappings = [m for m in successful_mappings 
                                if m.get("mapped_node", {}).get("enhanced_score", 0) >= 0.8]
            if high_conf_mappings:
                print(f"     🎯 High confidence mappings: {len(high_conf_mappings)}")
                for mapping in high_conf_mappings[:2]:  # Show first 2
                    mapped_node = mapping.get("mapped_node", {})
                    query_entity = mapping.get("query_entity", {})
                    print(f"       • {query_entity.get('name', 'Unknown')} → {mapped_node.get('node_name', 'Unknown')} "
                          f"(score: {mapped_node.get('enhanced_score', 0):.3f})")
            
            # Show path discovery
            path_discovery = result.get("path_discovery", {})
            path_stats = path_discovery.get("statistics", {})
            total_paths = path_stats.get("total_paths_found", 0)
            
            if total_paths > 0:
                print(f"   🛤️  Found {total_paths} biomedical pathways")
                
                # Show selected path
                selected_path = path_discovery.get("selected_path")
                if selected_path:
                    path_desc = selected_path.get('detailed_description', selected_path.get('path_string', ''))
                    if len(path_desc) > 100:
                        path_desc = path_desc[:100] + "..."
                    print(f"     Selected: {path_desc}")
                    
                    confidence = path_discovery.get("selected_path_confidence", 0)
                    confidence_emoji = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                    print(f"     Relevance confidence: {confidence_emoji} {confidence:.2f}")
            else:
                print(f"   🛤️  No biomedical pathways found")
            
            # Show clinical reasoning
            cot_result = result.get("chain_of_thought", {})
            if cot_result and "chain_of_thought" in cot_result:
                cot = cot_result["chain_of_thought"]
                final_answer = cot.get('final_answer', '')
                
                if final_answer:
                    print(f"   🧠 Clinical Answer: \"{final_answer}\"")
                    
                    # Show reasoning steps count
                    steps = cot.get('reasoning_steps', [])
                    if steps:
                        print(f"     Reasoning steps: {len(steps)}")
                        
                        # Show first step
                        first_step = steps[0]
                        if len(first_step) > 80:
                            first_step = first_step[:80] + "..."
                        print(f"     First step: {first_step}")
                    
                    # Show confidence
                    confidence = cot.get('confidence_score', 0)
                    if confidence > 0:
                        print(f"     Confidence score: {confidence:.2f}")
            
            # Show fertility relevance
            clinical_context = result.get("clinical_context", {})
            fertility_context = clinical_context.get("fertility_relevance", {})
            if fertility_context.get("is_fertility_question", False):
                print(f"   🎯 Fertility context detected")
                
                fertility_keywords = fertility_context.get("fertility_keywords_present", [])
                if fertility_keywords:
                    print(f"     Keywords: {', '.join(fertility_keywords[:3])}")
                
                fertility_entities = fertility_context.get("extracted_fertility_entities", 0)
                if fertility_entities > 0:
                    print(f"     Fertility entities: {fertility_entities}")
    
    def _display_overall_summary(self, all_results: List[Dict], execution_time: float):
        """Display overall summary of the demo."""
        print(f"\n" + "="*80)
        print("CLINICAL PIPELINE DEMO COMPLETE")
        print("="*80)
        
        # Filter out error results
        successful_results = [r for r in all_results if "error" not in r]
        error_results = [r for r in all_results if "error" in r]
        
        total_questions = len(self.test_cases)
        successful_questions = len(successful_results)
        
        print(f"\n📈 Overall Clinical Performance:")
        print(f"   Total clinical questions: {total_questions}")
        print(f"   Successfully processed: {successful_questions} ({successful_questions/total_questions*100:.1f}%)")
        print(f"   Execution time: {execution_time:.2f} seconds")
        print(f"   Average time per question: {execution_time/total_questions:.2f} seconds")
        
        if successful_results:
            # Entity extraction statistics
            total_entities_extracted = sum(
                r.get("entity_extraction", {}).get("total_entities", 0) 
                for r in successful_results
            )
            avg_entities_per_question = total_entities_extracted / successful_questions if successful_questions > 0 else 0
            
            print(f"\n🧬 Entity Extraction:")
            print(f"   Total entities extracted: {total_entities_extracted}")
            print(f"   Average per question: {avg_entities_per_question:.1f}")
            
            # Entity mapping statistics
            total_entities_mapped = sum(
                r.get("entity_mapping", {}).get("total_mapped", 0) 
                for r in successful_results
            )
            mapping_success_rate = total_entities_mapped / total_entities_extracted * 100 if total_entities_extracted > 0 else 0
            
            print(f"\n🗺️  Entity Mapping:")
            print(f"   Total entities mapped: {total_entities_mapped}")
            print(f"   Mapping success rate: {mapping_success_rate:.1f}%")
            
            # Path discovery statistics
            total_paths_found = sum(
                r.get("path_discovery", {}).get("statistics", {}).get("total_paths_found", 0) 
                for r in successful_results
            )
            questions_with_paths = sum(
                1 for r in successful_results 
                if r.get("path_discovery", {}).get("statistics", {}).get("total_paths_found", 0) > 0
            )
            
            print(f"\n🛤️  Path Discovery:")
            print(f"   Total pathways found: {total_paths_found}")
            print(f"   Questions with pathways: {questions_with_paths}/{successful_questions} "
                  f"({questions_with_paths/successful_questions*100:.1f}%)")
            
            # Clinical reasoning statistics
            questions_with_answers = sum(
                1 for r in successful_results 
                if r.get("chain_of_thought", {}).get("chain_of_thought", {}).get("final_answer")
            )
            avg_confidence = sum(
                r.get("chain_of_thought", {}).get("chain_of_thought", {}).get("confidence_score", 0)
                for r in successful_results
            ) / successful_questions if successful_questions > 0 else 0
            
            print(f"\n🧠 Clinical Reasoning:")
            print(f"   Questions answered: {questions_with_answers}/{successful_questions} "
                  f"({questions_with_answers/successful_questions*100:.1f}%)")
            print(f"   Average confidence: {avg_confidence:.2f}")
            
            # Entity type analysis
            disease_count = 0
            drug_count = 0
            unknown_count = 0
            
            for result in successful_results:
                for entity in result.get("entity_extraction", {}).get("extracted_entities", []):
                    entity_type = entity.get('type', '').lower()
                    if entity_type == 'disease':
                        disease_count += 1
                    elif entity_type == 'drug':
                        drug_count += 1
                    else:
                        unknown_count += 1
            
            total_entities = disease_count + drug_count + unknown_count
            if total_entities > 0:
                print(f"\n🏥 Biomedical Entity Distribution:")
                print(f"   🦠 Diseases: {disease_count} ({disease_count/total_entities*100:.1f}%)")
                print(f"   💊 Drugs: {drug_count} ({drug_count/total_entities*100:.1f}%)")
                if unknown_count > 0:
                    print(f"   ❓ Unknown: {unknown_count} ({unknown_count/total_entities*100:.1f}%)")
            
            # Fertility relevance analysis
            fertility_questions = sum(
                1 for r in successful_results 
                if r.get("clinical_context", {}).get("fertility_relevance", {}).get("is_fertility_question", False)
            )
            
            print(f"\n🎯 Fertility Relevance:")
            print(f"   Fertility-related questions: {fertility_questions}/{successful_questions} "
                  f"({fertility_questions/successful_questions*100:.1f}%)")
        
        if error_results:
            print(f"\n❌ Errors ({len(error_results)} cases):")
            for error_result in error_results:
                print(f"   • {error_result.get('clinical_question', 'Unknown')[:50]}...")
                print(f"     Error: {error_result.get('error', 'Unknown error')}")
    
    def run_interactive_demo(self):
        """Run interactive demo with user input."""
        print("\n" + "="*80)
        print("INTERACTIVE BIOMEDICAL PIPELINE DEMO")
        print("="*80)
        print("\nThis demo allows you to test the clinical pipeline with your own questions.")
        print("Type 'exit' to quit, 'sample' for sample questions, or 'analyze' for analysis.\n")
        
        demo_results = []
        
        while True:
            user_input = input("\n📝 Enter a clinical question (or command): ").strip()
            
            if user_input.lower() == 'exit':
                print("\n👋 Exiting interactive demo.")
                break
            
            elif user_input.lower() == 'sample':
                print("\n📋 Sample Clinical Questions:")
                for i, test_case in enumerate(self.test_cases[:3], 1):
                    print(f"{i}. {test_case['question']}")
                print("\nCopy and paste any question to test it.")
                continue
            
            elif user_input.lower() == 'analyze':
                if demo_results:
                    print("\n📊 Analyzing collected results...")
                    self.analyzer.add_results(demo_results)
                    self.analyzer.analyze_property_matches(detailed=True)
                else:
                    print("\n❌ No results to analyze yet. Run some questions first.")
                continue
            
            elif user_input.lower() == 'help':
                print("\n📖 Available commands:")
                print("  exit     - Exit the demo")
                print("  sample   - Show sample questions")
                print("  analyze  - Analyze collected results")
                print("  help     - Show this help message")
                print("  clear    - Clear collected results")
                continue
            
            elif user_input.lower() == 'clear':
                demo_results = []
                self.analyzer.clear_results()
                print("\n🧹 Cleared all collected results.")
                continue
            
            elif not user_input:
                continue
            
            # Process the clinical question
            print(f"\n🔍 Processing: '{user_input}'")
            
            try:
                # Run pipeline
                start_time = time.time()
                result = self.pipeline.extract_and_map_entities_with_cot(user_input)
                execution_time = time.time() - start_time
                
                # Add to results
                demo_results.append(result)
                
                # Display quick summary
                self._display_interactive_summary(result, execution_time)
                
                # Ask if user wants detailed view
                show_details = input("\n📋 Show detailed results? (y/n): ").lower().strip()
                if show_details == 'y':
                    self._display_detailed_interactive_results(result)
                
                # Ask if user wants to see clinical answer
                show_answer = input("\n🧠 Show clinical answer? (y/n): ").lower().strip()
                if show_answer == 'y':
                    cot_result = result.get("chain_of_thought", {})
                    if cot_result and "chain_of_thought" in cot_result:
                        cot = cot_result["chain_of_thought"]
                        final_answer = cot.get('final_answer', '')
                        if final_answer:
                            print(f"\n📋 Clinical Answer: {final_answer}")
                        else:
                            print("\n❌ No clinical answer generated.")
                    else:
                        print("\n❌ No clinical reasoning available.")
                
            except Exception as e:
                print(f"\n❌ Error processing question: {e}")
                error_result = {
                    "clinical_question": user_input,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                demo_results.append(error_result)
        
        # Final summary
        if demo_results:
            successful_results = [r for r in demo_results if "error" not in r]
            print(f"\n📈 Interactive Demo Summary:")
            print(f"   Questions processed: {len(demo_results)}")
            print(f"   Successful: {len(successful_results)}")
            print(f"   Errors: {len(demo_results) - len(successful_results)}")
            
            if successful_results:
                print(f"\n💾 Results saved for analysis. Use 'analyze' command to view detailed statistics.")
        
        return demo_results
    
    def _display_interactive_summary(self, result: Dict, execution_time: float):
        """Display interactive summary for a result."""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        entity_extraction = result.get("entity_extraction", {})
        entity_mapping = result.get("entity_mapping", {})
        path_discovery = result.get("path_discovery", {})
        
        print(f"\n✅ Processing complete in {execution_time:.2f} seconds")
        print(f"📝 Entities extracted: {entity_extraction.get('total_entities', 0)}")
        print(f"🗺️  Entities mapped: {entity_mapping.get('total_mapped', 0)} "
              f"({entity_mapping.get('success_rate', 0):.1f}% success)")
        print(f"🛤️  Pathways found: {path_discovery.get('statistics', {}).get('total_paths_found', 0)}")
        
        # Show if fertility-related
        clinical_context = result.get("clinical_context", {})
        if clinical_context.get("fertility_relevance", {}).get("is_fertility_question", False):
            print(f"🎯 Fertility-related question detected")
    
    def _display_detailed_interactive_results(self, result: Dict):
        """Display detailed interactive results."""
        print("\n" + "="*60)
        print("DETAILED RESULTS")
        print("="*60)
        
        # Entity extraction details
        entity_extraction = result.get("entity_extraction", {})
        extracted_entities = entity_extraction.get("extracted_entities", [])
        
        print(f"\n📋 Extracted Entities:")
        for i, entity in enumerate(extracted_entities, 1):
            entity_type = entity.get('type', 'Unknown')
            entity_emoji = "🦠" if entity_type.lower() == "disease" else "💊" if entity_type.lower() == "drug" else "❓"
            print(f"  {i}. {entity_emoji} {entity['name']} ({entity_type}, confidence: {entity.get('confidence', 0.7):.2f})")
        
        # Entity mapping details
        entity_mapping = result.get("entity_mapping", {})
        successful_mappings = entity_mapping.get("successful_mappings", [])
        
        print(f"\n🗺️  Entity Mappings:")
        if successful_mappings:
            for i, mapping in enumerate(successful_mappings[:5], 1):  # Show first 5
                query_entity = mapping.get("query_entity", {})
                mapped_node = mapping.get("mapped_node", {})
                score = mapped_node.get('enhanced_score', 0)
                
                score_emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                print(f"  {i}. {query_entity.get('name', 'Unknown')} → {mapped_node.get('node_name', 'Unknown')}")
                print(f"     Score: {score_emoji} {score:.3f}, Match type: {mapped_node.get('match_type', 'unknown')}")
            
            if len(successful_mappings) > 5:
                print(f"     ... and {len(successful_mappings) - 5} more mappings")
        else:
            print("  ❌ No successful mappings")
        
        # Path discovery details
        path_discovery = result.get("path_discovery", {})
        selected_path = path_discovery.get("selected_path")
        
        print(f"\n🛤️  Selected Pathway:")
        if selected_path:
            path_desc = selected_path.get('detailed_description', selected_path.get('path_string', ''))
            print(f"  {path_desc}")
            print(f"  Length: {selected_path.get('path_length', 'N/A')} relationships")
            print(f"  Confidence: {path_discovery.get('selected_path_confidence', 0):.2f}")
        else:
            print("  ❌ No pathway selected")
    
    def generate_demo_report(self, filename: str = "clinical_pipeline_demo_report"):
        """
        Generate comprehensive demo report.
        
        Args:
            filename: Base filename for report
        """
        if not self.results:
            print("❌ No demo results to report. Run the demo first.")
            return
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate JSON report
            json_report = {
                "demo_timestamp": timestamp,
                "test_cases": self.test_cases,
                "results": self.results,
                "summary": self._generate_report_summary()
            }
            
            with open(f"{filename}_{timestamp}.json", 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            print(f"✅ JSON report saved to {filename}_{timestamp}.json")
            
            # Generate text report
            text_report = self._generate_text_report()
            with open(f"{filename}_{timestamp}.txt", 'w') as f:
                f.write(text_report)
            print(f"✅ Text report saved to {filename}_{timestamp}.txt")
            
            # Generate analysis if analyzer has results
            if self.analyzer.results:
                self.analyzer.export_analysis(f"{filename}_analysis_{timestamp}", "json")
                self.analyzer.export_analysis(f"{filename}_analysis_{timestamp}", "txt")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def _generate_report_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for report."""
        successful_results = [r for r in self.results if "error" not in r]
        
        summary = {
            "total_questions": len(self.test_cases),
            "successful_processing": len(successful_results),
            "failed_processing": len(self.results) - len(successful_results),
            "entity_extraction_stats": {},
            "entity_mapping_stats": {},
            "path_discovery_stats": {},
            "clinical_reasoning_stats": {}
        }
        
        if successful_results:
            # Entity extraction
            total_entities = sum(r.get("entity_extraction", {}).get("total_entities", 0) for r in successful_results)
            summary["entity_extraction_stats"]["total_entities"] = total_entities
            summary["entity_extraction_stats"]["avg_per_question"] = total_entities / len(successful_results)
            
            # Entity mapping
            total_mapped = sum(r.get("entity_mapping", {}).get("total_mapped", 0) for r in successful_results)
            mapping_rate = total_mapped / total_entities * 100 if total_entities > 0 else 0
            summary["entity_mapping_stats"]["total_mapped"] = total_mapped
            summary["entity_mapping_stats"]["mapping_success_rate"] = mapping_rate
            
            # Path discovery
            total_paths = sum(r.get("path_discovery", {}).get("statistics", {}).get("total_paths_found", 0) 
                            for r in successful_results)
            questions_with_paths = sum(1 for r in successful_results 
                                      if r.get("path_discovery", {}).get("statistics", {}).get("total_paths_found", 0) > 0)
            summary["path_discovery_stats"]["total_paths"] = total_paths
            summary["path_discovery_stats"]["questions_with_paths"] = questions_with_paths
            
            # Clinical reasoning
            questions_with_answers = sum(1 for r in successful_results 
                                        if r.get("chain_of_thought", {}).get("chain_of_thought", {}).get("final_answer"))
            avg_confidence = sum(r.get("chain_of_thought", {}).get("chain_of_thought", {}).get("confidence_score", 0)
                               for r in successful_results) / len(successful_results) if successful_results else 0
            summary["clinical_reasoning_stats"]["questions_with_answers"] = questions_with_answers
            summary["clinical_reasoning_stats"]["avg_confidence"] = avg_confidence
        
        return summary
    
    def _generate_text_report(self) -> str:
        """Generate comprehensive text report."""
        lines = []
        
        lines.append("="*80)
        lines.append("CLINICAL BIOMEDICAL PIPELINE DEMO REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Test Cases: {len(self.test_cases)}")
        lines.append("")
        
        # Test cases
        lines.append("TEST CASES:")
        lines.append("-" * 40)
        for i, test_case in enumerate(self.test_cases, 1):
            lines.append(f"{i}. {test_case['question']}")
            lines.append(f"   Category: {test_case['category'].replace('_', ' ').title()}")
            lines.append("")
        
        # Results summary
        successful_results = [r for r in self.results if "error" not in r]
        error_results = [r for r in self.results if "error" in r]
        
        lines.append("RESULTS SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"Total questions: {len(self.test_cases)}")
        lines.append(f"Successfully processed: {len(successful_results)}")
        lines.append(f"Failed: {len(error_results)}")
        lines.append("")
        
        if successful_results:
            # Detailed statistics
            lines.append("DETAILED STATISTICS:")
            lines.append("-" * 40)
            
            total_entities = sum(r.get("entity_extraction", {}).get("total_entities", 0) for r in successful_results)
            total_mapped = sum(r.get("entity_mapping", {}).get("total_mapped", 0) for r in successful_results)
            total_paths = sum(r.get("path_discovery", {}).get("statistics", {}).get("total_paths_found", 0) 
                            for r in successful_results)
            
            lines.append(f"Entities extracted: {total_entities}")
            lines.append(f"Entities mapped: {total_mapped}")
            if total_entities > 0:
                lines.append(f"Mapping success rate: {total_mapped/total_entities*100:.1f}%")
            lines.append(f"Pathways found: {total_paths}")
            lines.append("")
            
            # Individual results
            lines.append("INDIVIDUAL RESULTS:")
            lines.append("-" * 40)
            
            for i, (test_case, result) in enumerate(zip(self.test_cases, self.results), 1):
                lines.append(f"\n{i}. {test_case['question']}")
                
                if "error" in result:
                    lines.append(f"   ❌ ERROR: {result['error']}")
                else:
                    entity_extraction = result.get("entity_extraction", {})
                    entity_mapping = result.get("entity_mapping", {})
                    path_discovery = result.get("path_discovery", {})
                    cot_result = result.get("chain_of_thought", {})
                    
                    lines.append(f"   📝 Entities extracted: {entity_extraction.get('total_entities', 0)}")
                    lines.append(f"   🗺️  Entities mapped: {entity_mapping.get('total_mapped', 0)}")
                    lines.append(f"   🛤️  Pathways found: {path_discovery.get('statistics', {}).get('total_paths_found', 0)}")
                    
                    if cot_result and "chain_of_thought" in cot_result:
                        cot = cot_result["chain_of_thought"]
                        if cot.get("final_answer"):
                            lines.append(f"   🧠 Clinical answer generated: Yes")
                            lines.append(f"   Confidence: {cot.get('confidence_score', 0):.2f}")
        
        return "\n".join(lines)