# similarity.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher
import hashlib
import json
import time
from config import Config
from embedding_model import EmbeddingModel

class SimilarityCalculator:
    """Handles similarity calculations between entities and KG nodes with property-enhanced matching"""
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, llm_client=None):
        """
        Initialize the SimilarityCalculator
        
        Args:
            embedding_model: Optional EmbeddingModel instance for text similarity
            llm_client: LLM client for medical term expansion
        """
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.config = Config
        self.messages = []  # Store messages for Streamlit
        self.kg_embeddings = None  # Store KG embeddings
        self._initialize_weights()
        
        # Cache for medical term expansions
        self.expansion_cache = {}
        self.similarity_cache = {}
        
        # Enhanced medical knowledge base
        self.medical_knowledge = {
            # Diseases with known synonyms and related terms
            'diseases': {
                'pcos': {
                    'primary_names': ['Polycystic Ovary Syndrome', 'Polycystic Ovarian Syndrome', 
                                     'Stein-Leventhal Syndrome'],
                    'abbreviations': ['PCOS', 'PCO'],
                    'symptoms': ['irregular periods', 'hirsutism', 'acne', 'infertility', 
                                'weight gain', 'insulin resistance'],
                    'related_terms': ['hyperandrogenism', 'anovulation', 'metabolic syndrome',
                                     'endometrial hyperplasia', 'ovarian cysts']
                },
                'diabetes': {
                    'primary_names': ['Diabetes Mellitus', 'Type 2 Diabetes', 'Type 1 Diabetes'],
                    'abbreviations': ['DM', 'T2DM', 'T1DM'],
                    'types': ['Type 2 Diabetes', 'Type 1 Diabetes', 'Gestational Diabetes',
                             'MODY', 'LADA'],
                    'related_terms': ['hyperglycemia', 'insulin resistance', 'diabetic neuropathy',
                                     'diabetic retinopathy', 'diabetic nephropathy']
                }
            },
            
            # Drugs with known synonyms and related terms
            'drugs': {
                'clomiphene citrate': {
                    'primary_names': ['Clomiphene Citrate'],
                    'brand_names': ['Clomid', 'Serophene', 'Milophene'],
                    'generic_names': ['Clomiphene'],
                    'class': ['Selective Estrogen Receptor Modulator'],
                    'indications': ['infertility', 'anovulation', 'ovulation induction']
                },
                'metformin': {
                    'primary_names': ['Metformin'],
                    'brand_names': ['Glucophage', 'Fortamet', 'Glumetza', 'Riomet'],
                    'class': ['Biguanide'],
                    'indications': ['Type 2 Diabetes', 'Polycystic Ovary Syndrome', 
                                   'insulin resistance']
                }
            }
        }
        
        # Medical term patterns for fuzzy matching with weights
        self.medical_patterns = {
            'pcos': {
                'primary': ['polycystic ovary', 'polycystic ovarian', 'stein-leventhal'],
                'secondary': ['ovarian cyst', 'anovulation', 'hyperandrogenism', 
                            'menstrual irregularity', 'infertility'],
                'weight': 0.8
            },
            'clomiphene': {
                'primary': ['clomid', 'serophene', 'ovulation induction'],
                'secondary': ['fertility drug', 'selective estrogen receptor modulator',
                            'anovulation treatment'],
                'weight': 0.8
            },
            'metformin': {
                'primary': ['glucophage', 'biguanide', 'insulin sensitizer'],
                'secondary': ['diabetes medication', 'pcos treatment', 'weight loss'],
                'weight': 0.7
            },
            'diabetes': {
                'primary': ['diabetes mellitus', 'hyperglycemia', 'insulin'],
                'secondary': ['blood sugar', 'a1c', 'hemoglobin a1c', 'glucose'],
                'weight': 0.7
            },
            'syndrome': {
                'primary': ['syndrome', 'disorder', 'disease'],
                'secondary': ['condition', 'illness', 'medical condition'],
                'weight': 0.6
            }
        }
        
        # Load schema information for better property matching
        self._load_entity_schemas()
    
    def _initialize_weights(self):
        """Initialize property weights from config"""
        self.property_weights = self.config.ENTITY.PROPERTY_WEIGHTS
        self.entity_schemas = self.config.ENTITY.PROPERTY_SCHEMAS
    
    def _load_entity_schemas(self):
        """Load and cache entity schemas for faster access"""
        self.schema_cache = {}
        for entity_type, schema in self.entity_schemas.items():
            self.schema_cache[entity_type] = {
                'primary_keys': set(schema.get('primary_keys', [])),
                'display_props': set(schema.get('display_properties', [])),
                'all_props': set(schema.get('properties', [])),
                'aliases': schema.get('aliases', {}),
                'important_props': schema.get('important_properties', [])
            }
    
    def _add_message(self, message: str):
        """Add a message to the messages list"""
        self.messages.append(message)
        print(f"[Similarity] {message}")
    
    def clear_messages(self):
        """Clear all stored messages"""
        self.messages = []
    
    def get_messages(self) -> List[str]:
        """Get all stored messages"""
        return self.messages
    
    def set_embedding_model(self, embedding_model: EmbeddingModel):
        """Set embedding model for text similarity"""
        self.embedding_model = embedding_model
        self._add_message("✅ Embedding model configured for similarity calculations")
    
    def set_llm_client(self, llm_client):
        """Set LLM client for medical term expansion"""
        self.llm_client = llm_client
        self._add_message("✅ LLM client configured for medical term expansion")
    
    def set_kg_embeddings(self, kg_embeddings: np.ndarray):
        """Set KG embeddings for similarity calculations"""
        self.kg_embeddings = kg_embeddings
        if kg_embeddings is not None:
            self._add_message(f"✅ KG embeddings set: shape {kg_embeddings.shape}")
        else:
            self._add_message("⚠️ KG embeddings set to None")
    
    def _get_medical_knowledge_expansions(self, text: str, entity_type: str = "Unknown") -> List[str]:
        """Get expansions from medical knowledge base"""
        text_lower = text.lower().strip()
        expansions = [text]
        
        # Check in disease knowledge base
        if entity_type.lower() == 'disease' or entity_type == 'Unknown':
            for disease_key, disease_info in self.medical_knowledge['diseases'].items():
                if disease_key in text_lower or text_lower in disease_key:
                    expansions.extend(disease_info['primary_names'])
                    expansions.extend(disease_info.get('abbreviations', []))
                    expansions.extend(disease_info.get('related_terms', []))
        
        # Check in drug knowledge base
        if entity_type.lower() == 'drug' or entity_type == 'Unknown':
            for drug_key, drug_info in self.medical_knowledge['drugs'].items():
                if drug_key in text_lower or text_lower in drug_key:
                    expansions.extend(drug_info['primary_names'])
                    expansions.extend(drug_info.get('brand_names', []))
                    expansions.extend(drug_info.get('generic_names', []))
                    expansions.extend(drug_info.get('indications', []))
        
        return expansions
    
    def _expand_medical_terms_llm(self, text: str, entity_type: str = "Unknown") -> List[str]:
        """
        Expand medical terms using LLM for better synonym generation.
        
        Args:
            text: Medical term to expand
            entity_type: Type of entity (disease/drug)
            
        Returns:
            List of expanded terms and synonyms
        """
        # Create cache key
        cache_key = f"{text.lower()}_{entity_type}"
        
        # Check cache first
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key]
        
        # Start with medical knowledge base expansions
        expansions = self._get_medical_knowledge_expansions(text, entity_type)
        
        # Add the original text if not already present
        if text not in expansions:
            expansions.insert(0, text)
        
        text_lower = text.lower().strip()
        
        # Try LLM expansion if available
        if self.llm_client is not None:
            try:
                # Prepare enhanced prompt for LLM
                prompt = f"""As a medical expert, provide comprehensive synonyms and related terms for: "{text}" (entity type: {entity_type})
                
                IMPORTANT: For each term, indicate relevance score (0-1) where 1 is most relevant.
                Return as JSON with structure: {{"terms": [{{"term": "term", "relevance": 0.9, "type": "synonym/brand/abbreviation"}}]}}
                
                Include:
                1. Full medical names
                2. Common abbreviations
                3. Brand names (if drug)
                4. Related medical conditions/treatments
                5. Common misspellings
                6. Layman's terms
                
                Example for "PCOS" (disease):
                {{
                    "terms": [
                        {{"term": "Polycystic Ovary Syndrome", "relevance": 1.0, "type": "primary_name"}},
                        {{"term": "Polycystic Ovarian Syndrome", "relevance": 0.95, "type": "synonym"}},
                        {{"term": "Stein-Leventhal Syndrome", "relevance": 0.8, "type": "historical_name"}},
                        {{"term": "PCO", "relevance": 0.7, "type": "abbreviation"}}
                    ]
                }}
                
                Example for "Clomiphene citrate" (drug):
                {{
                    "terms": [
                        {{"term": "Clomid", "relevance": 0.95, "type": "brand_name"}},
                        {{"term": "Serophene", "relevance": 0.9, "type": "brand_name"}},
                        {{"term": "Clomiphene", "relevance": 0.85, "type": "generic_name"}},
                        {{"term": "Selective Estrogen Receptor Modulator", "relevance": 0.7, "type": "drug_class"}}
                    ]
                }}
                
                Now for "{text}" (entity type: {entity_type}):"""
                
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.2
                )
                
                # Try to parse JSON response
                try:
                    # Clean response
                    response_text = response.strip()
                    if response_text.startswith('```json'):
                        response_text = response_text[7:-3]
                    elif response_text.startswith('```'):
                        response_text = response_text[3:-3]
                    
                    result = json.loads(response_text)
                    if 'terms' in result and isinstance(result['terms'], list):
                        llm_terms = []
                        for item in result['terms']:
                            if isinstance(item, dict) and 'term' in item:
                                term = item['term']
                                relevance = item.get('relevance', 0.5)
                                # Only include high relevance terms
                                if relevance >= 0.6:
                                    llm_terms.append(term)
                        
                        expansions.extend(llm_terms)
                        self._add_message(f"📚 LLM expanded '{text}' with {len(llm_terms)} high-relevance terms")
                        
                except json.JSONDecodeError:
                    # Fallback: extract terms from text
                    lines = response.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        # Look for quoted terms or capitalized medical terms
                        if '"' in line:
                            term = line.split('"')[1]
                            if term and term not in expansions:
                                expansions.append(term)
                        elif line and line[0].isupper() and len(line.split()) <= 5:
                            if line not in expansions:
                                expansions.append(line)
                
            except Exception as e:
                self._add_message(f"⚠️ LLM expansion failed for '{text}': {e}")
        
        # Add entity type specific expansions
        if entity_type.lower() == 'disease':
            # Common disease patterns
            if any(pattern in text_lower for pattern in ['syndrome', 'disease', 'disorder']):
                # Remove suffix to get base term
                base_term = re.sub(r'\s+(syndrome|disease|disorder)$', '', text_lower, flags=re.IGNORECASE)
                if base_term and base_term != text_lower:
                    expansions.append(base_term.title())
            
            # Add "syndrome" to terms that don't have it
            if 'syndrome' not in text_lower and 'disease' not in text_lower and 'disorder' not in text_lower:
                expansions.append(f"{text} Syndrome")
                expansions.append(f"{text} Disease")
        
        elif entity_type.lower() == 'drug':
            # Common drug patterns
            suffixes = [' citrate', ' hydrochloride', ' sulfate', ' sodium', ' maleate']
            for suffix in suffixes:
                if text_lower.endswith(suffix):
                    base = text_lower.replace(suffix, '')
                    expansions.append(base.title())
                    break
        
        # Add pattern-based expansions
        for pattern_key, pattern_info in self.medical_patterns.items():
            if pattern_key in text_lower:
                for primary_term in pattern_info['primary']:
                    if primary_term not in text_lower:
                        expansions.append(f"{text} ({primary_term})")
                for secondary_term in pattern_info['secondary']:
                    expansions.append(secondary_term)
        
        # Remove duplicates but preserve order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            exp_lower = exp.lower()
            if exp_lower not in seen:
                seen.add(exp_lower)
                unique_expansions.append(exp)
        
        # Cache the results
        self.expansion_cache[cache_key] = unique_expansions
        
        self._add_message(f"📚 Expanded '{text}' to {len(unique_expansions)} unique terms")
        return unique_expansions
    
    def _expand_medical_terms(self, text: str, entity_type: str = "Unknown") -> List[str]:
        """Wrapper method for medical term expansion"""
        return self._expand_medical_terms_llm(text, entity_type)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove parenthetical content but keep it for reference
        text_without_parentheses = re.sub(r'\([^)]*\)', '', text)
        
        # Remove common prefixes/suffixes
        text_without_parentheses = re.sub(r'^(the|a|an)\s+', '', text_without_parentheses)
        
        # Standardize common medical abbreviations
        replacements = {
            ' vs ': ' versus ',
            ' w/': ' with ',
            ' w/o': ' without ',
            ' & ': ' and ',
            ' + ': ' and ',
            ' dr.': ' doctor ',
            ' dr ': ' doctor ',
            ' mg ': ' milligram ',
            ' g ': ' gram ',
            ' kg ': ' kilogram ',
            ' ml ': ' milliliter ',
            ' l ': ' liter ',
        }
        
        for old, new in replacements.items():
            text_without_parentheses = text_without_parentheses.replace(old, new)
        
        # Remove special characters but keep alphanumeric and spaces
        text_without_parentheses = re.sub(r'[^\w\s\-]', ' ', text_without_parentheses)
        
        # Remove extra whitespace again
        text_without_parentheses = ' '.join(text_without_parentheses.split())
        
        return text_without_parentheses.strip()
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using multiple methods with caching"""
        # Create cache key
        cache_key = f"{str1.lower()}_{str2.lower()}"
        
        # Check cache
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        if not str1 or not str2:
            return 0.0
        
        str1_norm = self._normalize_text(str1)
        str2_norm = self._normalize_text(str2)
        
        # Exact match after normalization
        if str1_norm == str2_norm:
            self.similarity_cache[cache_key] = 1.0
            return 1.0
        
        # Check if one contains the other (with context)
        if str1_norm in str2_norm or str2_norm in str1_norm:
            # Calculate overlap percentage
            if str1_norm in str2_norm:
                overlap_ratio = len(str1_norm) / len(str2_norm)
            else:
                overlap_ratio = len(str2_norm) / len(str1_norm)
            
            # Higher score for better overlap
            score = 0.7 + (0.3 * overlap_ratio)
            self.similarity_cache[cache_key] = score
            return score
        
        # Use SequenceMatcher for fuzzy matching
        ratio = SequenceMatcher(None, str1_norm, str2_norm).ratio()
        
        # Token-based similarity with weighting
        tokens1 = set(str1_norm.split())
        tokens2 = set(str2_norm.split())
        
        if tokens1 and tokens2:
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            # Calculate weighted Jaccard (common medical terms get higher weight)
            medical_terms = set(['syndrome', 'disease', 'disorder', 'therapy', 'treatment', 
                                'medication', 'drug', 'tablet', 'capsule', 'injection'])
            
            weighted_intersection = 0
            weighted_union = 0
            
            for token in union:
                weight = 2.0 if token in medical_terms else 1.0
                if token in intersection:
                    weighted_intersection += weight
                weighted_union += weight
            
            jaccard = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
            
            # Weighted combination of methods (favor Jaccard for medical text)
            combined = (ratio * 0.3) + (jaccard * 0.7)
            self.similarity_cache[cache_key] = combined
            return combined
        
        self.similarity_cache[cache_key] = ratio
        return ratio
    
    def _get_entity_embedding(self, entity_name: str) -> Optional[np.ndarray]:
        """
        Get embedding for an entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            if self.embedding_model is None:
                return None
            
            # Use embedding model
            embedding, _ = self.embedding_model.encode_texts([entity_name], show_progress_bar=False)
            if len(embedding) > 0 and embedding[0] is not None:
                return embedding[0]
            else:
                return None
                
        except Exception as e:
            self._add_message(f"❌ Error getting embedding for '{entity_name}': {e}")
            return None
    
    def _calculate_pattern_bonus(self, entity_name: str, candidate_name: str, entity_type: str) -> float:
        """Calculate bonus score based on medical pattern matching"""
        entity_lower = entity_name.lower()
        candidate_lower = candidate_name.lower()
        
        bonus = 0.0
        
        # Check for known medical patterns
        for pattern_key, pattern_info in self.medical_patterns.items():
            if pattern_key in entity_lower:
                # Check primary pattern matches
                for primary_term in pattern_info['primary']:
                    if primary_term in candidate_lower:
                        bonus += pattern_info['weight'] * 0.3
                
                # Check secondary pattern matches
                for secondary_term in pattern_info['secondary']:
                    if secondary_term in candidate_lower:
                        bonus += pattern_info['weight'] * 0.15
        
        # Entity type specific bonuses
        if entity_type.lower() == 'disease':
            if 'syndrome' in entity_lower and 'syndrome' in candidate_lower:
                bonus += 0.1
            if 'disease' in entity_lower and 'disease' in candidate_lower:
                bonus += 0.1
        
        elif entity_type.lower() == 'drug':
            if any(suffix in entity_lower for suffix in [' citrate', ' hydrochloride', ' sulfate']):
                if any(suffix in candidate_lower for suffix in [' citrate', ' hydrochloride', ' sulfate']):
                    bonus += 0.1
        
        return min(bonus, 0.3)  # Cap bonus at 0.3
    
    def calculate_property_similarity(
        self, 
        entity_name: str, 
        candidate: Dict, 
        entity_type: str
    ) -> Tuple[float, str, Dict]:
        """
        Calculate enhanced similarity score based on properties.
        
        Args:
            entity_name: Name of the entity to match
            candidate: Candidate KG node dictionary
            entity_type: Type of entity (disease/drug)
            
        Returns:
            Tuple of (enhanced_score, match_type, property_info)
        """
        try:
            # Get entity variations using enhanced expansion
            entity_variations = self._expand_medical_terms(entity_name, entity_type)
            
            best_score = 0.0
            best_match_type = "none"
            best_property_info = {}
            
            # Get candidate details
            candidate_name = candidate.get("node_name", "")
            candidate_props = candidate.get("properties", {})
            
            # Get schema for this entity type
            schema = self.schema_cache.get(entity_type, self.schema_cache.get('Unknown', {}))
            primary_keys = schema.get('primary_keys', set())
            display_props = schema.get('display_properties', set())
            important_props = schema.get('important_props', [])
            
            # Track best matches for each variation
            for entity_variant in entity_variations[:10]:  # Limit to top 10 variations
                entity_norm = self._normalize_text(entity_variant)
                candidate_name_norm = self._normalize_text(candidate_name)
                
                # 1. Check exact name match (highest priority)
                if candidate_name_norm and candidate_name_norm == entity_norm:
                    match_type = "exact_name_match"
                    enhanced_score = 0.98  # Very high score for exact match
                    
                    # Pattern bonus
                    pattern_bonus = self._calculate_pattern_bonus(entity_name, candidate_name, entity_type)
                    enhanced_score = min(enhanced_score + pattern_bonus, 1.0)
                    
                    property_info = {
                        "match_type": match_type,
                        "matched_property": "node_name",
                        "property_value": candidate_name,
                        "original_entity": entity_name,
                        "matched_variant": entity_variant,
                        "pattern_bonus": pattern_bonus,
                        "score_breakdown": {"name_exact": self.property_weights["name_exact"]},
                        "confidence": "very_high"
                    }
                    if enhanced_score > best_score:
                        best_score = enhanced_score
                        best_match_type = match_type
                        best_property_info = property_info
                
                # 2. Check exact primary key matches
                for prop in primary_keys:
                    if prop in candidate_props:
                        prop_value = str(candidate_props[prop])
                        if self._normalize_text(prop_value) == entity_norm:
                            match_type = f"exact_primary_key:{prop}"
                            enhanced_score = 0.95
                            
                            pattern_bonus = self._calculate_pattern_bonus(entity_name, prop_value, entity_type)
                            enhanced_score = min(enhanced_score + pattern_bonus, 1.0)
                            
                            property_info = {
                                "match_type": match_type,
                                "matched_property": prop,
                                "property_value": candidate_props[prop],
                                "original_entity": entity_name,
                                "matched_variant": entity_variant,
                                "pattern_bonus": pattern_bonus,
                                "score_breakdown": {"exact_primary_key": self.property_weights["exact_primary_key"]},
                                "confidence": "very_high"
                            }
                            if enhanced_score > best_score:
                                best_score = enhanced_score
                                best_match_type = match_type
                                best_property_info = property_info
                
                # 3. Check contains name match with context
                if candidate_name_norm and (entity_norm in candidate_name_norm or candidate_name_norm in entity_norm):
                    match_type = "contains_name_match"
                    
                    # Calculate overlap with context
                    if entity_norm in candidate_name_norm:
                        overlap_ratio = len(entity_norm) / len(candidate_name_norm)
                        # Bonus if entity is the main part of candidate name
                        if overlap_ratio > 0.5:
                            overlap_bonus = 0.1
                        else:
                            overlap_bonus = 0.0
                    else:
                        overlap_ratio = len(candidate_name_norm) / len(entity_norm)
                        overlap_bonus = 0.0
                    
                    pattern_bonus = self._calculate_pattern_bonus(entity_name, candidate_name, entity_type)
                    
                    enhanced_score = 0.75 + (0.15 * overlap_ratio) + overlap_bonus + pattern_bonus
                    enhanced_score = min(enhanced_score, 0.97)  # Cap below exact match
                    
                    property_info = {
                        "match_type": match_type,
                        "matched_property": "node_name",
                        "property_value": candidate_name,
                        "original_entity": entity_name,
                        "matched_variant": entity_variant,
                        "overlap_ratio": overlap_ratio,
                        "overlap_bonus": overlap_bonus,
                        "pattern_bonus": pattern_bonus,
                        "score_breakdown": {"name_contains": self.property_weights["name_contains"]},
                        "confidence": "high" if enhanced_score > 0.85 else "medium"
                    }
                    if enhanced_score > best_score:
                        best_score = enhanced_score
                        best_match_type = match_type
                        best_property_info = property_info
                
                # 4. Check fuzzy string similarity for names with medical context
                name_similarity = self._calculate_string_similarity(entity_variant, candidate_name)
                if name_similarity > 0.65:  # Higher threshold for fuzzy matches
                    match_type = "fuzzy_name_match"
                    
                    pattern_bonus = self._calculate_pattern_bonus(entity_name, candidate_name, entity_type)
                    
                    # Scale similarity with pattern bonus
                    enhanced_score = (name_similarity * 0.85) + pattern_bonus
                    
                    property_info = {
                        "match_type": match_type,
                        "matched_property": "node_name",
                        "property_value": candidate_name,
                        "original_entity": entity_name,
                        "matched_variant": entity_variant,
                        "name_similarity": name_similarity,
                        "pattern_bonus": pattern_bonus,
                        "score_breakdown": {"fuzzy_name": name_similarity},
                        "confidence": "medium" if enhanced_score >= 0.75 else "low"
                    }
                    if enhanced_score > best_score:
                        best_score = enhanced_score
                        best_match_type = match_type
                        best_property_info = property_info
            
            # If we found a strong match, return it
            if best_score >= 0.7:
                return best_score, best_match_type, best_property_info
            
            # 5. Comprehensive property matching for weaker matches
            base_similarity = candidate.get("similarity_score", 0.0)
            entity_norm = self._normalize_text(entity_name)
            
            property_score = 0.0
            max_possible_score = 0.0
            matched_properties = []
            property_breakdown = {}
            
            # Check all candidate properties with priority for important ones
            for prop, value in candidate_props.items():
                if not isinstance(value, str):
                    continue
                
                prop_norm = self._normalize_text(value)
                
                # Determine property weight based on importance
                if prop in primary_keys:
                    weight = self.property_weights["contains_primary_key"] * 1.2
                elif prop in important_props:
                    weight = self.property_weights["contains_display_property"] * 1.1
                elif prop in display_props:
                    weight = self.property_weights["contains_display_property"]
                else:
                    weight = self.property_weights["other_property_match"]
                
                max_possible_score += weight
                
                # Check contains match
                if entity_norm in prop_norm or prop_norm in entity_norm:
                    property_score += weight
                    matched_properties.append(prop)
                    property_breakdown[prop] = weight
                else:
                    # Check fuzzy similarity
                    fuzzy_sim = self._calculate_string_similarity(entity_norm, prop_norm)
                    if fuzzy_sim > 0.5:  # Higher threshold for property fuzzy matches
                        property_score += weight * fuzzy_sim
                        matched_properties.append(f"{prop}_fuzzy")
                        property_breakdown[f"{prop}_fuzzy"] = weight * fuzzy_sim
            
            # Calculate property boost
            if max_possible_score > 0:
                property_boost = property_score / max_possible_score
            else:
                property_boost = 0
            
            # Dynamic weight adjustment based on match quality
            if property_boost > 0.6:
                property_weight = 0.8
            elif property_boost > 0.3:
                property_weight = 0.7
            elif property_boost > 0:
                property_weight = 0.5
            else:
                property_weight = 0.3
            
            text_weight = 1 - property_weight
            
            # Calculate enhanced score
            if base_similarity < 0.3:
                enhanced_score = property_boost * 0.9 + base_similarity * 0.1
            else:
                enhanced_score = (base_similarity * text_weight) + (property_boost * property_weight)
            
            # Add pattern bonus
            pattern_bonus = self._calculate_pattern_bonus(entity_name, candidate_name, entity_type)
            enhanced_score = min(enhanced_score + pattern_bonus, 1.0)
            
            # Determine match type
            if matched_properties:
                match_type = "property_enhanced"
                if any("_fuzzy" in prop for prop in matched_properties):
                    match_type = "fuzzy_property_match"
            else:
                match_type = "text_similarity_only"
            
            # Determine confidence level
            if enhanced_score >= 0.85:
                confidence = "very_high"
            elif enhanced_score >= 0.7:
                confidence = "high"
            elif enhanced_score >= 0.55:
                confidence = "medium"
            elif enhanced_score >= 0.4:
                confidence = "low"
            else:
                confidence = "very_low"
            
            property_info = {
                "match_type": match_type,
                "matched_properties": matched_properties,
                "property_boost": property_boost,
                "property_score": property_score,
                "max_possible_score": max_possible_score,
                "property_breakdown": property_breakdown,
                "original_entity": entity_name,
                "pattern_bonus": pattern_bonus,
                "score_breakdown": {
                    "base_similarity": base_similarity,
                    "property_boost": property_boost,
                    "pattern_bonus": pattern_bonus,
                    "final_score": enhanced_score
                },
                "weights": {
                    "property_weight": property_weight,
                    "text_weight": text_weight
                },
                "confidence": confidence
            }
            
            return enhanced_score, match_type, property_info
            
        except Exception as e:
            self._add_message(f"❌ Error in calculate_property_similarity: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, "error", {"error": str(e)}
    
    def get_enhanced_candidates(
        self, 
        entity_name: str, 
        entity_type: str, 
        kg_nodes: List[Dict],
        kg_embeddings: np.ndarray,
        top_k: int = None
    ) -> Tuple[List[Dict], List[str]]:
        """
        Find top-K most similar KG nodes with property-enhanced matching.
        
        Args:
            entity_name: Name of the entity to match
            entity_type: Type of entity (disease/drug)
            kg_nodes: List of KG node dictionaries
            kg_embeddings: Numpy array of KG node embeddings
            top_k: Number of top candidates to return
            
        Returns:
            Tuple of (enhanced_candidates, messages)
        """
        self.clear_messages()
        
        if not entity_name or not entity_name.strip():
            self._add_message("❌ No entity name provided for similarity calculation")
            return [], self.messages
        
        if not kg_nodes or kg_embeddings is None or len(kg_embeddings) == 0:
            self._add_message("❌ No KG data available for similarity calculation")
            return [], self.messages
        
        self._add_message(f"🔍 Finding similar candidates for '{entity_name}' (type: {entity_type})")
        
        # Strategy 1: Direct property matching with enhanced filtering
        self._add_message("📊 Attempting intelligent direct property matching...")
        direct_matches = self._get_intelligent_direct_matches(entity_name, entity_type, kg_nodes, top_k=50)
        
        if direct_matches:
            # Filter to only high-quality matches
            high_quality_matches = [m for m in direct_matches if m.get("enhanced_score", 0) > 0.7]
            
            if high_quality_matches:
                self._add_message(f"✅ Found {len(high_quality_matches)} high-quality direct matches")
                if top_k is None:
                    top_k = self.config.PIPELINE.TOP_K
                return high_quality_matches[:top_k], self.messages
            else:
                self._add_message(f"⚠️ Found {len(direct_matches)} direct matches, but none are high-quality")
        
        # Strategy 2: Embedding-based matching (if available)
        if self.embedding_model is not None:
            self._add_message("📊 Using embedding-based matching...")
            text_candidates = self._get_text_similarity_candidates(
                entity_name, kg_nodes, kg_embeddings
            )
            
            if text_candidates:
                self._add_message(f"📊 Found {len(text_candidates)} embedding-based candidates")
                
                # Filter by entity type with priority
                if entity_type != "Unknown":
                    type_filtered = [
                        c for c in text_candidates 
                        if entity_type.lower() == c.get("entity_type", "unknown").lower()
                    ]
                    if type_filtered:
                        text_candidates = type_filtered
                        self._add_message(f"📊 Filtered to {len(text_candidates)} candidates of type '{entity_type}'")
                
                # Enhance scores with property matching
                enhanced_candidates = []
                for candidate in text_candidates:
                    enhanced_score, match_type, property_info = self.calculate_property_similarity(
                        entity_name, candidate, entity_type
                    )
                    
                    enhanced_candidate = {
                        "kg_index": candidate.get("kg_index", -1),
                        "node_id": candidate.get("node_id", "unknown"),
                        "node_name": candidate.get("node_name", "Unknown"),
                        "labels": candidate.get("labels", []),
                        "entity_type": candidate.get("entity_type", "unknown"),
                        "similarity_score": candidate.get("similarity_score", 0.0),
                        "enhanced_score": enhanced_score,
                        "match_type": match_type,
                        "properties": candidate.get("properties", {}),
                        "property_text": candidate.get("property_text", ""),
                        "text_length": candidate.get("text_length", 0),
                        "property_info": property_info
                    }
                    enhanced_candidates.append(enhanced_candidate)
                
                # Sort and take top-k
                enhanced_candidates.sort(key=lambda x: x["enhanced_score"], reverse=True)
                
                if top_k is None:
                    top_k = self.config.PIPELINE.TOP_K
                
                top_candidates = enhanced_candidates[:top_k]
                
                if top_candidates and top_candidates[0].get("enhanced_score", 0) > 0.6:
                    self._add_message(f"✅ Found {len(top_candidates)} enhanced candidates (top {top_k})")
                    best_candidate = top_candidates[0]
                    self._add_message(f"🏆 Best match: {best_candidate['node_name']} "
                                    f"(score: {best_candidate['enhanced_score']:.3f}, "
                                    f"type: {best_candidate['match_type']})")
                    return top_candidates, self.messages
        
        # Strategy 3: Context-aware fallback matching
        self._add_message("🔄 Using context-aware fallback matching...")
        fallback_candidates = self._context_aware_fallback_matching(entity_name, entity_type, kg_nodes, top_k)
        
        if fallback_candidates:
            self._add_message(f"✅ Found {len(fallback_candidates)} fallback candidates")
            if top_k is None:
                top_k = self.config.PIPELINE.TOP_K
            return fallback_candidates[:top_k], self.messages
        
        self._add_message("❌ No matches found with any method")
        return [], self.messages
    
    def _get_intelligent_direct_matches(
        self,
        entity_name: str,
        entity_type: str,
        kg_nodes: List[Dict],
        top_k: int = 50
    ) -> List[Dict]:
        """Find direct property matches with intelligent filtering"""
        # Get entity variations using enhanced expansion
        entity_variations = self._expand_medical_terms(entity_name, entity_type)
        
        candidates = []
        self._add_message(f"📊 Checking {len(entity_variations)} entity variations")
        
        # Get schema
        schema = self.schema_cache.get(entity_type, self.schema_cache.get('Unknown', {}))
        primary_keys = schema.get('primary_keys', set())
        display_props = schema.get('display_properties', set())
        important_props = schema.get('important_props', [])
        
        entity_lower = entity_name.lower()
        
        for idx, node in enumerate(kg_nodes):
            node_name = node.get("node_name", "")
            node_type = node.get("entity_type", "").lower()
            
            # Filter by type
            if entity_type != "Unknown" and entity_type.lower() != node_type:
                continue
            
            node_lower = node_name.lower()
            
            # Skip generic or irrelevant nodes
            if self._is_generic_node(node_name, node_lower, entity_type):
                continue
            
            best_variant_score = 0.0
            best_variant = entity_name
            matched_properties = []
            
            # Check each entity variation (prioritize original and primary variations)
            priority_variations = []
            for variant in entity_variations:
                variant_lower = variant.lower()
                # Prioritize variations that contain key medical terms
                if any(term in variant_lower for term in ['syndrome', 'disease', 'citrate', 'hydrochloride']):
                    priority_variations.insert(0, variant)
                else:
                    priority_variations.append(variant)
            
            for entity_variant in priority_variations[:15]:  # Limit to top 15 variations
                entity_norm = self._normalize_text(entity_variant)
                
                # Name similarity with pattern bonus
                name_similarity = self._calculate_string_similarity(entity_variant, node_name)
                pattern_bonus = self._calculate_pattern_bonus(entity_name, node_name, entity_type)
                name_score = min(name_similarity + pattern_bonus, 1.0)
                
                # Property matching
                properties = node.get("properties", {})
                property_matches = []
                property_score = 0.0
                
                for prop, value in properties.items():
                    if not isinstance(value, str):
                        continue
                    
                    value_norm = self._normalize_text(value)
                    
                    # Exact match (highest score)
                    if entity_norm == value_norm:
                        if prop in primary_keys:
                            weight = 1.0
                        elif prop in important_props:
                            weight = 0.9
                        elif prop in display_props:
                            weight = 0.8
                        else:
                            weight = 0.7
                        
                        property_matches.append({
                            "property": prop,
                            "value": value,
                            "match_type": "exact",
                            "weight": weight
                        })
                        property_score += weight
                    
                    # Contains match
                    elif entity_norm in value_norm or value_norm in entity_norm:
                        if prop in primary_keys:
                            weight = 0.8
                        elif prop in important_props:
                            weight = 0.7
                        elif prop in display_props:
                            weight = 0.6
                        else:
                            weight = 0.5
                        
                        property_matches.append({
                            "property": prop,
                            "value": value,
                            "match_type": "contains",
                            "weight": weight
                        })
                        property_score += weight
                
                # Calculate variant score with intelligent weighting
                if property_matches:
                    # Property matches are more important, but consider match quality
                    avg_property_weight = sum(pm["weight"] for pm in property_matches) / len(property_matches)
                    property_quality = avg_property_weight  # Higher weight = better match
                    
                    variant_score = (property_score * 0.7 * property_quality) + (name_score * 0.3)
                else:
                    variant_score = name_score
                
                if variant_score > best_variant_score:
                    best_variant_score = variant_score
                    best_variant = entity_variant
                    matched_properties = property_matches
            
            # Include candidate if score is reasonable and not generic
            if best_variant_score > 0.5 and not self._is_generic_match(entity_name, node_name, best_variant_score):
                match_type = "intelligent_direct_match"
                if best_variant != entity_name:
                    match_type = f"{match_type}_expanded"
                
                # Add final pattern bonus
                final_pattern_bonus = self._calculate_pattern_bonus(entity_name, node_name, entity_type)
                best_variant_score = min(best_variant_score + final_pattern_bonus, 1.0)
                
                candidate = {
                    "kg_index": int(idx),
                    "node_id": node.get("node_id", f"node_{idx}"),
                    "node_name": node_name,
                    "labels": node.get("labels", []),
                    "entity_type": node.get("entity_type", "unknown"),
                    "similarity_score": self._calculate_string_similarity(entity_name, node_name),
                    "enhanced_score": best_variant_score,
                    "match_type": match_type,
                    "properties": properties,
                    "property_text": node.get("property_text", ""),
                    "text_length": node.get("text_length", 0),
                    "property_info": {
                        "match_type": match_type,
                        "matched_variant": best_variant,
                        "matched_properties": [pm["property"] for pm in matched_properties],
                        "property_matches": matched_properties,
                        "property_score": sum(pm.get("weight", 0) for pm in matched_properties),
                        "pattern_bonus": final_pattern_bonus,
                        "confidence": "high" if best_variant_score > 0.75 else "medium" if best_variant_score > 0.6 else "low"
                    }
                }
                candidates.append(candidate)
        
        # Sort and filter candidates
        candidates.sort(key=lambda x: x["enhanced_score"], reverse=True)
        
        # Remove duplicates or very similar candidates
        unique_candidates = []
        seen_names = set()
        
        for candidate in candidates:
            candidate_name = candidate["node_name"].lower()
            if candidate_name not in seen_names:
                seen_names.add(candidate_name)
                unique_candidates.append(candidate)
        
        if unique_candidates:
            self._add_message(f"📊 Intelligent direct matching found {len(unique_candidates)} unique candidates")
        
        return unique_candidates[:top_k]
    
    def _is_generic_node(self, node_name: str, node_lower: str, entity_type: str) -> bool:
        """Check if a node is too generic to be a good match"""
        generic_terms = {
            'disease': ['syndrome', 'disease', 'disorder', 'condition', 'illness'],
            'drug': ['drug', 'medication', 'therapy', 'treatment', 'agent']
        }
        
        # Single word nodes are often too generic
        if len(node_name.split()) == 1:
            return True
        
        # Check for overly generic names
        generic_words = generic_terms.get(entity_type.lower(), [])
        if any(word == node_lower for word in generic_words):
            return True
        
        # Check for patterns like "X syndrome" where X is single letter
        if re.match(r'^[A-Z]\s+syndrome$', node_name, re.IGNORECASE):
            return True
        
        return False
    
    def _is_generic_match(self, entity_name: str, candidate_name: str, score: float) -> bool:
        """Check if a match is too generic"""
        entity_lower = entity_name.lower()
        candidate_lower = candidate_name.lower()
        
        # If entity is specific but candidate is generic, reject even with high score
        specific_terms = ['pcos', 'clomiphene', 'metformin', 'insulin']
        generic_candidates = ['syndrome', 'disease', 'drug', 'medication', 'therapy']
        
        if any(term in entity_lower for term in specific_terms):
            if any(term in candidate_lower for term in generic_candidates):
                if len(candidate_name.split()) <= 2:  # Very short generic names
                    return True
        
        return False
    
    def _context_aware_fallback_matching(
        self,
        entity_name: str,
        entity_type: str,
        kg_nodes: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """Context-aware fallback matching using medical knowledge"""
        entity_variations = self._expand_medical_terms(entity_name, entity_type)
        candidates = []
        
        self._add_message(f"🔄 Context-aware fallback with {len(entity_variations)} variations")
        
        entity_lower = entity_name.lower()
        
        # Get relevant medical knowledge
        relevant_patterns = []
        for pattern_key, pattern_info in self.medical_patterns.items():
            if pattern_key in entity_lower:
                relevant_patterns.append((pattern_key, pattern_info))
        
        for idx, node in enumerate(kg_nodes):
            node_name = node.get("node_name", "")
            if not isinstance(node_name, str):
                continue
            
            node_norm = self._normalize_text(node_name)
            node_type = node.get("entity_type", "").lower()
            
            # Skip generic nodes
            if self._is_generic_node(node_name, node_norm, entity_type):
                continue
            
            best_score = 0.0
            best_variant = entity_name
            
            # Check each entity variation
            for entity_variant in entity_variations[:10]:  # Limit to top 10
                entity_norm = self._normalize_text(entity_variant)
                
                # Calculate similarity
                similarity = self._calculate_string_similarity(entity_norm, node_norm)
                
                # Type bonus
                type_match = entity_type.lower() == node_type if entity_type != "Unknown" else True
                type_bonus = 0.15 if type_match else 0.0
                
                variant_score = min(similarity + type_bonus, 1.0)
                
                if variant_score > best_score:
                    best_score = variant_score
                    best_variant = entity_variant
            
            # Medical pattern bonus
            pattern_bonus = 0.0
            for pattern_key, pattern_info in relevant_patterns:
                # Check primary pattern matches
                for primary_term in pattern_info['primary']:
                    if primary_term in node_norm:
                        pattern_bonus += pattern_info['weight'] * 0.25
                
                # Check secondary pattern matches
                for secondary_term in pattern_info['secondary']:
                    if secondary_term in node_norm:
                        pattern_bonus += pattern_info['weight'] * 0.15
            
            best_score = min(best_score + pattern_bonus, 1.0)
            
            # Include if score is reasonable and not generic
            if best_score > 0.5 and not self._is_generic_match(entity_name, node_name, best_score):
                match_type = "context_aware_fallback"
                if best_variant != entity_name:
                    match_type = f"{match_type}_expanded"
                
                # Special boosts for known medical patterns
                if 'pcos' in entity_lower and any(pattern in node_norm for pattern in ['polycystic', 'ovary', 'ovarian']):
                    best_score = max(best_score, 0.7)
                    match_type = "medical_pattern_match"
                
                if 'clomiphene' in entity_lower and any(pattern in node_norm for pattern in ['clomid', 'fertility', 'ovulation']):
                    best_score = max(best_score, 0.7)
                    match_type = "drug_pattern_match"
                
                candidate = {
                    "kg_index": int(idx),
                    "node_id": node.get("node_id", f"node_{idx}"),
                    "node_name": node_name,
                    "labels": node.get("labels", []),
                    "entity_type": node.get("entity_type", "unknown"),
                    "similarity_score": self._calculate_string_similarity(entity_name, node_name),
                    "enhanced_score": best_score,
                    "match_type": match_type,
                    "properties": node.get("properties", {}),
                    "property_text": node.get("property_text", ""),
                    "text_length": node.get("text_length", 0),
                    "property_info": {
                        "match_type": match_type,
                        "matched_variant": best_variant,
                        "pattern_bonus": pattern_bonus,
                        "confidence": "high" if best_score >= 0.7 else "medium" if best_score >= 0.6 else "low"
                    }
                }
                candidates.append(candidate)
        
        # Sort and return
        candidates.sort(key=lambda x: x["enhanced_score"], reverse=True)
        candidates = candidates[:top_k]
        
        if candidates:
            self._add_message(f"🔄 Context-aware fallback found {len(candidates)} candidates")
        
        return candidates
    
    def _get_text_similarity_candidates(
        self,
        entity_name: str,
        kg_nodes: List[Dict],
        kg_embeddings: np.ndarray
    ) -> List[Dict]:
        """Get candidates based on text similarity using embeddings"""
        try:
            self._add_message(f"📊 Calculating text similarity for '{entity_name}'")
            
            # Get query embedding
            query_embedding = self._get_entity_embedding(entity_name)
            
            if query_embedding is None:
                return []
            
            # Check embedding dimensions
            if query_embedding.shape[0] != kg_embeddings.shape[1]:
                if query_embedding.shape[0] < kg_embeddings.shape[1]:
                    padding = np.zeros(kg_embeddings.shape[1] - query_embedding.shape[0], dtype=np.float32)
                    query_embedding = np.concatenate([query_embedding, padding])
                else:
                    query_embedding = query_embedding[:kg_embeddings.shape[1]]
            
            # Compute similarities
            query_embedding_2d = query_embedding.reshape(1, -1).astype(np.float32)
            kg_embeddings = kg_embeddings.astype(np.float32)
            
            similarities = cosine_similarity(query_embedding_2d, kg_embeddings)[0]
            
            # Dynamic threshold based on distribution
            max_sim = similarities.max()
            mean_sim = similarities.mean()
            std_sim = similarities.std()
            
            if max_sim < 0.4:
                threshold = max(mean_sim + std_sim, 0.15)
            else:
                threshold = 0.35  # Higher threshold for better quality
            
            self._add_message(f"✅ Computed similarities: max={max_sim:.4f}, mean={mean_sim:.4f}, threshold={threshold:.4f}")
            
            # Collect candidates above threshold
            all_candidates = []
            for idx, score in enumerate(similarities):
                if idx < len(kg_nodes) and score > threshold:
                    kg_node = kg_nodes[idx]
                    candidate = {
                        "kg_index": int(idx),
                        "node_id": kg_node.get("node_id", f"node_{idx}"),
                        "node_name": kg_node.get("node_name", "Unknown"),
                        "labels": kg_node.get("labels", []),
                        "entity_type": kg_node.get("entity_type", "unknown"),
                        "similarity_score": float(score),
                        "properties": kg_node.get("properties", {}),
                        "property_text": kg_node.get("property_text", ""),
                        "text_length": kg_node.get("text_length", 0)
                    }
                    all_candidates.append(candidate)
            
            # Sort
            all_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            if all_candidates:
                self._add_message(f"📊 Found {len(all_candidates)} text similarity candidates")
                return all_candidates[:100]  # Limit to top 100
            
            self._add_message("⚠️ No candidates above similarity threshold")
            return []
            
        except Exception as e:
            self._add_message(f"❌ Error in text similarity calculation: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_pairwise_similarity(
        self,
        entity1: Dict,
        entity2: Dict,
        entity_type: str = "Unknown"
    ) -> Dict[str, Any]:
        """Calculate comprehensive similarity between two entities"""
        similarity_metrics = {
            "text_similarity": 0.0,
            "property_similarity": 0.0,
            "name_similarity": 0.0,
            "overall_similarity": 0.0,
            "match_type": "none",
            "property_matches": [],
            "confidence": "low"
        }
        
        # Name similarity
        name1 = entity1.get("name", entity1.get("node_name", ""))
        name2 = entity2.get("name", entity2.get("node_name", ""))
        
        if name1 and name2:
            similarity_metrics["name_similarity"] = self._calculate_string_similarity(name1, name2)
            
            if similarity_metrics["name_similarity"] > 0.8:
                similarity_metrics["match_type"] = "exact_name_match"
            elif similarity_metrics["name_similarity"] > 0.6:
                similarity_metrics["match_type"] = "contains_name_match"
        
        # Property similarity
        props1 = entity1.get("properties", {})
        props2 = entity2.get("properties", {})
        
        if props1 and props2:
            common_props = set(props1.keys()) & set(props2.keys())
            property_matches = []
            total_similarity = 0.0
            
            for prop in common_props:
                val1 = str(props1[prop])
                val2 = str(props2[prop])
                
                prop_similarity = self._calculate_string_similarity(val1, val2)
                total_similarity += prop_similarity
                
                if prop_similarity > 0.8:
                    property_matches.append({
                        "property": prop,
                        "match_type": "exact",
                        "value1": props1[prop],
                        "value2": props2[prop],
                        "similarity": prop_similarity
                    })
                elif prop_similarity > 0.6:
                    property_matches.append({
                        "property": prop,
                        "match_type": "similar",
                        "value1": props1[prop],
                        "value2": props2[prop],
                        "similarity": prop_similarity
                    })
            
            similarity_metrics["property_matches"] = property_matches
            
            if common_props:
                similarity_metrics["property_similarity"] = total_similarity / len(common_props)
        
        # Overall similarity
        weights = {"name": 0.4, "property": 0.6}
        similarity_metrics["overall_similarity"] = (
            similarity_metrics["name_similarity"] * weights["name"] +
            similarity_metrics["property_similarity"] * weights["property"]
        )
        
        # Confidence
        score = similarity_metrics["overall_similarity"]
        if score >= 0.9:
            similarity_metrics["confidence"] = "very_high"
        elif score >= 0.7:
            similarity_metrics["confidence"] = "high"
        elif score >= 0.5:
            similarity_metrics["confidence"] = "medium"
        else:
            similarity_metrics["confidence"] = "low"
        
        return similarity_metrics
    
    def batch_similarity_calculation(
        self,
        entity_names: List[str],
        entity_types: List[str],
        kg_nodes: List[Dict],
        kg_embeddings: np.ndarray,
        top_k_per_entity: int = None
    ) -> Dict[str, Any]:
        """Calculate similarity for multiple entities in batch"""
        if top_k_per_entity is None:
            top_k_per_entity = self.config.PIPELINE.TOP_K
        
        batch_results = {
            "entities": [],
            "statistics": {
                "total_entities": len(entity_names),
                "entities_with_matches": 0,
                "total_candidates": 0,
                "average_best_score": 0.0
            }
        }
        
        all_best_scores = []
        
        for i, (entity_name, entity_type) in enumerate(zip(entity_names, entity_types)):
            self._add_message(f"Processing {i+1}/{len(entity_names)}: {entity_name}")
            
            candidates, _ = self.get_enhanced_candidates(
                entity_name=entity_name,
                entity_type=entity_type,
                kg_nodes=kg_nodes,
                kg_embeddings=kg_embeddings,
                top_k=top_k_per_entity
            )
            
            entity_result = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "candidates_found": len(candidates),
                "candidates": candidates
            }
            
            if candidates:
                batch_results["statistics"]["entities_with_matches"] += 1
                batch_results["statistics"]["total_candidates"] += len(candidates)
                best_score = candidates[0].get("enhanced_score", 0.0)
                all_best_scores.append(best_score)
                entity_result["best_match"] = candidates[0].get("node_name", "Unknown")
                entity_result["best_score"] = best_score
                entity_result["match_type"] = candidates[0].get("match_type", "none")
            else:
                entity_result["best_match"] = None
                entity_result["best_score"] = 0.0
                entity_result["match_type"] = "none"
            
            batch_results["entities"].append(entity_result)
        
        # Statistics
        if all_best_scores:
            batch_results["statistics"]["average_best_score"] = np.mean(all_best_scores)
        
        self._add_message(f"✅ Batch completed: {batch_results['statistics']['entities_with_matches']}/{batch_results['statistics']['total_entities']} entities matched")
        
        return batch_results
    
    def validate_similarity_calculation(self, candidate: Dict) -> Dict[str, Any]:
        """Validate a similarity calculation result"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "scores": {}
        }
        
        # Required fields
        required_fields = ["node_name", "enhanced_score", "match_type"]
        for field in required_fields:
            if field not in candidate:
                validation_results["is_valid"] = False
                validation_results["issues"].append(f"Missing required field: {field}")
        
        # Score range
        enhanced_score = candidate.get("enhanced_score", -1)
        if not (0 <= enhanced_score <= 1):
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Enhanced score out of range: {enhanced_score}")
        
        # Property info
        property_info = candidate.get("property_info", {})
        if not property_info:
            validation_results["warnings"].append("Missing property_info")
        elif "match_type" not in property_info:
            validation_results["warnings"].append("Property info missing match_type")
        
        # Store scores
        validation_results["scores"] = {
            "enhanced_score": enhanced_score,
            "similarity_score": candidate.get("similarity_score", 0.0),
            "confidence": property_info.get("confidence", "unknown")
        }
        
        return validation_results
    
    def clear_caches(self):
        """Clear all caches"""
        self.expansion_cache.clear()
        self.similarity_cache.clear()
        self._add_message("🧹 Cleared all caches")