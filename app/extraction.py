# extraction.py
import re
import json
from typing import List, Dict, Optional, Any
import logging
from config import Config

class EntityExtractor:
    """Handles biomedical entity extraction from text with focus on fertility-related entities"""
    
    def __init__(self, llm_client=None):
        """
        Initialize the EntityExtractor
        
        Args:
            llm_client: Optional LLM client for advanced extraction
        """
        self.llm_client = llm_client
        self.llm_model = Config.PIPELINE.LLM_MODEL
        self.entity_config = Config.ENTITY
        self.messages = []  # Store messages for Streamlit
        self._initialize_logger()
    
    def _initialize_logger(self):
        """Initialize logger"""
        self.logger = logging.getLogger(__name__)
    
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
        """Set LLM client for advanced extraction"""
        self.llm_client = client
        self._add_message(f"✅ LLM client configured for entity extraction")
    
    @staticmethod
    def safe_json_parse(text: str) -> Dict:
        """
        Safely parse JSON from LLM response.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON as dictionary
        """
        if not text:
            return {}
        
        # Remove code blocks
        text = re.sub(r"```json\s*|```\s*", "", text).strip()
        
        # Try to find and fix common JSON issues
        # Fix missing quotes around keys
        text = re.sub(r'(\w+)\s*:', r'"\1":', text)
        
        # Fix trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            return parsed
        except json.JSONDecodeError as e:
            # Try to find JSON object/array
            json_pattern = r'(\{.*\}|\[.*\])'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                try:
                    return json.loads(matches[0])
                except:
                    # Last attempt: try eval for simple dicts (be careful!)
                    try:
                        import ast
                        parsed = ast.literal_eval(matches[0])
                        if isinstance(parsed, (dict, list)):
                            return parsed
                    except:
                        return {}
        
        return {}
    
    def extract_entities_from_text(self, text: str) -> List[Dict]:
        """
        Extract fertility-related biomedical entities from text.
        Uses LLM if available, otherwise falls back to pattern matching.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities with metadata
        """
        self.clear_messages()
        
        if not text or not text.strip():
            self._add_message("❌ No text provided for entity extraction")
            return []
        
        self._add_message(f"🔍 Extracting entities from text (length: {len(text)} chars)")
        
        # Try LLM extraction first if client is available
        if self.llm_client:
            try:
                entities = self._extract_with_llm(text)
                if entities:
                    self._add_message(f"✅ Extracted {len(entities)} entities using LLM")
                    return entities
                else:
                    self._add_message("⚠️ LLM extraction returned no entities, using fallback")
            except Exception as e:
                self._add_message(f"⚠️ LLM extraction failed: {e}, using fallback")
        
        # Fallback to pattern-based extraction
        entities = self._fertility_fallback_extraction(text)
        self._add_message(f"✅ Extracted {len(entities)} entities using fallback patterns")
        
        return entities
    
    def _extract_with_llm(self, text: str) -> List[Dict]:
        """
        Extract fertility-related biomedical entities from text using LLM.
        Focuses on fertility diseases and their corresponding drugs/treatments.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not self.llm_client:
            return []
        
        # Get fertility-specific examples
        fertility_diseases = self.entity_config.EXAMPLES.get("disease", [])
        fertility_drugs = self.entity_config.EXAMPLES.get("drug", [])
        
        extraction_prompt = f"""
        TASK: Extract ONLY fertility-related biomedical entities from the text below.
        Focus on reproductive health diseases and their corresponding drugs/treatments.

        TEXT: "{text}"

        FERTILITY ENTITY TYPES:
        1. DISEASE: Fertility/reproductive health conditions ONLY
           Examples: {', '.join(fertility_diseases[:5])}
           Types: Infertility conditions, reproductive disorders, hormonal imbalances affecting fertility
        
        2. DRUG: Fertility medications and treatments ONLY  
           Examples: {', '.join(fertility_drugs[:5])}
           Types: Fertility drugs, hormone therapies, assisted reproduction medications

        STRICT RULES:
        1. Extract ONLY fertility-related entities
        2. Skip general diseases/drugs not related to reproduction
        3. Extract EXACT text spans as they appear
        4. Include confidence based on fertility relevance (0.0 to 1.0)
        5. Return valid JSON only

        EXAMPLES OF FERTILITY ENTITIES (EXTRACT):
        - "Polycystic ovary syndrome (PCOS)" → disease
        - "Clomiphene citrate" → drug  
        - "Endometriosis" → disease
        - "In vitro fertilization (IVF)" → drug/treatment
        - "Premature ovarian insufficiency" → disease
        - "Gonadotropins" → drug

        EXAMPLES OF NON-FERTILITY (IGNORE):
        - "Hypertension" → IGNORE (unless in fertility context)
        - "Aspirin" → IGNORE (unless prescribed for fertility)
        - "Diabetes" → IGNORE (unless causing infertility)
        - "Common cold" → IGNORE

        CONTEXT-BASED EXTRACTION:
        If text mentions "infertility", "fertility treatment", "reproduction", "pregnancy" etc.,
        then extract related diseases/drugs even if not explicitly fertility-related.

        EXAMPLES:
        Input: "Patient with Polycystic ovary syndrome was treated with Clomiphene citrate"
        Output: {{"Entity": [
          {{"id": "1", "type": "disease", "name": "Polycystic ovary syndrome", "confidence": 0.95}},
          {{"id": "2", "type": "drug", "name": "Clomiphene citrate", "confidence": 0.9}}
        ]}}

        Input: "Endometriosis causing infertility treated with GnRH agonists"
        Output: {{"Entity": [
          {{"id": "1", "type": "disease", "name": "Endometriosis", "confidence": 0.9}},
          {{"id": "2", "type": "drug", "name": "GnRH agonists", "confidence": 0.85}}
        ]}}

        Input: "Diethylstilbestrol exposure linked to fertility issues"
        Output: {{"Entity": [
          {{"id": "1", "type": "drug", "name": "Diethylstilbestrol", "confidence": 0.95}},
          {{"id": "2", "type": "disease", "name": "fertility issues", "confidence": 0.8}}
        ]}}

        OUTPUT FORMAT:
        {{
          "Entity": [
            {{"id": "1", "type": "disease" OR "drug", "name": "exact_name_from_text", "confidence": 0.95}}
          ]
        }}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a fertility medicine specialist. Extract ONLY fertility-related diseases and drugs. Ignore unrelated medical entities."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            parsed = self.safe_json_parse(content)
            
            # Handle both "Entity" and "entities" keys
            entities = []
            if "Entity" in parsed and isinstance(parsed["Entity"], list):
                entities = parsed["Entity"]
            elif "entities" in parsed and isinstance(parsed["entities"], list):
                entities = parsed["entities"]
            else:
                self._add_message("⚠️ LLM extraction failed to return proper JSON, using fertility-specific fallback")
                return self._fertility_fallback_extraction(text)
            
            # Filter to only fertility-related entities
            fertility_entities = self._filter_fertility_entities(entities, text)
            
            # If no fertility entities found but text has fertility context, use fallback
            if not fertility_entities and self._has_fertility_context(text):
                return self._fertility_fallback_extraction(text)
            
            return fertility_entities
                
        except Exception as e:
            self._add_message(f"❌ Error in LLM entity extraction: {e}")
            return self._fertility_fallback_extraction(text)
    
    def _filter_fertility_entities(self, entities: List[Dict], original_text: str) -> List[Dict]:
        """Filter entities to keep only fertility-related ones"""
        fertility_entities = []
        fertility_diseases = self.entity_config.EXAMPLES.get("disease", [])
        fertility_drugs = self.entity_config.EXAMPLES.get("drug", [])
        
        fertility_keywords = [
            "fertility", "infertility", "reproductive", "ovarian", "uterine", 
            "endometri", "pcos", "sperm", "oocyte", "embryo", "ivf", 
            "pregnancy", "conception", "menstrual", "hormon", "gonad"
        ]
        
        for entity in entities:
            if not isinstance(entity, dict) or "name" not in entity or "type" not in entity:
                continue
                
            entity_name = entity["name"].lower()
            entity_type = entity["type"].lower()
            
            # Check if entity is fertility-related
            is_fertility_related = False
            
            # Check entity name for fertility keywords
            if any(keyword in entity_name for keyword in fertility_keywords):
                is_fertility_related = True
            # Check if it's in our fertility examples
            elif entity["name"] in fertility_diseases or entity["name"] in fertility_drugs:
                is_fertility_related = True
            # Check if it's a known fertility condition/drug pattern
            elif entity_type == "disease" and any(term in entity_name for term in 
                  ["infertility", "endometriosis", "pcos", "fibroid", "amenorrhea"]):
                is_fertility_related = True
            elif entity_type == "drug" and any(term in entity_name for term in 
                  ["clomiphene", "letrozole", "gonadotropin", "fsh", "lh", "hcg"]):
                is_fertility_related = True
            
            if is_fertility_related:
                # Ensure proper entity type
                if entity_type not in ["disease", "drug"]:
                    # Map to correct type based on content
                    if any(term in entity_name for term in ["syndrome", "disease", "disorder", "infertility"]):
                        entity["type"] = "disease"
                    else:
                        entity["type"] = "drug"
                
                # Add confidence if missing
                if "confidence" not in entity:
                    # Adjust confidence based on fertility relevance
                    if any(keyword in entity_name for keyword in fertility_keywords):
                        entity["confidence"] = 0.85
                    else:
                        entity["confidence"] = 0.7
                
                # Add id if missing
                if "id" not in entity:
                    entity["id"] = str(len(fertility_entities) + 1)
                
                # Add exact text span if possible
                if "span_start" not in entity or "span_end" not in entity:
                    # Try to find the exact span in original text
                    import re
                    pattern = re.escape(entity["name"])
                    match = re.search(pattern, original_text, re.IGNORECASE)
                    if match:
                        entity["span_start"] = match.start()
                        entity["span_end"] = match.end()
                
                fertility_entities.append(entity)
        
        return fertility_entities
    
    def _has_fertility_context(self, text: str) -> bool:
        """Check if text has fertility-related context"""
        text_lower = text.lower()
        fertility_keywords = [
            "fertility", "infertility", "reproductive", "ovarian", "uterine", 
            "pregnancy", "conception", "menstrual", "hormon", "gonad",
            "sperm", "egg", "embryo", "ivf", "pcos", "endometri"
        ]
        return any(keyword in text_lower for keyword in fertility_keywords)
    
    def _fertility_fallback_extraction(self, text: str) -> List[Dict]:
        """
        Fallback method specifically for fertility-related entities.
        Uses pattern matching and keyword extraction.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Fertility-specific patterns
        fertility_patterns = {
            "disease": [
                # Infertility conditions
                r'\b(?:infertility|subfertility)\b',
                r'\b(?:male|female) factor (?:infertility|subfertility)\b',
                r'\b(?:unexplained|idiopathic) infertility\b',
                
                # Ovarian disorders
                r'\b(?:Polycystic ovary syndrome|PCOS|polycystic ovarian syndrome)\b',
                r'\b(?:Premature ovarian (?:insufficiency|failure)|POI|POF)\b',
                r'\b(?:Diminished|Decreased) ovarian reserve\b',
                r'\bPoor ovarian response\b',
                r'\bAnovulation\b',
                r'\bLuteal phase defect\b',
                
                # Uterine disorders
                r'\bEndometriosis\b',
                r'\b(?:Uterine|Uterus) fibroids?\b',
                r'\bAsherman\'s syndrome\b',
                r'\b(?:Uterine|Endometrial) adhesions\b',
                r'\b(?:Septate|Bicornuate) uterus\b',
                
                # Tubal factors
                r'\b(?:Tubal|Fallopian tube) (?:blockage|occlusion|damage)\b',
                r'\b(?:Hydrosalpinx|Salpingitis)\b',
                
                # Male infertility
                r'\b(?:Oligospermia|Azoospermia|Asthenospermia|Teratospermia)\b',
                r'\b(?:Low sperm count|Poor sperm motility|Abnormal sperm morphology)\b',
                r'\bVaricocele\b',
                r'\bObstructive azoospermia\b',
                
                # Miscarriage/recurrent loss
                r'\bRecurrent (?:pregnancy loss|miscarriage|abortion)\b',
                r'\b(?:Habitual|Recurrent) aborter\b',
                
                # Endocrine/hormonal
                r'\bHypothalamic amenorrhea\b',
                r'\b(?:Hyperprolactinemia|Elevated prolactin)\b',
                r'\b(?:Thyroid|Adrenal) disorders affecting fertility\b',
                
                # Specific examples from config
                r'\bfolliculotropic mycosis fungoides\b',
                r'\blocalized pagetoid reticulosis\b',
                r'\bclassic Hodgkin lymphoma, lymphocyte-rich type\b',
                r'\bHodgkin\'s paragranuloma\b',
                r'\bhairy cell leukemia variant\b',
                r'\b(?:lymphosarcoma|erythema multiforme)\b'
            ],
            "drug": [
                # Ovulation induction
                r'\b(?:Clomiphene citrate|Clomid|Serophene)\b',
                r'\bLetrozole\b',
                r'\b(?:Tamoxifen|Nolvadex)\b',
                
                # Gonadotropins
                r'\b(?:Gonadotropins|Gonadotropin therapy)\b',
                r'\b(?:FSH|follicle-stimulating hormone)\b',
                r'\b(?:LH|luteinizing hormone)\b',
                r'\b(?:hMG|human menopausal gonadotropin)\b',
                r'\b(?:hCG|human chorionic gonadotropin)\b',
                
                # GnRH analogs
                r'\b(?:GnRH agonists|Gonadotropin-releasing hormone agonists)\b',
                r'\b(?:Leuprolide|Lupron)\b',
                r'\b(?:Goserelin|Zoladex)\b',
                r'\b(?:GnRH antagonists|Gonadotropin-releasing hormone antagonists)\b',
                r'\b(?:Ganirelix|Antagon|Cetrorelix|Cetrotide)\b',
                
                # Insulin sensitizers
                r'\bMetformin\b',
                
                # Hormone supplements
                r'\bProgesterone\b',
                r'\b(?:Micronized progesterone|Prometrium|Endometrin|Crinone)\b',
                r'\bEstradiol\b',
                r'\b(?:Estrogen|Estrogen therapy)\b',
                
                # Specific drug examples from config (focusing on hormonal/fertility relevance)
                r'\bDiethylstilbestrol\b',
                r'\b(?:Liothyronine|Levothyroxine)\b',  # Thyroid hormones can affect fertility
                r'\b(?:Hydrocortisone|Prednisolone|Betamethasone)\b',  # Corticosteroids for immune issues
                r'\bHydrocortisone (?:cypionate|phosphate|probutate|valerate)\b',
                r'\bPrednisolone (?:phosphate|acetate)\b',
                r'\bBetamethasone phosphate\b',
                
                # Assisted reproduction
                r'\b(?:IVF|in vitro fertilization)\b',
                r'\b(?:ICSI|intracytoplasmic sperm injection)\b',
                r'\b(?:IUI|intrauterine insemination)\b',
                
                # Other fertility treatments
                r'\b(?:Bromocriptine|Parlodel)\b',
                r'\b(?:Cabergoline|Dostinex)\b',
                r'\bDanazol\b',
                r'\b(?:Aspirin|Heparin) for fertility\b'
            ]
        }
        
        # Check if text has fertility context
        if not self._has_fertility_context(text):
            return entities
        
        text_lower = text.lower()
        
        # Extract using patterns
        for entity_type, patterns in fertility_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        # Get the exact text (preserve original case)
                        start_pos = match.start()
                        end_pos = match.end()
                        entity_name = text[start_pos:end_pos]
                        
                        # Check if we already have this entity (case-insensitive)
                        if not any(e["name"].lower() == entity_name.lower() for e in entities):
                            # Calculate confidence
                            confidence = 0.9  # High confidence for exact pattern matches
                            
                            # Adjust confidence based on specificity
                            if entity_type == "disease":
                                if any(term in entity_name.lower() for term in ["infertility", "pcos", "endometriosis"]):
                                    confidence = 0.95
                            elif entity_type == "drug":
                                if any(term in entity_name.lower() for term in ["clomiphene", "letrozole", "gonadotropin"]):
                                    confidence = 0.92
                            
                            entities.append({
                                "id": str(len(entities) + 1),
                                "type": entity_type,
                                "name": entity_name,
                                "confidence": confidence,
                                "span_start": start_pos,
                                "span_end": end_pos,
                                "source": "pattern_match"
                            })
                except re.error as e:
                    self._add_message(f"⚠️ Regex pattern error: {e}")
                    continue
        
        # If no pattern matches but fertility context exists, try keyword-based extraction
        if not entities and self._has_fertility_context(text):
            entities = self._keyword_based_extraction(text)
        
        # Deduplicate
        unique_entities = []
        seen_names = set()
        for entity in entities:
            name_lower = entity["name"].lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _keyword_based_extraction(self, text: str) -> List[Dict]:
        """Extract entities based on keywords and context"""
        entities = []
        
        # Look for capitalized medical terms in fertility context
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for i, word in enumerate(words):
            # Check if word looks like a medical term
            if len(word) > 5 and word not in ["Patient", "Treatment", "Therapy", "Doctor", "Hospital", "Clinical"]:
                # Check context around the word
                context_start = max(0, i - 2)
                context_end = min(len(words), i + 3)
                context = " ".join(words[context_start:context_end]).lower()
                
                # Check for fertility context
                if any(fert_term in context for fert_term in ["infertility", "fertility", "treatment", "therapy", "medication", "prescribed"]):
                    # Try to determine if it's a disease or drug
                    entity_type = "drug" if any(drug_term in context for drug_term in ["prescribed", "medication", "drug", "therapy", "dose"]) else "disease"
                    
                    # Find the exact position in text
                    pattern = re.escape(word)
                    match = re.search(pattern, text)
                    if match:
                        entities.append({
                            "id": str(len(entities) + 1),
                            "type": entity_type,
                            "name": word,
                            "confidence": 0.6,
                            "span_start": match.start(),
                            "span_end": match.end(),
                            "source": "context_inference"
                        })
        
        return entities
    
    def extract_with_details(self, text: str) -> Dict[str, Any]:
        """
        Extract entities with additional details and statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extraction results and metadata
        """
        entities = self.extract_entities_from_text(text)
        
        # Calculate statistics
        disease_count = sum(1 for e in entities if e.get("type") == "disease")
        drug_count = sum(1 for e in entities if e.get("type") == "drug")
        
        # Calculate average confidence
        if entities:
            avg_confidence = sum(e.get("confidence", 0) for e in entities) / len(entities)
        else:
            avg_confidence = 0
        
        return {
            "entities": entities,
            "statistics": {
                "total_entities": len(entities),
                "disease_count": disease_count,
                "drug_count": drug_count,
                "average_confidence": avg_confidence
            },
            "text_length": len(text),
            "has_fertility_context": self._has_fertility_context(text),
            "extraction_method": "LLM" if self.llm_client and entities and any(e.get("source") != "pattern_match" for e in entities) else "Pattern"
        }
    
    def batch_extract(self, texts: List[str]) -> List[List[Dict]]:
        """
        Extract entities from multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of entity lists for each text
        """
        results = []
        for i, text in enumerate(texts):
            self._add_message(f"Extracting entities from text {i+1}/{len(texts)}")
            entities = self.extract_entities_from_text(text)
            results.append(entities)
        return results
    
    def validate_extraction(self, entities: List[Dict]) -> Dict[str, Any]:
        """
        Validate extracted entities
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Validation results
        """
        validation_results = {
            "total_entities": len(entities),
            "valid_entities": 0,
            "invalid_entities": 0,
            "issues": []
        }
        
        for i, entity in enumerate(entities):
            # Check required fields
            if not isinstance(entity, dict):
                validation_results["issues"].append(f"Entity {i} is not a dictionary")
                validation_results["invalid_entities"] += 1
                continue
            
            required_fields = ["id", "type", "name"]
            missing_fields = [field for field in required_fields if field not in entity]
            
            if missing_fields:
                validation_results["issues"].append(f"Entity {i} missing fields: {missing_fields}")
                validation_results["invalid_entities"] += 1
                continue
            
            # Check entity type
            if entity["type"] not in ["disease", "drug"]:
                validation_results["issues"].append(f"Entity {i} has invalid type: {entity['type']}")
                validation_results["invalid_entities"] += 1
                continue
            
            # Check confidence range
            confidence = entity.get("confidence", 0)
            if not (0 <= confidence <= 1):
                validation_results["issues"].append(f"Entity {i} has invalid confidence: {confidence}")
                validation_results["invalid_entities"] += 1
                continue
            
            validation_results["valid_entities"] += 1
        
        return validation_results