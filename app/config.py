# config.py
class Neo4jConfig:
    """Neo4j database configuration"""
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "hedi123456"

class PipelineConfig:
    """Pipeline configuration parameters"""
    MAX_NODES = 1500
    TOP_K = 5
    SIM_THRESHOLD = 0.6
    EXACT_MATCH_THRESHOLD = 0.9
    LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_PATH_LENGTH = 4
    MAX_PATHS_PER_PAIR = 5
    TOP_PATHS_FOR_PRUNING = 3

class EntityConfig:
    """Entity type and schema configuration"""
    # Medical Entity types
    TYPES = ["disease", "drug"]
    
    # Entity Property Schemas
    PROPERTY_SCHEMAS = {
        "disease": {
            "properties": [
                "SNOMEDCT_US_definition",
                "mayo_causes",
                "mayo_complications",
                "mayo_prevention",
                "mayo_risk_factors",
                "mayo_see_doc",
                "mayo_symptoms",
                "mondo_definitions",
                "node_id",
                "node_index",
                "node_name",
                "node_source",
                "orphanet_clinical_description",
                "orphanet_definition",
                "orphanet_epidemiology",
                "orphanet_management_and_treatment",
                "orphanet_prevalence",
                "umls_descriptions"
            ],
            "primary_keys": ["node_id", "node_name"],
            "display_properties": [
                "node_name",
                "mondo_definitions",
                "mayo_symptoms",
                "orphanet_prevalence"
            ]
        },
        "drug": {
            "properties": [
                "atc_4",
                "category",
                "clogp",
                "description",
                "group",
                "half_life",
                "indication",
                "mechanism_of_action",
                "molecular_weight",
                "node_id",
                "node_index",
                "node_name",
                "node_source",
                "pathway",
                "pharmacodynamics",
                "protein_binding",
                "state",
                "tpsa"
            ],
            "primary_keys": ["node_id", "node_name"],
            "display_properties": [
                "node_name",
                "description",
                "indication",
                "mechanism_of_action",
                "category"
            ]
        }
    }
    
    # Property weightings for similarity
    PROPERTY_WEIGHTS = {
        "exact_primary_key": 10.0,
        "exact_display_property": 5.0,
        "contains_primary_key": 3.0,
        "contains_display_property": 2.0,
        "other_property_match": 1.0,
        "name_exact": 8.0,
        "name_contains": 4.0
    }
    
    # Entity examples for better prompting
    EXAMPLES = {
        "disease": [
            "folliculotropic mycosis fungoides",
            "localized pagetoid reticulosis", 
            "erythema multiforme",
            "classic Hodgkin lymphoma, lymphocyte-rich type",
            "Hodgkin's paragranuloma",
            "lymphosarcoma",
            "hairy cell leukemia variant"
        ],
        "drug": [
            "Diethylstilbestrol",
            "Liothyronine",
            "Levothyroxine",
            "Hydrocortisone cypionate",
            "Hydrocortisone phosphate",
            "Hydrocortisone probutate", 
            "Hydrocortisone valerate",
            "Prednisolone phosphate",
            "Prednisolone acetate",
            "Betamethasone phosphate"
        ]
    }
    
    @classmethod
    def get_schema(cls, entity_type: str):
        """Get property schema for a specific entity type"""
        return cls.PROPERTY_SCHEMAS.get(entity_type, {})
    
    @classmethod
    def get_display_properties(cls, entity_type: str):
        """Get display properties for a specific entity type"""
        schema = cls.get_schema(entity_type)
        return schema.get("display_properties", [])
    
    @classmethod
    def get_primary_keys(cls, entity_type: str):
        """Get primary keys for a specific entity type"""
        schema = cls.get_schema(entity_type)
        return schema.get("primary_keys", [])
    
    @classmethod
    def get_all_properties(cls, entity_type: str):
        """Get all properties for a specific entity type"""
        schema = cls.get_schema(entity_type)
        return schema.get("properties", [])
    
    @classmethod
    def get_examples(cls, entity_type: str):
        """Get examples for a specific entity type"""
        return cls.EXAMPLES.get(entity_type, [])
    
    @classmethod
    def get_weight(cls, weight_type: str):
        """Get weight value for a specific weight type"""
        return cls.PROPERTY_WEIGHTS.get(weight_type, 1.0)

class Config:
    """Main configuration class that aggregates all configs"""
    NEO4J = Neo4jConfig
    PIPELINE = PipelineConfig
    ENTITY = EntityConfig
    
    @classmethod
    def validate(cls):
        """Validate all configuration parameters"""
        validation_errors = []
        
        # Validate Neo4j config
        if not cls.NEO4J.URI:
            validation_errors.append("Neo4j URI is not set")
        if not cls.NEO4J.USER:
            validation_errors.append("Neo4j USER is not set")
        if not cls.NEO4J.PASSWORD:
            validation_errors.append("Neo4j PASSWORD is not set")
        
        # Validate Pipeline config
        if cls.PIPELINE.MAX_NODES <= 0:
            validation_errors.append("MAX_NODES must be positive")
        if cls.PIPELINE.TOP_K <= 0:
            validation_errors.append("TOP_K must be positive")
        if not 0 <= cls.PIPELINE.SIM_THRESHOLD <= 1:
            validation_errors.append("SIM_THRESHOLD must be between 0 and 1")
        if not 0 <= cls.PIPELINE.EXACT_MATCH_THRESHOLD <= 1:
            validation_errors.append("EXACT_MATCH_THRESHOLD must be between 0 and 1")
        
        # Validate Entity config
        if not cls.ENTITY.TYPES:
            validation_errors.append("ENTITY_TYPES cannot be empty")
        
        for entity_type in cls.ENTITY.TYPES:
            if entity_type not in cls.ENTITY.PROPERTY_SCHEMAS:
                validation_errors.append(f"No property schema defined for entity type: {entity_type}")
        
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")
        
        return True