"""
Pydantic schemas for clinical trial data extraction.

Defines structured output schemas for all extraction fields using Pydantic v2,
ensuring type safety and validation across the extraction pipeline.

Author: Clinical Trials AI Team
Version: 1.0.0
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class ExtractionFieldType(str, Enum):
    """Enumeration of all extraction field types."""
    STUDY_OVERVIEW = "study_overview"
    BRIEF_DESCRIPTION = "brief_description"
    PRIMARY_SECONDARY_OBJECTIVES = "primary_secondary_objectives"
    TREATMENT_ARMS_INTERVENTIONS = "treatment_arms_interventions"
    ELIGIBILITY_CRITERIA = "eligibility_criteria"
    ENROLLMENT_PARTICIPANT_FLOW = "enrollment_participant_flow"
    ADVERSE_EVENTS_PROFILE = "adverse_events_profile"
    STUDY_LOCATIONS = "study_locations"
    SPONSOR_INFORMATION = "sponsor_information"


class ExtractionResult(BaseModel):
    """
    Standard extraction result schema for a single field.
    
    Attributes:
        content: The extracted information as a single string, preserving all original formatting
        page_references: List of page numbers where this information appears
    """
    content: str = Field(
        ...,
        description="The extracted information as a single string, preserving all original formatting, bullet points, and details"
    )
    page_references: List[int] = Field(
        ...,  # Make required for OpenAI structured outputs
        description="List of page numbers where this information appears"
    )

    class Config:
        extra = "forbid"  # Disallow additional properties for OpenAI structured outputs
        json_schema_extra = {
            "examples": [
                {
                    "content": "Phase II, Randomized Controlled Trial. Study Title: A multicenter...",
                    "page_references": [1, 2, 5]
                }
            ],
            "additionalProperties": False
        }


class StudyOverviewExtraction(ExtractionResult):
    """
    Extraction schema for study_overview field.
    
    Contains:
    - Study title, NCT ID, protocol number
    - Phase (Phase I/II/III/IV)
    - Study type (randomized/non-randomized, controlled/uncontrolled, blinded/open-label)
    - Disease/condition being studied
    - Study duration
    """
    pass


class BriefDescriptionExtraction(ExtractionResult):
    """
    Extraction schema for brief_description field.
    
    Contains:
    - Concise 2-3 sentence summary of study purpose
    - Rationale for the study
    - Overall study design
    """
    pass


class PrimarySecondaryObjectivesExtraction(ExtractionResult):
    """
    Extraction schema for primary_secondary_objectives field.
    
    Contains:
    - PRIMARY objectives (clearly labeled) with full definitions and timeframes
    - SECONDARY objectives (clearly labeled) with their definitions and timeframes
    """
    pass


class TreatmentArmsInterventionsExtraction(ExtractionResult):
    """
    Extraction schema for treatment_arms_interventions field.
    
    Contains:
    - All treatment arms with arm names
    - Interventions/drugs for each arm
    - Exact dosing schedules
    - Routes of administration
    - Treatment duration
    - Comparator or combination therapy details
    """
    pass


class EligibilityCriteriaExtraction(ExtractionResult):
    """
    Extraction schema for eligibility_criteria field.
    
    Contains:
    - Complete inclusion criteria (required characteristics for enrollment)
    - Exclusion criteria (disqualifying factors)
    - All participant requirement specifications
    """
    pass


class EnrollmentParticipantFlowExtraction(ExtractionResult):
    """
    Extraction schema for enrollment_participant_flow field.
    
    Contains:
    - Target sample size
    - Actual enrollment numbers
    - Randomization methodology
    - Screening process details
    - Participant allocation
    - Flow through study phases
    """
    pass


class AdverseEventsProfileExtraction(ExtractionResult):
    """
    Extraction schema for adverse_events_profile field.
    
    Contains:
    - Reported adverse events with frequencies/percentages
    - Serious adverse events (SAEs)
    - Toxicity grades
    - Overall safety profile data
    """
    pass


class StudyLocationsExtraction(ExtractionResult):
    """
    Extraction schema for study_locations field.
    
    Contains:
    - All study sites and countries
    - Institutions involved
    - Names of principal investigators or site coordinators
    """
    pass


class SponsorInformationExtraction(ExtractionResult):
    """
    Extraction schema for sponsor_information field.
    
    Contains:
    - Primary sponsor organization
    - Collaborating institutions
    - Funding sources
    - Contact information
    """
    pass


class ClinicalTrialExtractionResult(BaseModel):
    """
    Complete extraction result for all fields in a clinical trial document.
    
    Contains all 9 extracted fields with their respective content and page references.
    """
    study_overview: ExtractionResult = Field(
        description="Study overview information"
    )
    brief_description: ExtractionResult = Field(
        description="Brief description of the study"
    )
    primary_secondary_objectives: ExtractionResult = Field(
        description="Primary and secondary objectives/endpoints"
    )
    treatment_arms_interventions: ExtractionResult = Field(
        description="Treatment arms and intervention details"
    )
    eligibility_criteria: ExtractionResult = Field(
        description="Inclusion and exclusion criteria"
    )
    enrollment_participant_flow: ExtractionResult = Field(
        description="Enrollment numbers and participant flow"
    )
    adverse_events_profile: ExtractionResult = Field(
        description="Adverse events and safety profile"
    )
    study_locations: ExtractionResult = Field(
        description="Study sites and locations"
    )
    sponsor_information: ExtractionResult = Field(
        description="Sponsor and funding information"
    )

    class Config:
        extra = "forbid"  # Disallow additional properties for OpenAI structured outputs
        json_schema_extra = {
            "examples": [
                {
                    "study_overview": {
                        "content": "Study details...",
                        "page_references": [1, 2]
                    },
                    "brief_description": {
                        "content": "Brief summary...",
                        "page_references": [2]
                    },
                    # ... other fields
                }
            ],
            "additionalProperties": False
        }


# Mapping of field names to their schema classes
FIELD_SCHEMA_MAP: Dict[str, type] = {
    ExtractionFieldType.STUDY_OVERVIEW.value: StudyOverviewExtraction,
    ExtractionFieldType.BRIEF_DESCRIPTION.value: BriefDescriptionExtraction,
    ExtractionFieldType.PRIMARY_SECONDARY_OBJECTIVES.value: PrimarySecondaryObjectivesExtraction,
    ExtractionFieldType.TREATMENT_ARMS_INTERVENTIONS.value: TreatmentArmsInterventionsExtraction,
    ExtractionFieldType.ELIGIBILITY_CRITERIA.value: EligibilityCriteriaExtraction,
    ExtractionFieldType.ENROLLMENT_PARTICIPANT_FLOW.value: EnrollmentParticipantFlowExtraction,
    ExtractionFieldType.ADVERSE_EVENTS_PROFILE.value: AdverseEventsProfileExtraction,
    ExtractionFieldType.STUDY_LOCATIONS.value: StudyLocationsExtraction,
    ExtractionFieldType.SPONSOR_INFORMATION.value: SponsorInformationExtraction,
}


def get_field_schema_dict(field_name: str) -> Dict[str, Any]:
    """
    Get the JSON schema dictionary for a specific extraction field.
    
    Args:
        field_name: Name of the extraction field
        
    Returns:
        Dictionary representation of the Pydantic schema suitable for OpenAI API
    """
    if field_name not in FIELD_SCHEMA_MAP:
        raise ValueError(f"Unknown field: {field_name}")
    
    schema_class = FIELD_SCHEMA_MAP[field_name]
    return schema_class.model_json_schema()


def get_extraction_result_schema_dict() -> Dict[str, Any]:
    """
    Get the JSON schema dictionary for the standard ExtractionResult.
    
    Returns:
        Dictionary representation of the ExtractionResult schema suitable for OpenAI API
    """
    schema = ExtractionResult.model_json_schema()
    # Ensure additionalProperties is set to false for OpenAI structured outputs
    schema["additionalProperties"] = False
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict) and "type" in prop_schema and prop_schema["type"] == "object":
                prop_schema["additionalProperties"] = False
    return schema


def get_all_fields_schema_dict() -> Dict[str, Any]:
    """
    Get the JSON schema dictionary for the complete ClinicalTrialExtractionResult.
    
    Returns:
        Dictionary representation of the full extraction schema suitable for OpenAI API
    """
    schema = ClinicalTrialExtractionResult.model_json_schema()
    # Ensure additionalProperties is set to false for OpenAI structured outputs
    schema["additionalProperties"] = False
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict) and "type" in prop_schema and prop_schema["type"] == "object":
                prop_schema["additionalProperties"] = False
    return schema


# Default field configurations with metadata
FIELD_CONFIGS_WITH_SCHEMA = {
    ExtractionFieldType.STUDY_OVERVIEW.value: {
        "schema_class": StudyOverviewExtraction,
        "keywords": ["protocol", "study design", "overview", "summary", "background", "rationale"],
        "max_tokens": 40000,
        "priority": 1,
        "description": "Study title, NCT ID, protocol number, phase, study type (randomized/non-randomized, controlled/uncontrolled, blinded/open-label), disease/condition, and study duration"
    },
    ExtractionFieldType.BRIEF_DESCRIPTION.value: {
        "schema_class": BriefDescriptionExtraction,
        "keywords": ["description", "purpose", "aims", "goals"],
        "max_tokens": 30000,
        "priority": 2,
        "description": "Concise 2-3 sentence summary of the study purpose, rationale, and overall design"
    },
    ExtractionFieldType.PRIMARY_SECONDARY_OBJECTIVES.value: {
        "schema_class": PrimarySecondaryObjectivesExtraction,
        "keywords": ["primary objective", "secondary objective", "primary endpoint", "secondary endpoint", "primary outcome", "secondary outcome", "aim"],
        "max_tokens": 35000,
        "priority": 1,
        "description": "PRIMARY objectives (clearly labeled) with full definitions and timeframes, followed by SECONDARY objectives (clearly labeled) with their definitions and timeframes"
    },
    ExtractionFieldType.TREATMENT_ARMS_INTERVENTIONS.value: {
        "schema_class": TreatmentArmsInterventionsExtraction,
        "keywords": ["treatment", "intervention", "arm", "group", "therapy", "dose", "regimen"],
        "max_tokens": 30000,  # Reduced from 40k to prevent timeouts on large chunks
        "priority": 1,
        "description": "All treatment arms with arm names, interventions/drugs for each arm, exact dosing schedules, routes of administration, treatment duration, and any comparator or combination therapy details"
    },
    ExtractionFieldType.ELIGIBILITY_CRITERIA.value: {
        "schema_class": EligibilityCriteriaExtraction,
        "keywords": ["eligibility", "inclusion", "exclusion", "criteria", "participant"],
        "max_tokens": 35000,
        "priority": 2,
        "description": "Complete inclusion criteria (required characteristics for enrollment) and exclusion criteria (disqualifying factors) for study participants"
    },
    ExtractionFieldType.ENROLLMENT_PARTICIPANT_FLOW.value: {
        "schema_class": EnrollmentParticipantFlowExtraction,
        "keywords": ["enrollment", "randomization", "participant flow", "screening", "allocation"],
        "max_tokens": 35000,
        "priority": 2,
        "description": "Target sample size, actual enrollment numbers, randomization methodology, screening process, participant allocation, and flow through study phases"
    },
    ExtractionFieldType.ADVERSE_EVENTS_PROFILE.value: {
        "schema_class": AdverseEventsProfileExtraction,
        "keywords": ["adverse event", "safety", "toxicity", "side effect", "AE", "SAE"],
        "max_tokens": 50000,
        "priority": 1,
        "description": "Reported adverse events with frequencies/percentages, serious adverse events (SAEs), toxicity grades, and overall safety profile data"
    },
    ExtractionFieldType.STUDY_LOCATIONS.value: {
        "schema_class": StudyLocationsExtraction,
        "keywords": ["site", "location", "center", "institution", "investigator"],
        "max_tokens": 25000,
        "priority": 3,
        "description": "All study sites, countries, institutions, and names of principal investigators or site coordinators"
    },
    ExtractionFieldType.SPONSOR_INFORMATION.value: {
        "schema_class": SponsorInformationExtraction,
        "keywords": ["sponsor", "funding", "organization", "contact", "investigator"],
        "max_tokens": 20000,
        "priority": 3,
        "description": "Primary sponsor organization, collaborating institutions, funding sources, and contact information"
    }
}
