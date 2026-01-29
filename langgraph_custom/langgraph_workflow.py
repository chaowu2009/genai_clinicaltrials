"""
Simple LangGraph Workflow for Clinical Trial Analysis
Following official LangGraph documentation patterns
Integrated with app_v1.py functionality
"""

import os
os.environ['PYDANTIC_V2'] = 'true'

from typing import TypedDict, Annotated, Literal, Iterator, Optional, Callable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk
import json
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
import os
import time
import logging

# Import parsers - use EnhancedClinicalTrialParser for better accuracy
from langgraph_custom.enhanced_parser import EnhancedClinicalTrialParser, ClinicalTrialData

# Import multi-turn extractor for handling large PDFs without rate limiting
try:
    from .multi_turn_extractor import MultiTurnExtractor, estimate_extraction_cost
    MULTI_TURN_AVAILABLE = True
except ImportError:
    MULTI_TURN_AVAILABLE = False
    print("WARNING: multi_turn_extractor not available - falling back to single-call extraction")

# Import RAG tool for clinical trials search
try:
    from .rag_tool import create_clinical_trials_rag_tool, ClinicalTrialSearchTool
    RAG_TOOL_AVAILABLE = True
    print("RAG tool loaded successfully")
except ImportError as e:
    RAG_TOOL_AVAILABLE = False
    print(f"WARNING: RAG tool not available - clinical trials similarity search disabled: {e}")
except Exception as e:
    RAG_TOOL_AVAILABLE = False
    print(f"ERROR: RAG tool failed to load: {e}")

load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize LLM with streaming support
# Using gpt-4o-mini for higher rate limits (200k TPM vs 30k TPM for gpt-4o)
# gpt-4o-mini is 15x cheaper and has 128k context window (same as gpt-4o)
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    streaming=True,
    request_timeout=300,  # 5 minutes for large chunks
    timeout=300  # Connection timeout
)

# Create a non-streaming LLM instance for re-extraction
llm_non_streaming = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1, 
    streaming=False,
    request_timeout=300,  # 5 minutes for large chunks
    timeout=300  # Connection timeout
)

# System message for GPT-4 summarization (from app_v1)
SYSTEM_MESSAGE = "You are a clinical research summarization expert. Create concise, well-formatted summaries that focus only on available information. Avoid filler text and sections with insufficient data. Use clear markdown formatting and keep summaries under 400 words while including all key available information."

class WorkflowState(TypedDict):
    """State for the workflow following LangGraph patterns"""
    input_url: str
    input_type: Literal["pdf", "url", "unknown"]
    raw_data: dict  # Raw API response or PDF data
    parsed_json: dict  # Structured 9-field schema (final, after LLM if used)
    parser_only_json: dict  # Parser output before LLM enhancement
    data_to_summarize: dict  # Formatted for GPT (like app_v1)
    confidence_score: float
    completeness_score: float
    missing_fields: list
    nct_id: str
    chat_query: str
    chat_response: str
    stream_response: Iterator  # For streaming
    error: str
    used_llm_fallback: bool  # Track if LLM fallback was used
    # Multi-turn extraction progress tracking
    extraction_progress: dict  # {"current_field": str, "completed": int, "total": int}
    extraction_cost_estimate: dict  # Cost and time estimates
    progress_log: list  # Real-time progress messages for UI display
    # RAG tool integration
    use_rag_tool: bool  # Whether to use RAG tool for this query
    rag_tool_results: str  # Results from RAG tool search
    # RAG tool integration
    use_rag_tool: bool  # Whether to use RAG tool for this query
    rag_tool_results: str  # Results from RAG tool search


def classify_input(state: WorkflowState) -> WorkflowState:
    """Node: Classify input as PDF, URL, or RAG-only query"""
    input_url = state["input_url"]
    chat_query = state.get("chat_query", "")
    
    # If input_type is already set (e.g., by UI), respect it
    if state.get("input_type") == "rag_only":
        # Ensure empty data structures for RAG-only queries
        state["data_to_summarize"] = {}
        state["parsed_json"] = {}
        state["raw_data"] = {}
        return state
    
    # Check if this is a RAG-only query (no URL provided but has chat query)
    if (not input_url or input_url.strip() == "") and chat_query and should_use_rag_tool(chat_query):
        state["input_type"] = "rag_only"
        # Set empty data structures for RAG-only queries
        state["data_to_summarize"] = {}
        state["parsed_json"] = {}
        state["raw_data"] = {}
        return state
    
    # Handle cases with existing parsed data (follow-up questions)
    if (not input_url or input_url.strip() == "") and state.get("parsed_json") and chat_query:
        state["input_type"] = "followup"
        return state
    
    # Original URL/PDF classification logic
    if not input_url or input_url.strip() == "":
        state["input_type"] = "unknown"
        state["error"] = "Please provide a ClinicalTrials.gov URL, PDF file, or ask a question about clinical trials."
        return state
    
    if input_url.lower().endswith('.pdf') or 'pdf' in input_url.lower():
        state["input_type"] = "pdf"
    elif 'clinicaltrials.gov' in input_url.lower() or re.search(r"NCT\d{8}", input_url):
        state["input_type"] = "url"
    else:
        state["input_type"] = "unknown"
        state["error"] = "Invalid input type. Please provide a ClinicalTrials.gov URL or PDF file."
    
    return state


def route_input(state: WorkflowState) -> Literal["pdf_parser", "url_extractor", "route_decision", "error"]:
    """Conditional edge: Route based on input type"""
    if state["input_type"] == "pdf":
        return "pdf_parser"
    elif state["input_type"] == "url":
        return "url_extractor"
    elif state["input_type"] == "rag_only" or state["input_type"] == "followup":
        return "route_decision"  # Skip study processing, go directly to chat/RAG routing
    else:
        return "error"


def parse_pdf(state: WorkflowState) -> WorkflowState:
    """Node: Parse PDF document using EnhancedClinicalTrialParser for better accuracy"""
    try:
        # Use enhanced parser with advanced features
        parser = EnhancedClinicalTrialParser(use_ocr=False, use_nlp=False)
        
        file_path = state["input_url"]
        
        # Handle file path or URL
        if file_path.startswith('http'):
            # Download PDF temporarily
            response = requests.get(file_path)
            temp_path = f"temp_download_{Path(file_path).stem}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            file_path = temp_path
        
        # Parse PDF with enhanced parser
        clinical_data, tables = parser.parse_pdf(file_path, extract_tables=True)
        
        # Convert ClinicalTrialData to dict
        from dataclasses import asdict
        parsed_schema = {
            "study_overview": clinical_data.study_overview,
            "brief_description": clinical_data.brief_description,
            "primary_secondary_objectives": clinical_data.primary_secondary_objectives,
            "treatment_arms_interventions": clinical_data.treatment_arms_interventions,
            "eligibility_criteria": clinical_data.eligibility_criteria,
            "enrollment_participant_flow": clinical_data.enrollment_participant_flow,
            "adverse_events_profile": clinical_data.adverse_events_profile,
            "study_locations": clinical_data.study_locations,
            "sponsor_information": clinical_data.sponsor_information
        }
        
        state["parsed_json"] = parsed_schema
        state["raw_data"] = {"clinical_data": asdict(clinical_data), "tables": [asdict(t) for t in tables]}
        
        # Convert to display format for summarization
        state["data_to_summarize"] = {
            "Study Overview": parsed_schema.get("study_overview", ""),
            "Brief Description": parsed_schema.get("brief_description", ""),
            "Primary and Secondary Objectives": parsed_schema.get("primary_secondary_objectives", ""),
            "Treatment Arms and Interventions": parsed_schema.get("treatment_arms_interventions", ""),
            "Eligibility Criteria": parsed_schema.get("eligibility_criteria", ""),
            "Enrollment and Participant Flow": parsed_schema.get("enrollment_participant_flow", ""),
            "Adverse Events Profile": parsed_schema.get("adverse_events_profile", ""),
            "Study Locations": parsed_schema.get("study_locations", ""),
            "Sponsor Information": parsed_schema.get("sponsor_information", "")
        }
        
        # Calculate metrics
        state["confidence_score"], state["completeness_score"], state["missing_fields"] = calculate_metrics(parsed_schema)
        state["nct_id"] = "PDF-" + Path(state["input_url"]).stem
        
        # Clean up temp file if downloaded
        if state["input_url"].startswith('http'):
            import os
            if os.path.exists(file_path):
                os.remove(file_path)
        
    except Exception as e:
        state["error"] = f"PDF parsing error: {str(e)}"
    
    return state


def extract_from_url(state: WorkflowState) -> WorkflowState:
    """Node: Extract data from ClinicalTrials.gov URL - EXACT copy of original get_protocol_data"""
    try:
        # Extract NCT number
        nct_match = re.search(r"NCT\d{8}", state["input_url"])
        if not nct_match:
            state["error"] = "Invalid ClinicalTrials.gov URL - NCT number not found"
            return state
        
        nct_number = nct_match.group(0)
        
        # Fetch from API
        api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_number}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        study_data = response.json()
        state["raw_data"] = study_data
        
        protocol_section = study_data.get('protocolSection', {})
        results_section = study_data.get('resultsSection', {})
        
        if not protocol_section:
            state["error"] = "Error: Study data could not be found for this NCT number."
            return state

        # Identification Module
        identification_module = protocol_section.get('identificationModule', {})
        nct_id = identification_module.get('nctId', 'N/A')
        official_title = identification_module.get('officialTitle', 'N/A')
        state["nct_id"] = nct_id

        # Status Module
        status_module = protocol_section.get('statusModule', {})
        overall_status = status_module.get('overallStatus', 'N/A')
        
        # Description Module
        description_module = protocol_section.get('descriptionModule', {})
        brief_summary = description_module.get('briefSummary', 'N/A')
        detailed_description = description_module.get('detailedDescription', 'N/A')
        
        # Design Module
        design_module = protocol_section.get('designModule', {})
        study_type = design_module.get('studyType', 'N/A')
        study_phases = design_module.get('phases', [])
        study_phase = ", ".join(study_phases) if study_phases else 'N/A'
        
        design_info = design_module.get('designInfo', {})
        allocation = design_info.get('allocation', 'N/A')
        intervention_model = design_info.get('interventionModel', 'N/A')
        primary_purpose = design_info.get('primaryPurpose', 'N/A')
        
        masking_info = design_info.get('maskingInfo', {})
        masking = masking_info.get('masking', 'N/A')
        who_masked = masking_info.get('whoMasked', [])
        who_masked_str = ", ".join(who_masked) if who_masked else 'N/A'
        
        enrollment_info = design_module.get('enrollmentInfo', {})
        enrollment_count = enrollment_info.get('count', 'N/A')
        enrollment_type = enrollment_info.get('type', 'N/A')
        
        study_design_text = f"Study Type: {study_type}\nPhases: {study_phase}\nAllocation: {allocation}\nIntervention Model: {intervention_model}\nPrimary Purpose: {primary_purpose}\nMasking: {masking}\nWho Masked: {who_masked_str}\nEnrollment: {enrollment_count} ({enrollment_type})"
        
        # Interventions and Arm Groups
        arms_interventions_module = protocol_section.get('armsInterventionsModule', {})
        arm_groups_list = arms_interventions_module.get('armGroups', [])
        if not isinstance(arm_groups_list, list):
            arm_groups_list = []
        
        # Extract arm groups with enhanced dosing information
        arm_groups_text = ""
        for i, ag in enumerate(arm_groups_list, 1):
            arm_label = ag.get('label', f'Arm {i}')
            arm_type = ag.get('type', 'N/A')
            arm_description = ag.get('description', 'N/A')
            intervention_names = ag.get('interventionNames', [])
            intervention_names_str = ", ".join(intervention_names) if intervention_names else "N/A"
            
            # Try to extract dose information from description
            dose_info = ""
            if arm_description and arm_description != 'N/A':
                # Look for common dose patterns
                dose_patterns = [
                    r'(\d+(?:\.\d+)?)\s*mg/kg',  # mg/kg dosing
                    r'(\d+(?:\.\d+)?)\s*mg/m2',  # mg/m2 dosing  
                    r'(\d+(?:\.\d+)?)\s*mg',     # mg dosing
                    r'(\d+(?:\.\d+)?)\s*mcg',    # mcg dosing
                    r'(\d+(?:\.\d+)?)\s*units',  # units
                ]
                
                found_doses = []
                for pattern in dose_patterns:
                    matches = re.findall(pattern, arm_description, re.IGNORECASE)
                    if matches:
                        unit = pattern.split('\\s*')[1].replace(')', '')
                        for match in matches:
                            found_doses.append(f"{match} {unit}")
                
                if found_doses:
                    dose_info = f"  Doses: {', '.join(found_doses)}\n"
            
            arm_groups_text += f"**Arm {i}: {arm_label}**\n  Type: {arm_type}\n  Description: {arm_description}\n{dose_info}  Interventions: {intervention_names_str}\n\n"
        
        # Extract interventions with correct field names
        interventions_list = arms_interventions_module.get('interventions', [])
        if not isinstance(interventions_list, list):
            interventions_list = []
        
        # Extract interventions with enhanced drug information
        interventions_text = ""
        for i, intervention in enumerate(interventions_list, 1):
            name = intervention.get('name', 'N/A')
            int_type = intervention.get('type', 'N/A')
            description = intervention.get('description', 'N/A')
            arm_group_labels = intervention.get('armGroupLabels', [])
            other_names = intervention.get('otherNames', [])
            
            arm_labels_str = ", ".join(arm_group_labels) if arm_group_labels else "N/A"
            other_names_str = ", ".join(other_names) if other_names else "N/A"
            
            # Extract drug class or mechanism information from other names
            drug_info = ""
            if other_names:
                for other_name in other_names:
                    if any(keyword in other_name.upper() for keyword in ['ANTI-', 'INHIBITOR', 'AGONIST', 'ANTAGONIST']):
                        drug_info = f"  Mechanism: {other_name}\n"
                        break
            
            interventions_text += f"**Drug {i}: {name}**\n  Type: {int_type}\n  Description: {description}\n{drug_info}  Used in Arms: {arm_labels_str}\n  Other Names/Codes: {other_names_str}\n\n"

        # Eligibility Module
        eligibility_module = protocol_section.get('eligibilityModule', {})
        eligibility_criteria_data = eligibility_module.get('eligibilityCriteria', 'N/A')
        if isinstance(eligibility_criteria_data, dict):
            eligibility_criteria = eligibility_criteria_data.get('textblock', 'N/A')
        else:
            eligibility_criteria = eligibility_criteria_data
        
        # Enhanced outcomes extraction with objective identification
        outcomes_module = protocol_section.get('outcomesModule', {})
        primary_outcomes = outcomes_module.get('primaryOutcomes', [])
        secondary_outcomes = outcomes_module.get('secondaryOutcomes', [])
        
        outcomes_text = ""
        
        # Extract and categorize primary outcomes
        if primary_outcomes:
            safety_outcomes = []
            efficacy_outcomes = []
            pk_outcomes = []
            
            for outcome in primary_outcomes:
                measure = outcome.get('measure', 'N/A')
                description = outcome.get('description', 'N/A')
                time_frame = outcome.get('timeFrame', 'N/A')
                
                # Categorize outcomes based on keywords
                measure_lower = measure.lower()
                if any(keyword in measure_lower for keyword in ['safety', 'adverse', 'toxicity', 'mtd', 'dose']):
                    safety_outcomes.append({'measure': measure, 'description': description, 'time_frame': time_frame})
                elif any(keyword in measure_lower for keyword in ['response', 'efficacy', 'survival', 'progression']):
                    efficacy_outcomes.append({'measure': measure, 'description': description, 'time_frame': time_frame})
                elif any(keyword in measure_lower for keyword in ['pharmacokinetic', 'concentration', 'clearance']):
                    pk_outcomes.append({'measure': measure, 'description': description, 'time_frame': time_frame})
                else:
                    efficacy_outcomes.append({'measure': measure, 'description': description, 'time_frame': time_frame})
            
            outcomes_text += "**Primary Objectives:**\n"
            
            if safety_outcomes:
                outcomes_text += "\n*Safety Objectives:*\n"
                for outcome in safety_outcomes:
                    outcomes_text += f"- {outcome['measure']}\n  Description: {outcome['description']}\n  Time Frame: {outcome['time_frame']}\n\n"
            
            if efficacy_outcomes:
                outcomes_text += "\n*Efficacy Objectives:*\n"
                for outcome in efficacy_outcomes:
                    outcomes_text += f"- {outcome['measure']}\n  Description: {outcome['description']}\n  Time Frame: {outcome['time_frame']}\n\n"
            
            if pk_outcomes:
                outcomes_text += "\n*Pharmacokinetic Objectives:*\n"
                for outcome in pk_outcomes:
                    outcomes_text += f"- {outcome['measure']}\n  Description: {outcome['description']}\n  Time Frame: {outcome['time_frame']}\n\n"
        
        # Extract secondary outcomes
        if secondary_outcomes:
            outcomes_text += "**Secondary Objectives:**\n"
            for i, outcome in enumerate(secondary_outcomes[:10], 1):  # Limit to first 10
                measure = outcome.get('measure', 'N/A')
                description = outcome.get('description', 'N/A')
                time_frame = outcome.get('timeFrame', 'N/A')
                outcomes_text += f"{i}. {measure}\n   Description: {description}\n   Time Frame: {time_frame}\n\n"
            
            if len(secondary_outcomes) > 10:
                outcomes_text += f"... and {len(secondary_outcomes)-10} additional secondary outcomes\n"

        # Enhanced adverse events extraction grouped by organ system
        adverse_events_module = results_section.get('adverseEventsModule', {})
        serious_events = adverse_events_module.get('seriousEvents', [])
        if not isinstance(serious_events, list):
            serious_events = []
        other_events = adverse_events_module.get('otherEvents', [])
        if not isinstance(other_events, list):
            other_events = []
        
        adverse_events_text = ""
        if serious_events or other_events:
            if serious_events:
                # Group serious events by organ system
                serious_by_system = {}
                for event in serious_events:
                    term = event.get('term', 'N/A')
                    organ_system = event.get('organSystem', 'Other')
                    stats = event.get('stats', [])
                    total_affected = sum(stat.get('numAffected', 0) for stat in stats if isinstance(stat, dict))
                    total_at_risk = sum(stat.get('numAtRisk', 0) for stat in stats if isinstance(stat, dict))
                    
                    if organ_system not in serious_by_system:
                        serious_by_system[organ_system] = []
                    serious_by_system[organ_system].append(f"{term} ({total_affected}/{total_at_risk})")
                
                adverse_events_text += "\n**Serious Adverse Events by System:**\n"
                for system, events in serious_by_system.items():
                    adverse_events_text += f"\n**{system}:**\n"
                    for event in events[:5]:  # Limit to top 5 per system
                        adverse_events_text += f"- {event}\n"
                    if len(events) > 5:
                        adverse_events_text += f"- ... and {len(events)-5} more\n"
            
            if other_events:
                # Group common events by organ system (top 3 per system)
                common_by_system = {}
                for event in other_events:
                    term = event.get('term', 'N/A')
                    organ_system = event.get('organSystem', 'Other')
                    stats = event.get('stats', [])
                    total_affected = sum(stat.get('numAffected', 0) for stat in stats if isinstance(stat, dict))
                    total_at_risk = sum(stat.get('numAtRisk', 0) for stat in stats if isinstance(stat, dict))
                    
                    # Only include events affecting > 5% of patients
                    if total_at_risk > 0 and (total_affected / total_at_risk) > 0.05:
                        if organ_system not in common_by_system:
                            common_by_system[organ_system] = []
                        common_by_system[organ_system].append({
                            'term': term,
                            'affected': total_affected,
                            'at_risk': total_at_risk,
                            'rate': total_affected / total_at_risk
                        })
                
                # Sort by rate and take top events per system
                for system in common_by_system:
                    common_by_system[system].sort(key=lambda x: x['rate'], reverse=True)
                    common_by_system[system] = common_by_system[system][:3]
                
                if common_by_system:
                    adverse_events_text += "\n**Common Adverse Events by System (>5% incidence):**\n"
                    for system, events in common_by_system.items():
                        if events:  # Only show systems with events
                            adverse_events_text += f"\n**{system}:**\n"
                            for event in events:
                                rate_pct = event['rate'] * 100
                                adverse_events_text += f"- {event['term']}: {event['affected']}/{event['at_risk']} ({rate_pct:.1f}%)\n"
        else:
            adverse_events_text = "No adverse events reported in the structured API data."

        # Extract participant flow data for patient numbers
        participant_flow_text = ""
        if results_section:
            participant_flow_module = results_section.get('participantFlowModule', {})
            groups = participant_flow_module.get('groups', [])
            if groups:
                participant_flow_text += "**Participant Enrollment by Group:**\n"
                for group in groups:
                    group_title = group.get('title', 'N/A')
                    group_description = group.get('description', 'N/A')
                    participant_flow_text += f"- {group_title}: {group_description}\n"
                
                periods = participant_flow_module.get('periods', [])
                if periods:
                    for period in periods:
                        milestones = period.get('milestones', [])
                        for milestone in milestones:
                            if milestone.get('type') == 'STARTED':
                                participant_flow_text += "\n**Enrollment Numbers:**\n"
                                achievements = milestone.get('achievements', [])
                                for achievement in achievements:
                                    group_id = achievement.get('groupId', 'N/A')
                                    num_subjects = achievement.get('numSubjects', 'N/A')
                                    # Find corresponding group title
                                    group_title = next((g.get('title', 'N/A') for g in groups if g.get('id') == group_id), group_id)
                                    participant_flow_text += f"- {group_title}: {num_subjects} patients\n"
                                break
                        break
        
        # Extract sponsor and collaborator information
        sponsor_collaborators_module = protocol_section.get('sponsorCollaboratorsModule', {})
        lead_sponsor = sponsor_collaborators_module.get('leadSponsor', {})
        sponsor_name = lead_sponsor.get('name', 'N/A')
        sponsor_class = lead_sponsor.get('class', 'N/A')
        
        collaborators = sponsor_collaborators_module.get('collaborators', [])
        collaborator_text = ""
        if collaborators:
            collaborator_names = [collab.get('name', 'N/A') for collab in collaborators]
            collaborator_text = f"Collaborators: {', '.join(collaborator_names)}"
        
        sponsor_info = f"Lead Sponsor: {sponsor_name} ({sponsor_class})\n{collaborator_text}"
        
        # Extract contacts and locations for site information
        contacts_locations_module = protocol_section.get('contactsLocationsModule', {})
        locations = contacts_locations_module.get('locations', [])
        location_text = ""
        if locations:
            location_text += f"**Study Locations ({len(locations)} sites):**\n"
            # Group by country
            countries = {}
            for location in locations:
                country = location.get('country', 'Unknown')
                city = location.get('city', 'N/A')
                facility = location.get('facility', 'N/A')
                if country not in countries:
                    countries[country] = []
                countries[country].append(f"{facility}, {city}")
            
            for country, sites in countries.items():
                location_text += f"- {country}: {len(sites)} sites\n"
                # Show first few sites as examples
                for site in sites[:3]:
                    location_text += f"  • {site}\n"
                if len(sites) > 3:
                    location_text += f"  • ... and {len(sites)-3} more sites\n"
        
        # Extract basic demographic eligibility details
        eligibility_module = protocol_section.get('eligibilityModule', {})
        min_age = eligibility_module.get('minimumAge', 'N/A')
        max_age = eligibility_module.get('maximumAge', 'N/A')
        sex = eligibility_module.get('sex', 'N/A')
        healthy_volunteers = eligibility_module.get('healthyVolunteers', False)
        
        # Enhanced eligibility criteria processing
        def process_eligibility_criteria(eligibility_text):
            """Process eligibility criteria to separate inclusion and exclusion criteria"""
            if not eligibility_text or eligibility_text == 'N/A':
                return "No eligibility criteria available", [], []
            
            text = str(eligibility_text).strip()
            lines = text.split('\n')
            
            inclusion_criteria = []
            exclusion_criteria = []
            current_section = None
            
            inclusion_headers = ['inclusion criteria', 'inclusion', 'eligibility criteria', 'eligible participants']
            exclusion_headers = ['exclusion criteria', 'exclusion', 'excluded participants', 'exclusionary criteria']
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                if any(header in line_lower for header in inclusion_headers):
                    current_section = 'inclusion'
                    continue
                elif any(header in line_lower for header in exclusion_headers):
                    current_section = 'exclusion'
                    continue
                
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or 
                           line[0].isdigit() or line.startswith('o ')):
                    clean_line = line.strip()
                    
                    if clean_line.startswith('-'):
                        clean_line = clean_line[1:].strip()
                    elif clean_line.startswith('•'):
                        clean_line = clean_line[1:].strip()
                    elif clean_line.startswith('*'):
                        clean_line = clean_line[1:].strip()
                    elif clean_line[0].isdigit() and '. ' in clean_line[:5]:
                        period_index = clean_line.find('. ')
                        if period_index > 0 and period_index < 5:
                            clean_line = clean_line[period_index + 2:].strip()
                    
                    if len(clean_line) < 5:
                        continue
                    
                    if current_section == 'inclusion':
                        inclusion_criteria.append(clean_line)
                    elif current_section == 'exclusion':
                        exclusion_criteria.append(clean_line)
                    else:
                        if any(word in line_lower for word in ['must', 'should', 'required', 'age ≥', 'age >', 'performance status', 'confirmed', 'diagnosis']):
                            inclusion_criteria.append(clean_line)
                        elif any(word in line_lower for word in ['cannot', 'must not', 'prohibited', 'contraindicated', 'excluded']):
                            exclusion_criteria.append(clean_line)
                        else:
                            inclusion_criteria.append(clean_line)
                elif len(line) > 20 and current_section:
                    if current_section == 'inclusion':
                        inclusion_criteria.append(line)
                    elif current_section == 'exclusion':
                        exclusion_criteria.append(line)
            
            summary_parts = []
            if inclusion_criteria:
                key_inclusion = inclusion_criteria[:4]
                summary_parts.append(f"Key Inclusion: {'; '.join(key_inclusion[:2])}")
            
            if exclusion_criteria:
                key_exclusion = exclusion_criteria[:3]
                summary_parts.append(f"Key Exclusions: {'; '.join(key_exclusion[:2])}")
            
            summary = ". ".join(summary_parts) if summary_parts else "Standard eligibility criteria apply"
            
            return summary, inclusion_criteria, exclusion_criteria

        # Process eligibility criteria
        eligibility_criteria_summary, inclusion_list, exclusion_list = process_eligibility_criteria(eligibility_criteria)
        
        # Create detailed eligibility text
        detailed_eligibility = ""
        
        detailed_eligibility += f"**Demographics:** Age {min_age}"
        if max_age and max_age != 'N/A':
            detailed_eligibility += f" to {max_age}"
        detailed_eligibility += f", {sex}, Healthy volunteers: {'Yes' if healthy_volunteers else 'No'}\n\n"
        
        if inclusion_list:
            detailed_eligibility += "**Inclusion Criteria:**\n"
            for i, criterion in enumerate(inclusion_list[:8], 1):
                detailed_eligibility += f"{i}. {criterion}\n"
            if len(inclusion_list) > 8:
                detailed_eligibility += f"... and {len(inclusion_list)-8} additional inclusion criteria\n"
            detailed_eligibility += "\n"
        
        if exclusion_list:
            detailed_eligibility += "**Exclusion Criteria:**\n"
            for i, criterion in enumerate(exclusion_list[:8], 1):
                detailed_eligibility += f"{i}. {criterion}\n"
            if len(exclusion_list) > 8:
                detailed_eligibility += f"... and {len(exclusion_list)-8} additional exclusion criteria\n"
            detailed_eligibility += "\n"
        
        if not inclusion_list and not exclusion_list and eligibility_criteria != 'N/A':
            detailed_eligibility += "**Full Eligibility Criteria:**\n"
            if len(eligibility_criteria) > 800:
                detailed_eligibility += eligibility_criteria[:800] + "... [truncated]"
            else:
                detailed_eligibility += eligibility_criteria
        
        eligibility_comprehensive = f"{eligibility_criteria_summary}\n\n{detailed_eligibility.strip()}"

        # Structured data for section-wise summarization
        data_to_summarize = {
            "Study Overview": f"{official_title} | Status: {overall_status} | Type: {study_type} - {study_phase}",
            "Brief Description": brief_summary,
            "Primary and Secondary Objectives": outcomes_text if outcomes_text else None,
            "Treatment Arms and Interventions": f"{arm_groups_text}\n\n{interventions_text}" if (arm_groups_text or interventions_text) else None,
            "Eligibility Criteria": eligibility_comprehensive,
            "Enrollment and Participant Flow": participant_flow_text if participant_flow_text else None,
            "Adverse Events Profile": adverse_events_text if adverse_events_text and "No adverse events reported" not in adverse_events_text else None,
            "Study Locations": f"{len(locations)} sites across {len(set(loc.get('country', 'Unknown') for loc in locations))} countries" if locations else None,
            "Sponsor Information": sponsor_info if sponsor_info and sponsor_name != "N/A" else None
        }
        
        # Create parsed_json with proper field names (lowercase with underscores)
        # Preserve None values to accurately calculate metrics
        parsed_json = {
            "study_overview": data_to_summarize.get("Study Overview"),
            "brief_description": data_to_summarize.get("Brief Description"),
            "primary_secondary_objectives": data_to_summarize.get("Primary and Secondary Objectives"),
            "treatment_arms_interventions": data_to_summarize.get("Treatment Arms and Interventions"),
            "eligibility_criteria": data_to_summarize.get("Eligibility Criteria"),
            "enrollment_participant_flow": data_to_summarize.get("Enrollment and Participant Flow"),
            "adverse_events_profile": data_to_summarize.get("Adverse Events Profile"),
            "study_locations": data_to_summarize.get("Study Locations"),
            "sponsor_information": data_to_summarize.get("Sponsor Information")
        }
        
        state["data_to_summarize"] = data_to_summarize
        state["parsed_json"] = parsed_json
        
        # Calculate metrics based on parsed_json
        state["confidence_score"], state["completeness_score"], state["missing_fields"] = calculate_metrics(parsed_json)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            state["error"] = f"Error: Study with NCT number was not found on ClinicalTrials.gov."
        else:
            state["error"] = f"HTTP error occurred while fetching the protocol: {e}"
    except Exception as e:
        state["error"] = f"An error occurred while fetching the protocol: {e}"
    
    return state
    """Node: Extract data from ClinicalTrials.gov URL (EXACT copy of app_v1 get_protocol_data)"""
    try:
        # Extract NCT number
        nct_match = re.search(r"NCT\d{8}", state["input_url"])
        if not nct_match:
            state["error"] = "Invalid ClinicalTrials.gov URL - NCT number not found"
            return state
        
        nct_number = nct_match.group(0)
        state["nct_id"] = nct_number
        
        # Fetch from API (same as app_v1)
        api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_number}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        study_data = response.json()
        state["raw_data"] = study_data
        
        protocol_section = study_data.get('protocolSection', {})
        results_section = study_data.get('resultsSection', {})
        
        if not protocol_section:
            state["error"] = "Error: Study data could not be found for this NCT number."
            return state

        # Extract all data EXACTLY as in app_v1.py
        # (This is a condensed version - the full extraction logic from app_v1)
        identification_module = protocol_section.get('identificationModule', {})
        official_title = identification_module.get('officialTitle', 'N/A')
        
        status_module = protocol_section.get('statusModule', {})
        overall_status = status_module.get('overallStatus', 'N/A')
        
        description_module = protocol_section.get('descriptionModule', {})
        brief_summary = description_module.get('briefSummary', 'N/A')
        
        design_module = protocol_section.get('designModule', {})
        study_type = design_module.get('studyType', 'N/A')
        study_phases = design_module.get('phases', [])
        study_phase = ", ".join(study_phases) if study_phases else 'N/A'
        
        arms_interventions_module = protocol_section.get('armsInterventionsModule', {})
        arm_groups_list = arms_interventions_module.get('armGroups', [])
        interventions_list = arms_interventions_module.get('interventions', [])
        
        # Format arms
        arm_groups_text = ""
        for i, ag in enumerate(arm_groups_list, 1):
            arm_label = ag.get('label', f'Arm {i}')
            arm_type = ag.get('type', 'N/A')
            arm_description = ag.get('description', 'N/A')
            intervention_names = ag.get('interventionNames', [])
            intervention_names_str = ", ".join(intervention_names) if intervention_names else "N/A"
            arm_groups_text += f"**Arm {i}: {arm_label}**\n  Type: {arm_type}\n  Description: {arm_description}\n  Interventions: {intervention_names_str}\n\n"
        
        # Format interventions
        interventions_text = ""
        for i, intervention in enumerate(interventions_list, 1):
            name = intervention.get('name', 'N/A')
            int_type = intervention.get('type', 'N/A')
            description = intervention.get('description', 'N/A')
            interventions_text += f"**Drug {i}: {name}**\n  Type: {int_type}\n  Description: {description}\n\n"
        
        eligibility_module = protocol_section.get('eligibilityModule', {})
        eligibility_criteria = eligibility_module.get('eligibilityCriteria', 'N/A')
        
        outcomes_module = protocol_section.get('outcomesModule', {})
        primary_outcomes = outcomes_module.get('primaryOutcomes', [])
        secondary_outcomes = outcomes_module.get('secondaryOutcomes', [])
        
        outcomes_text = "**Primary Objectives:**\n"
        for outcome in primary_outcomes[:5]:
            measure = outcome.get('measure', 'N/A')
            outcomes_text += f"- {measure}\n"
        
        if secondary_outcomes:
            outcomes_text += "\n**Secondary Objectives:**\n"
            for outcome in secondary_outcomes[:5]:
                measure = outcome.get('measure', 'N/A')
                outcomes_text += f"- {measure}\n"
        
        # Participant flow
        participant_flow_text = ""
        if results_section:
            participant_flow_module = results_section.get('participantFlowModule', {})
            groups = participant_flow_module.get('groups', [])
            if groups:
                participant_flow_text += "**Participant Enrollment:**\n"
                for group in groups:
                    group_title = group.get('title', 'N/A')
                    group_description = group.get('description', 'N/A')
                    participant_flow_text += f"- {group_title}: {group_description}\n"
        
        # Adverse events
        adverse_events_text = ""
        adverse_events_module = results_section.get('adverseEventsModule', {})
        serious_events = adverse_events_module.get('seriousEvents', [])
        other_events = adverse_events_module.get('otherEvents', [])
        
        if serious_events or other_events:
            adverse_events_text += "**Adverse Events Reported:**\n"
            if serious_events:
                adverse_events_text += f"- Serious events: {len(serious_events)}\n"
            if other_events:
                adverse_events_text += f"- Other events: {len(other_events)}\n"
        else:
            adverse_events_text = "No adverse events reported in the structured API data."
        
        # Locations
        contacts_locations_module = protocol_section.get('contactsLocationsModule', {})
        locations = contacts_locations_module.get('locations', [])
        location_text = ""
        if locations:
            location_text = f"{len(locations)} sites across multiple countries"
        
        # Sponsor
        sponsor_collaborators_module = protocol_section.get('sponsorCollaboratorsModule', {})
        lead_sponsor = sponsor_collaborators_module.get('leadSponsor', {})
        sponsor_name = lead_sponsor.get('name', 'N/A')
        sponsor_class = lead_sponsor.get('class', 'N/A')
        sponsor_info = f"Lead Sponsor: {sponsor_name} ({sponsor_class})"
        
        # Create data_to_summarize dict (same format as app_v1 - for display)
        data_to_summarize = {
            "Study Overview": f"{official_title} | Status: {overall_status} | Type: {study_type} - {study_phase}",
            "Brief Description": brief_summary,
            "Primary and Secondary Objectives": outcomes_text if outcomes_text else None,
            "Treatment Arms and Interventions": f"{arm_groups_text}\n\n{interventions_text}" if (arm_groups_text or interventions_text) else None,
            "Eligibility Criteria": eligibility_criteria,
            "Enrollment and Participant Flow": participant_flow_text if participant_flow_text else None,
            "Adverse Events Profile": adverse_events_text if adverse_events_text and "No adverse events reported" not in adverse_events_text else None,
            "Study Locations": location_text if location_text else None,
            "Sponsor Information": sponsor_info if sponsor_info and sponsor_name != "N/A" else None
        }
        
        # Create parsed_json with proper field names (lowercase with underscores)
        # Preserve None values to accurately calculate metrics
        parsed_json = {
            "study_overview": data_to_summarize.get("Study Overview"),
            "brief_description": data_to_summarize.get("Brief Description"),
            "primary_secondary_objectives": data_to_summarize.get("Primary and Secondary Objectives"),
            "treatment_arms_interventions": data_to_summarize.get("Treatment Arms and Interventions"),
            "eligibility_criteria": data_to_summarize.get("Eligibility Criteria"),
            "enrollment_participant_flow": data_to_summarize.get("Enrollment and Participant Flow"),
            "adverse_events_profile": data_to_summarize.get("Adverse Events Profile"),
            "study_locations": data_to_summarize.get("Study Locations"),
            "sponsor_information": data_to_summarize.get("Sponsor Information")
        }
        
        state["data_to_summarize"] = data_to_summarize
        state["parsed_json"] = parsed_json  # Use proper field names for metrics
        
        # Calculate metrics based on parsed_json
        state["confidence_score"], state["completeness_score"], state["missing_fields"] = calculate_metrics(parsed_json)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            state["error"] = f"Error: Study with NCT number was not found on ClinicalTrials.gov."
        else:
            state["error"] = f"HTTP error occurred while fetching the protocol: {e}"
    except Exception as e:
        state["error"] = f"An error occurred while fetching the protocol: {e}"
    
    return state


def should_use_rag_tool(query: str) -> bool:
    """Use LLM to determine if a query should use the RAG tool for clinical trials search"""
    if not query or not RAG_TOOL_AVAILABLE:
        return False
    
    # Ensure query is a string
    if isinstance(query, dict):
        query = str(query)
    elif not isinstance(query, str):
        query = str(query)
    
    # Use LLM to determine if this query needs RAG search
    try:
        detection_prompt = f"""You are an AI assistant that determines if a user query requires searching a clinical trials database.

User query: "{query}"

Analyze this query and determine if it:
1. Asks for similar studies, trials, or research
2. Requests information about other treatments, drugs, or interventions
3. Seeks comparative information about clinical trials
4. Asks about alternative therapies or medications
5. Requests finding or searching for clinical trials

Respond with only "YES" if the query requires database search, or "NO" if it doesn't.

Examples:
- "Can you fetch similar studies using DTG + TAF + FTC?" → YES
- "What other trials use pembrolizumab?" → YES
- "Find studies with similar drugs" → YES
- "What is the primary endpoint of this study?" → NO
- "Generate summary" → NO"""
        
        detection_response = llm.invoke([HumanMessage(content=detection_prompt)])
        result = detection_response.content.strip().upper()
        
        return result == "YES"
        
    except Exception as e:
        print(f"WARNING: LLM detection error: {e}")
        # Fallback to simple keyword detection
        query_lower = query.lower()
        simple_keywords = ['similar', 'other', 'find', 'search', 'fetch', 'show', 'list', 'compare']
        return any(keyword in query_lower for keyword in simple_keywords)


def generate_rag_search_query(user_query: str, current_study_data: dict = None) -> str:
    """Use LLM to generate an appropriate search query for the RAG tool based on user intent"""
    
    # Ensure query is a string
    if isinstance(user_query, dict):
        user_query = str(user_query)
    elif not isinstance(user_query, str):
        user_query = str(user_query)
    
    try:
        # Prepare context about current study if available
        study_context = ""
        if current_study_data:
            # Extract key intervention information from current study
            interventions = current_study_data.get("Treatment Arms and Interventions", "")
            if isinstance(interventions, dict):
                interventions = interventions.get("content", "")
            
            study_overview = current_study_data.get("Study Overview", "")
            if isinstance(study_overview, dict):
                study_overview = study_overview.get("content", "")
            
            if interventions or study_overview:
                study_context = f"""
Current study context:
- Study Overview: {study_overview[:200] if study_overview else 'Not available'}
- Interventions: {interventions[:200] if interventions else 'Not available'}"""
        
        query_generation_prompt = f"""You are an expert in clinical trials research. A user is asking about finding similar clinical trials.

User query: "{user_query}"
{study_context}

Your task: Generate a precise search query to find relevant clinical trials in a database. The search query should capture:
1. Key drug names, interventions, or treatment combinations mentioned
2. Disease areas or conditions if specified
3. Treatment modalities (e.g., immunotherapy, chemotherapy, combination therapy)

Instructions:
- Extract and normalize drug names (e.g., "DTG" → "dolutegravir", "FTC" → "emtricitabine")
- Include both abbreviations and full names when possible
- For combination therapies, include individual components
- Focus on the most relevant search terms that would find similar studies
- Keep the search query concise but comprehensive
- If no specific drugs are mentioned, use broader treatment categories

Examples:
- User: "fetch similar studies using DTG + TAF + FTC" → "dolutegravir tenofovir emtricitabine HIV combination"
- User: "find other pembrolizumab trials" → "pembrolizumab immunotherapy anti-PD1"
- User: "show me similar cancer studies" → "cancer oncology treatment"

Generate only the search query, no explanations:"""
        
        response = llm.invoke([HumanMessage(content=query_generation_prompt)])
        search_query = response.content.strip()
        
        # Fallback if search query is empty or too short
        if not search_query or len(search_query.split()) < 2:
            search_query = user_query.strip()
        
        return search_query
        
    except Exception as e:
        print(f"⚠️  LLM search query generation error: {e}")
        # Fallback to using the user query directly
        return user_query.strip()


def calculate_metrics(parsed_json: dict) -> tuple[float, float, list]:
    """Calculate confidence and completeness scores based on meaningful content (using word count)"""
    required_fields = [
        "study_overview",
        "brief_description",
        "primary_secondary_objectives",
        "treatment_arms_interventions",
        "eligibility_criteria",
        "enrollment_participant_flow",
        "adverse_events_profile",
        "study_locations",
        "sponsor_information"
    ]
    
    filled_fields = 0
    missing_fields = []
    total_words = 0
    field_debug_info = {}
    
    for field in required_fields:
        raw_value = parsed_json.get(field, "")
        
        # Extract content from dict format or use string directly
        if isinstance(raw_value, dict):
            content = raw_value.get("content", "")
        else:
            content = raw_value
        
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        # Get word count for debugging
        word_count = len(content.split()) if content else 0
        field_debug_info[field] = {
            "chars": len(content),
            "words": word_count,
            "has_content": bool(content and content.strip())
        }
        
        # Check for meaningful content (not just "N/A" or short placeholders)
        # Lower threshold from 30 to 10 characters to catch more valid extractions
        if (content and 
            content.strip() != "N/A" and
            content.strip() != "" and
            content.strip() != "Not found in provided text" and
            "not available" not in content.lower() and
            "no data" not in content.lower() and
            len(content.strip()) > 10):  # Lowered threshold from 30 to 10
            filled_fields += 1
            total_words += word_count
        else:
            missing_fields.append(field)
    
    # Completeness: percentage of fields with meaningful data
    completeness_score = filled_fields / len(required_fields) if required_fields else 0.0
    
    # Confidence: based on average word count per field
    if filled_fields > 0:
        avg_word_count = total_words / filled_fields
        # More generous confidence scaling: 15 words = 30%, 50 words = 70%, 100+ = 95%
        confidence_score = min(0.95, (avg_word_count - 15) / 85 + 0.3)
        confidence_score = max(0.1, confidence_score)  # Ensure minimum 10% if any content
    else:
        confidence_score = 0.0
    
    # Log debug info - commented out to avoid logger issues
    # logger.debug(f"Metrics calculation debug:")
    # for field, info in field_debug_info.items():
    #     logger.debug(f"  {field}: {info['words']} words, {info['chars']} chars, has_content={info['has_content']}")
    # logger.debug(f"Filled fields: {filled_fields}/{len(required_fields)}, total_words: {total_words}")
    
    return confidence_score, completeness_score, missing_fields


def check_quality(state: WorkflowState) -> Literal["llm_fallback", "chat_node"]:
    """Conditional edge: Check if LLM fallback is needed - triggers if completeness < 90%"""
    if state["confidence_score"] < 0.9 or state["completeness_score"] < 0.9:
        return "llm_fallback"
    else:
        return "chat_node"


def llm_fallback_old(state: WorkflowState) -> WorkflowState:
    """Node: Use LLM to extract missing fields from full document with chunking (OLD VERSION - KEPT FOR REFERENCE)"""
    try:
        # Mark that LLM fallback is being used
        state["used_llm_fallback"] = True
        
        # Get full document text
        if state["input_type"] == "pdf":
            parser = EnhancedClinicalTrialParser()
            
            file_path = state["input_url"]
            # Handle URLs
            if file_path.startswith('http'):
                response = requests.get(file_path)
                temp_path = f"temp_llm_fallback.pdf"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                file_path = temp_path
            
            # Extract full text using enhanced parser
            full_text, metadata = parser.extract_text_multimethod(file_path)
            
            # Clean up temp file
            if state["input_url"].startswith('http'):
                import os
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            full_text = json.dumps(state["raw_data"], indent=2)
        
        # LLM extraction prompt
        system_prompt = """You are a clinical trial data extraction expert. 
Extract the following fields from the document chunk provided. 
Preserve original content exactly with references, do not modify or summarize.

Required fields:
1. study_overview
2. brief_description
3. primary_secondary_objectives
4. treatment_arms_interventions
5. eligibility_criteria
6. enrollment_participant_flow
7. adverse_events_profile
8. study_locations
9. sponsor_information

Return as JSON. If a field is not found in this chunk, return null for that field."""

        # Chunk the document into 30k character chunks
        chunk_size = 30000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        # Process each chunk and merge results
        all_extracted = {}
        for i, chunk in enumerate(chunks):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Document chunk {i+1}/{len(chunks)}:\n\n{chunk}")
            ]
            
            response = llm.invoke(messages)
            
            # Parse LLM response
            try:
                chunk_extracted = json.loads(response.content)
                # Merge non-null fields
                for field, value in chunk_extracted.items():
                    if value and value != "null" and (field not in all_extracted or not all_extracted.get(field)):
                        all_extracted[field] = value
            except json.JSONDecodeError:
                continue
        
        # Merge with existing data (only update missing or empty fields)
        for field, value in all_extracted.items():
            if value and (field not in state["parsed_json"] or not state["parsed_json"].get(field)):
                state["parsed_json"][field] = value
        
        # Recalculate metrics
        state["confidence_score"], state["completeness_score"], state["missing_fields"] = calculate_metrics(state["parsed_json"])
        
    except Exception as e:
        state["error"] = f"LLM fallback error: {str(e)}"
    
    return state


def llm_fallback(state: WorkflowState) -> WorkflowState:
    """Node: Use multi-turn LLM extraction to handle large PDFs without rate limiting"""
    
    # Helper function to log progress
    def log_progress(message: str):
        """Add progress message to state and print to console"""
        if "progress_log" not in state:
            state["progress_log"] = []
        state["progress_log"].append(message)
        print(message)
    
    try:
        # Mark that LLM fallback is being used
        state["used_llm_fallback"] = True
        
        # Save parser-only output before LLM enhancement
        import copy
        state["parser_only_json"] = copy.deepcopy(state.get("parsed_json", {}))
        log_progress("📋 Saved parser-only output for comparison")
        
        # Get full document text
        if state["input_type"] == "pdf":
            parser = EnhancedClinicalTrialParser()
            
            file_path = state["input_url"]
            # Handle URLs
            if file_path.startswith('http'):
                response = requests.get(file_path)
                temp_path = f"temp_llm_fallback.pdf"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                file_path = temp_path
            
            # Extract full text using enhanced parser
            full_text, metadata = parser.extract_text_multimethod(file_path)
            
            # Clean up temp file
            if state["input_url"].startswith('http'):
                import os
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            # For URLs, use the raw API data
            full_text = json.dumps(state["raw_data"], indent=2)
        
        log_progress(f"📄 Document size: {len(full_text):,} characters ({len(full_text.split()):,} words)")
        
        # Cache full text for later use (like re-extraction)
        state["full_text"] = full_text
        
        # Estimate cost and time for multi-turn extraction
        if MULTI_TURN_AVAILABLE:
            cost_estimate = estimate_extraction_cost(full_text, model="gpt-4o-mini")
            state["extraction_cost_estimate"] = cost_estimate
            
            log_progress(f"💰 Estimated cost: ${cost_estimate['total_cost']:.3f}")
            log_progress(f"⏱️  Estimated time: {cost_estimate['estimated_time_minutes']:.1f} minutes")
            log_progress(f"🔢 Total tokens: {cost_estimate['total_tokens']:,}")
            
            # Use multi-turn extraction to avoid rate limits
            log_progress(f"🔄 Using multi-turn extraction (9 fields, one at a time)")
            
            extractor = MultiTurnExtractor(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens_per_call=180000,  # Leave buffer under 200k limit
                delay_between_calls=2.0  # 2 seconds between calls
            )
            
            # Progress callback for state updates
            def progress_callback(field_name: str, current: int, total: int):
                state["extraction_progress"] = {
                    "current_field": field_name,
                    "completed": current - 1,
                    "total": total
                }
                log_progress(f"  [{current}/{total}] Extracting: {field_name}")
            
            # Extract all fields using multi-turn strategy
            extracted_data = extractor.extract_with_retry(
                full_text,
                progress_callback=progress_callback,
                max_retries=2
            )
            
            log_progress(f"✅ Multi-turn extraction complete: {len(extracted_data)} fields")
            
        else:
            # Fallback to single-call extraction if multi-turn not available
            log_progress(f"⚠️  Multi-turn extractor not available, using single-call fallback")
            log_progress(f"⚠️  This may hit rate limits for large documents!")
            
            extracted_data = _single_call_extraction(full_text)
        
        # Debug: Show what was extracted
        total_words = 0
        for field, value in extracted_data.items():
            if value and value != "null" and value:
                word_count = len(str(value).split()) if value else 0
                total_words += word_count
                log_progress(f"   ✓ {field}: {word_count:,} words")
            else:
                log_progress(f"   ✗ {field}: EMPTY")
        
        log_progress(f"📊 Total extracted: {total_words:,} words")
        
        # Update state with LLM extracted data (replace all fields) and enhance with page tracking
        full_text_for_pages = state.get("full_text", full_text)
        
        for field, value in extracted_data.items():
            if value and value != "null" and value:
                # If the value is already a dict with page_references, keep it
                if isinstance(value, dict) and "page_references" in value:
                    state["parsed_json"][field] = value
                else:
                    # Extract page numbers for this content
                    content_str = value if isinstance(value, str) else str(value)
                    page_refs = []
                    
                    # Try to extract page numbers from the full text
                    if full_text_for_pages and len(content_str) > 20:
                        import re
                        # Find page markers around content snippets
                        content_words = content_str.lower().split()[:15]  # First 15 words
                        for i, word in enumerate(content_words):
                            if len(word) > 4:  # Skip small words
                                # Find this word in the full text and get surrounding page markers
                                pattern = re.escape(word)
                                matches = list(re.finditer(pattern, full_text_for_pages, re.IGNORECASE))
                                for match in matches[:3]:  # Check first 3 matches
                                    # Look backwards for page marker
                                    text_before = full_text_for_pages[:match.start()]
                                    page_matches = list(re.finditer(r'--- Page (\\d+) ---', text_before))
                                    if page_matches:
                                        try:
                                            page_num = int(page_matches[-1].group(1))
                                            page_refs.append(page_num)
                                        except (ValueError, IndexError):
                                            continue
                    
                    # Create enhanced value with page references
                    enhanced_value = {
                        "content": content_str,
                        "page_references": sorted(list(set(page_refs))) if page_refs else []
                    }
                    
                    state["parsed_json"][field] = enhanced_value
        
        # Also update data_to_summarize for display
        field_mapping = {
            "study_overview": "Study Overview",
            "brief_description": "Brief Description",
            "primary_secondary_objectives": "Primary and Secondary Objectives",
            "treatment_arms_interventions": "Treatment Arms and Interventions",
            "eligibility_criteria": "Eligibility Criteria",
            "enrollment_participant_flow": "Enrollment and Participant Flow",
            "adverse_events_profile": "Adverse Events Profile",
            "study_locations": "Study Locations",
            "sponsor_information": "Sponsor Information"
        }
        
        for field, value in extracted_data.items():
            if value and value != "null" and value:
                display_key = field_mapping.get(field, field)
                
                # Handle both dict format (with page_references) and string format
                if isinstance(state["parsed_json"][field], dict):
                    content = state["parsed_json"][field].get("content", "")
                else:
                    content = str(value) if value else ""
                
                if content and content.strip():
                    state["data_to_summarize"][display_key] = content
        
        log_progress(f"✅ State updated with extracted data")
        
        # Recalculate metrics
        state["confidence_score"], state["completeness_score"], state["missing_fields"] = calculate_metrics(state["parsed_json"])
        log_progress(f"📊 Final metrics - Confidence: {state['confidence_score']:.1%}, Completeness: {state['completeness_score']:.1%}")
        
    except Exception as e:
        log_progress(f"❌ Multi-turn extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        state["error"] = f"LLM fallback error: {str(e)}"
    
    return state


def _single_call_extraction(full_text: str) -> dict:
    """
    Fallback single-call extraction (may hit rate limits for large documents).
    Used only if MultiTurnExtractor is not available.
    """
    # Enhanced LLM extraction prompt
    system_prompt = """You are a clinical trial data extraction expert.

Extract structured information from the clinical trial document and create a comprehensive JSON output.

**CRITICAL REQUIREMENTS:**
1. Extract EXACT text from the original document - do NOT paraphrase
2. Include ALL relevant details from the text
3. Preserve technical terminology, drug names, dosages, measurements exactly as written
4. If information spans multiple sections, include all relevant content

**Required Fields to Extract:**

1. **study_overview**: Title, NCT ID, protocol number, phase, study type, disease
2. **brief_description**: Study's brief summary or background (first 500-1000 words)
3. **primary_secondary_objectives**: Primary and secondary endpoints with exact outcome measures
4. **treatment_arms_interventions**: All treatment arms, drug names, doses, schedules
5. **eligibility_criteria**: Complete inclusion and exclusion criteria
6. **enrollment_participant_flow**: Target enrollment, actual enrollment, patient disposition
7. **adverse_events_profile**: Adverse event tables, serious adverse events, Grade 3+ events
8. **study_locations**: Site names, cities, countries, principal investigators
9. **sponsor_information**: Sponsor name, medical monitor, CRO information

**Output Format:**
Return a valid JSON object:
{
  "study_overview": "EXACT TEXT",
  "brief_description": "EXACT TEXT",
  "primary_secondary_objectives": "EXACT TEXT",
  "treatment_arms_interventions": "EXACT TEXT",
  "eligibility_criteria": "EXACT TEXT",
  "enrollment_participant_flow": "EXACT TEXT",
  "adverse_events_profile": "EXACT TEXT",
  "study_locations": "EXACT TEXT",
  "sponsor_information": "EXACT TEXT"
}

If a field cannot be found, use "" (empty string).
Return ONLY the JSON object, no additional text."""

    user_prompt = f"""Clinical Trial Document:

{full_text}

---

Extract all required fields following the instructions."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Invoke LLM with full document
    response = llm.invoke(messages)
    
    # Parse LLM response - safely handle different content types
    raw_content = response.content
    if isinstance(raw_content, str):
        response_content = raw_content.strip()
    elif isinstance(raw_content, dict):
        response_content = json.dumps(raw_content)
    elif isinstance(raw_content, list):
        # Handle list of content blocks
        text_parts = []
        for item in raw_content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(json.dumps(item))
            else:
                text_parts.append(str(item))
        response_content = "".join(text_parts).strip()
    else:
        response_content = str(raw_content).strip() if raw_content else ""
    
    # Remove markdown code blocks if present
    if response_content.startswith('```json'):
        response_content = response_content[7:]
    if response_content.startswith('```'):
        response_content = response_content[3:]
    if response_content.endswith('```'):
        response_content = response_content[:-3]
    
    extracted_data = json.loads(response_content.strip())
    
    return extracted_data


def rag_search_node(state: WorkflowState) -> WorkflowState:
    """Node: Dedicated RAG search for clinical trials"""
    try:
        query = state.get("chat_query", "")
        
        # Ensure query is a string
        if isinstance(query, dict):
            query = str(query)
        elif not isinstance(query, str):
            query = str(query)
        
        print(f"RAG Search Node - Processing query: {query}")
        
        # Generate search query using LLM based on user intent
        current_study_data = state.get("data_to_summarize", {})
        search_query = generate_rag_search_query(query, current_study_data)
        
        print(f"Generated search query: {search_query}")
        
        if not search_query or len(search_query.strip()) < 3:
            print(f"WARNING: Could not generate meaningful search query from: {query}")
            state["rag_tool_results"] = "Could not generate appropriate search query from your request."
            state["use_rag_tool"] = False
            state["error"] = "Could not generate search query for RAG search"
            return state
        
        if not RAG_TOOL_AVAILABLE:
            print(f"WARNING: RAG tool not available")
            state["rag_tool_results"] = "Clinical trials search database is not available."
            state["use_rag_tool"] = False
            state["error"] = "RAG tool not available"
            return state
        
        try:
            # Initialize RAG tool
            rag_tool = create_clinical_trials_rag_tool()
            
            # Search for similar studies using generated query
            print(f"Searching RAG database for: {search_query}")
            rag_results = rag_tool._run(
                search_query=search_query,  # Using search_query instead of drug_name
                n_results=8,  # Get more results for better context
                conditions=None  # Could be enhanced to extract conditions from query
            )
            
            # Store RAG results
            state["rag_tool_results"] = rag_results
            state["use_rag_tool"] = True
            
            print(f"RAG search completed for '{search_query}'")
            print(f"RAG results length: {len(rag_results)} characters")
            
        except Exception as e:
            print(f"ERROR: RAG tool error: {e}")
            import traceback
            traceback.print_exc()
            state["rag_tool_results"] = f"Error during RAG search: {str(e)}"
            state["use_rag_tool"] = False
            state["error"] = f"RAG search failed: {str(e)}"
            
    except Exception as e:
        print(f"ERROR: RAG search node error: {e}")
        import traceback
        traceback.print_exc()
        state["rag_tool_results"] = f"RAG search node error: {str(e)}"
        state["use_rag_tool"] = False
        state["error"] = f"RAG node failed: {str(e)}"
    
    return state


def route_decision_node(state: WorkflowState) -> WorkflowState:
    """Node: Just pass through - routing is handled by conditional edges"""
    return state


def route_to_rag_or_chat(state: WorkflowState) -> Literal["rag_search", "chat_node"]:
    """Conditional edge: Route to RAG search or directly to chat based on query type"""
    query = state.get("chat_query", "")
    
    # Ensure query is a string
    if isinstance(query, dict):
        query = str(query)
    elif not isinstance(query, str):
        query = str(query)
    
    # Use RAG tool for similarity searches
    if query and query != "generate_summary" and should_use_rag_tool(query):
        return "rag_search"
    else:
        return "chat_node"


def chat_node(state: WorkflowState) -> WorkflowState:
    """Node: Handle chat interactions with Q&A (with streaming support like app_v1)"""
    try:
        query = state.get("chat_query", "")
        
        # Ensure query is a string
        if isinstance(query, dict):
            query = str(query)
        elif not isinstance(query, str):
            query = str(query)
        
        # Check if we have RAG results from the rag_search_node
        if state.get("use_rag_tool", False) and state.get("rag_tool_results"):
            print(f"Chat Node - Processing RAG-enhanced query: {query}")
            
            # Create enhanced response using both study data and RAG results
            data_to_summarize = state.get("data_to_summarize", {})
            rag_results = state["rag_tool_results"]
            
            # Handle both RAG-only queries and queries with study context
            if data_to_summarize and any(data_to_summarize.values()):
                # We have current study data - compare with RAG results
                context = json.dumps(data_to_summarize, indent=2)
                enhanced_prompt = f"""You are a clinical research assistant. The user asked: "{query}"

You have access to two sources of information:

1. **Current Study Data:**
{context}

2. **Similar Clinical Trials from Database Search:**
{rag_results}

**Instructions:**
- Answer the user's question about similar studies using the database search results
- Focus on studies that use similar drug combinations or individual components
- Be specific about study NCT IDs, drug names, and provide clickable URLs
- Highlight key similarities and differences in treatments, phases, and outcomes
- Compare the found studies with the current study data when relevant
- Keep the response well-organized and informative
- If no similar studies were found, explain that and suggest alternative approaches

Please provide a comprehensive answer that addresses the user's question about similar studies."""
            else:
                # RAG-only query without current study context
                enhanced_prompt = f"""You are a clinical research assistant. The user asked: "{query}"

You have access to clinical trials database search results:

**Similar Clinical Trials from Database Search:**
{rag_results}

**Instructions:**
- Answer the user's question using the database search results
- Focus on studies that match the requested drug combinations or criteria
- Be specific about study NCT IDs, drug names, and provide clickable URLs
- Highlight key information about treatments, phases, and outcomes
- Keep the response well-organized and informative
- If no similar studies were found in the database, explain that clearly and suggest alternative approaches

Please provide a comprehensive answer that addresses the user's question about similar studies."""
            
            messages = [
                SystemMessage(content="You are a clinical research assistant expert at finding and explaining clinical trial similarities. When users ask about similar studies or drug combinations, always search the database first. Provide detailed, well-formatted responses with specific study references and URLs."),
                HumanMessage(content=enhanced_prompt)
            ]
            
            # Generate response with RAG context
            response_text = ""
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content'):
                    response_text += chunk.content
            
            state["chat_response"] = response_text
            return state
        
        # Handle regular queries (summary generation or non-RAG questions)
        if not query or query == "generate_summary":
            # Generate initial summary using EXACT app_v1 prompt
            data_to_summarize = state["data_to_summarize"]
            
            # Filter sections with meaningful content - RELAXED FILTERING
            # Don't filter out sections that start with "No" - they might have valid data
            sections_to_include = {}
            for section, raw_content in data_to_summarize.items():
                # Extract content from dict format or use string directly
                if isinstance(raw_content, dict):
                    content = raw_content.get("content", "")
                else:
                    content = raw_content
                
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content) if content else ""
                
                if (content and 
                    content != "N/A" and 
                    content.strip() != "" and
                    len(content.strip()) > 30):  # Only check for minimum length
                    sections_to_include[section] = content
            
            # Debug: Print what sections are included
            print(f"📊 Sections with data for summary: {list(sections_to_include.keys())}")
            print(f"📊 Total content size: {sum(len(str(v)) for v in sections_to_include.values()):,} characters")
            
            # Prepare consolidated content
            consolidated_content = ""
            for section, content in sections_to_include.items():
                consolidated_content += f"\n\n**{section}:**\n{content}\n"
            
            # EXACT prompt from app_v1
            study_overview = data_to_summarize.get('Study Overview', '')
            # Handle both dict and string formats
            if isinstance(study_overview, dict):
                study_overview = study_overview.get('content', '')
            if isinstance(study_overview, str) and study_overview:
                study_title = study_overview.split('|')[0].strip()
            else:
                study_title = 'Clinical Trial Protocol'
            
            concise_prompt = f"""Generate a concise, well-formatted clinical trial summary using ONLY the information provided below. Follow this structure and format:

# Clinical Trial Summary
## {study_title}

### Study Overview
- Disease: [Extract disease information]
- Phase: [Extract phase information]
- Design: [Extract design information]
- Brief Description: [Extract brief description - 2-3 sentences max]


### Primary Objectives
[List main safety and/or efficacy endpoints - bullet points, be specific]

### Treatment Arms & Interventions
[Create a simple table if multiple arms exist, otherwise describe briefly]

### Eligibility Criteria
#### Key inclusion criteria
#### Key exclusion criteria

### Enrollment & Participant Flow
[Patient numbers and enrollment status if available]

### Safety Profile
[Only include if adverse events data is available - summarize key findings]

---

**Available Data:**
{consolidated_content}

**Formatting Requirements:**
- Start with just "Clinical Trial Summary" as the main heading (NCT ID will be in header)
- Use the study title as the secondary heading
- Use clear section headers (###)
- Keep each section to 1-3 sentences or a simple table
- Do not skip any key details if available; do not fabricate missing info; strictly summarize the content from the Protocol.
- Use bullet points for lists
- Only include sections where meaningful data exists
- Skip any section that says "not available" or has insufficient information
- Make it readable and concise - aim for 200-400 words total
- Use markdown formatting for better readability"""

            messages = [
                SystemMessage(content="You are a clinical research summarization expert. Create concise, well-formatted summaries that focus only on available information. Avoid filler text and sections with insufficient data. Use clear markdown formatting and keep summaries under 400 words while including all key available information."),
                HumanMessage(content=concise_prompt)
            ]
        else:
            # Follow-up question handling (same as app_v1)
            context = json.dumps(state["data_to_summarize"], indent=2)
            
            system_prompt = "You are a medical summarization assistant. Answer questions based on the provided protocol text. Do not invent information."
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Clinical Trial Data:\n{context}\n\nQuestion: {query}")
            ]
        
        # Stream response
        response_text = ""
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content'):
                response_text += chunk.content
        
        state["chat_response"] = response_text
        
    except Exception as e:
        state["error"] = f"Chat error: {str(e)}"
        state["chat_response"] = "Error processing query"
    
    return state


def chat_node_stream(state: WorkflowState) -> Iterator[str]:
    """Node: Handle chat with streaming (for Streamlit display)"""
    try:
        query = state.get("chat_query", "")
        
        # Check if we have RAG results to stream
        if state.get("use_rag_tool", False) and state.get("rag_tool_results"):
            # Create enhanced response using RAG results
            data_to_summarize = state.get("data_to_summarize", {})
            rag_results = state["rag_tool_results"]
            
            # Handle both RAG-only queries and queries with study context
            if data_to_summarize and any(data_to_summarize.values()):
                # We have current study data - compare with RAG results
                context = json.dumps(data_to_summarize, indent=2)
                enhanced_prompt = f"""You are a clinical research assistant. The user asked: "{query}"

You have access to two sources of information:

1. **Current Study Data:**
{context}

2. **Similar Clinical Trials from Database Search:**
{rag_results}

**Instructions:**
- Answer the user's question about similar studies using the database search results
- Focus on studies that use similar drug combinations or individual components
- Be specific about study NCT IDs, drug names, and provide clickable URLs
- Highlight key similarities and differences in treatments, phases, and outcomes
- Compare the found studies with the current study data when relevant
- Keep the response well-organized and informative
- If no similar studies were found, explain that and suggest alternative approaches

Please provide a comprehensive answer that addresses the user's question about similar studies."""
            else:
                # RAG-only query without current study context
                enhanced_prompt = f"""You are a clinical research assistant. The user asked: "{query}"

You have access to clinical trials database search results:

**Similar Clinical Trials from Database Search:**
{rag_results}

**Instructions:**
- Answer the user's question using the database search results
- Focus on studies that match the requested drug combinations or criteria
- Be specific about study NCT IDs, drug names, and provide clickable URLs
- Highlight key information about treatments, phases, and outcomes
- Keep the response well-organized and informative
- If no similar studies were found in the database, explain that clearly and suggest alternative approaches

Please provide a comprehensive answer that addresses the user's question about similar studies."""
            
            messages = [
                SystemMessage(content="You are a clinical research assistant expert at finding and explaining clinical trial similarities. When users ask about similar studies or drug combinations, always search the database first. Provide detailed, well-formatted responses with specific study references and URLs."),
                HumanMessage(content=enhanced_prompt)
            ]
            
            # Stream RAG response
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
            return
        
        # Handle regular queries (summary generation or non-RAG questions)
        if not query or query == "generate_summary":
            # Generate initial summary
            data_to_summarize = state["data_to_summarize"]
            
            # Filter sections
            sections_to_include = {}
            for section, content in data_to_summarize.items():
                if (content and 
                    content != "N/A" and 
                    isinstance(content, str) and
                    "No " not in content[:20] and
                    "not available" not in content.lower() and
                    len(content.strip()) > 30):
                    sections_to_include[section] = content
            
            consolidated_content = ""
            for section, content in sections_to_include.items():
                # Ensure content is a string before adding to consolidated content
                if isinstance(content, dict):
                    content = content.get('content', str(content))
                if not isinstance(content, str):
                    content = str(content) if content else ""
                consolidated_content += f"\n\n**{section}:**\n{content}\n"
            
            # Handle both dict and string formats for study title
            study_overview_raw = data_to_summarize.get('Study Overview', '')
            if isinstance(study_overview_raw, dict):
                study_overview_raw = study_overview_raw.get('content', '')
            # Ensure study_overview_raw is a string before calling split
            if not isinstance(study_overview_raw, str):
                study_overview_raw = str(study_overview_raw) if study_overview_raw else ''
            study_title = study_overview_raw.split('|')[0].strip() if study_overview_raw else 'Clinical Trial Protocol'
            
            concise_prompt = f"""Generate a concise, well-formatted clinical trial summary using ONLY the information provided below. Follow this structure and format:

# Clinical Trial Summary
## {study_title}

### Study Overview
- Disease: [Extract disease information]
- Phase: [Extract phase information]
- Design: [Extract design information]
- Brief Description: [Extract brief description - 2-3 sentences max]


### Primary Objectives
[List main safety and/or efficacy endpoints - bullet points, be specific]

### Treatment Arms & Interventions
[Create a simple table if multiple arms exist, otherwise describe briefly]

### Eligibility Criteria
#### Key inclusion criteria
#### Key exclusion criteria

### Enrollment & Participant Flow
[Patient numbers and enrollment status if available]

### Safety Profile
[Only include if adverse events data is available - summarize key findings]

---

**Available Data:**
{consolidated_content}

**Formatting Requirements:**
- Start with just "Clinical Trial Summary" as the main heading (NCT ID will be in header)
- Use the study title as the secondary heading
- Use clear section headers (###)
- Keep each section to 1-3 sentences or a simple table
- Do not skip any key details if available; do not fabricate missing info; strictly summarize the content from the Protocol.
- Use bullet points for lists
- Only include sections where meaningful data exists
- Skip any section that says "not available" or has insufficient information
- Make it readable and concise - aim for 200-400 words total
- Use markdown formatting for better readability"""

            messages = [
                SystemMessage(content="You are a clinical research summarization expert. Create concise, well-formatted summaries that focus only on available information. Avoid filler text and sections with insufficient data. Use clear markdown formatting and keep summaries under 400 words while including all key available information."),
                HumanMessage(content=concise_prompt)
            ]
        else:
            context = json.dumps(state["data_to_summarize"], indent=2)
            system_prompt = "You are a medical summarization assistant. Answer questions based on the provided protocol text. Do not invent information."
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Clinical Trial Data:\n{context}\n\nQuestion: {query}")
            ]
        
        # Stream chunks
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
                
    except Exception as e:
        yield f"Error: {str(e)}"


def convert_api_to_schema(study_data: dict) -> dict:
    """Convert ClinicalTrials.gov API data to standard schema (deprecated - using extract_from_url instead)"""
    # This function is kept for backward compatibility but not used in main workflow
    protocol = study_data.get('protocolSection', {})
    
    identification = protocol.get('identificationModule', {})
    description = protocol.get('descriptionModule', {})
    design = protocol.get('designModule', {})
    arms = protocol.get('armsInterventionsModule', {})
    eligibility = protocol.get('eligibilityModule', {})
    contacts = protocol.get('contactsLocationsModule', {})
    sponsor = protocol.get('sponsorCollaboratorsModule', {})
    outcomes = protocol.get('outcomesModule', {})
    
    return {
        "study_overview": identification.get('officialTitle', ''),
        "brief_description": description.get('briefSummary', ''),
        "primary_secondary_objectives": json.dumps({
            'primary': outcomes.get('primaryOutcomes', []),
            'secondary': outcomes.get('secondaryOutcomes', [])
        }),
        "treatment_arms_interventions": json.dumps({
            'arms': arms.get('armGroups', []),
            'interventions': arms.get('interventions', [])
        }),
        "eligibility_criteria": eligibility.get('eligibilityCriteria', ''),
        "enrollment_participant_flow": json.dumps(design.get('enrollmentInfo', {})),
        "adverse_events_profile": "Not available in protocol section",
        "study_locations": json.dumps(contacts.get('locations', [])),
        "sponsor_information": json.dumps(sponsor)
    }


# Build the graph
def build_workflow() -> StateGraph:
    """Build the LangGraph workflow"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("classify_input", classify_input)
    workflow.add_node("pdf_parser", parse_pdf)
    workflow.add_node("url_extractor", extract_from_url)
    workflow.add_node("llm_fallback", llm_fallback)
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("route_decision", route_decision_node)
    
    # Add edges
    workflow.set_entry_point("classify_input")
    
    workflow.add_conditional_edges(
        "classify_input",
        route_input,
        {
            "pdf_parser": "pdf_parser",
            "url_extractor": "url_extractor",
            "route_decision": "route_decision",  # New path for RAG-only and follow-up queries
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "pdf_parser",
        check_quality,
        {
            "llm_fallback": "llm_fallback",
            "chat_node": "route_decision"
        }
    )
    
    workflow.add_conditional_edges(
        "url_extractor",
        check_quality,
        {
            "llm_fallback": "llm_fallback",
            "chat_node": "route_decision"
        }
    )
    
    # Route from llm_fallback to decision point
    workflow.add_edge("llm_fallback", "route_decision")
    
    # Route decision point decides between RAG search and direct chat
    workflow.add_conditional_edges(
        "route_decision",
        route_to_rag_or_chat,
        {
            "rag_search": "rag_search",
            "chat_node": "chat_node"
        }
    )
    
    # RAG search always goes to chat after completing search
    workflow.add_edge("rag_search", "chat_node")
    
    workflow.add_edge("chat_node", END)
    
    return workflow.compile()


def get_shared_llm_instance():
    """Get the shared non-streaming LLM instance for re-extraction"""
    return llm_non_streaming

def get_workflow():
    """Get the workflow app instance"""
    return build_workflow()


# Main execution
if __name__ == "__main__":
    app = build_workflow()
    
    # Example usage
    result = app.invoke({
        "input_url": "https://clinicaltrials.gov/study/NCT03991871",
        "input_type": "unknown",
        "raw_data": {},
        "parsed_json": {},
        "data_to_summarize": {},
        "confidence_score": 0.0,
        "completeness_score": 0.0,
        "missing_fields": [],
        "nct_id": "",
        "chat_query": "generate_summary",
        "chat_response": "",
        "stream_response": None,
        "error": "",
        "use_rag_tool": False,
        "rag_tool_results": ""
    })
    
    print(json.dumps(result, indent=2, default=str))
