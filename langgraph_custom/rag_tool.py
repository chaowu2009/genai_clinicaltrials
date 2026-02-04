"""
RAG Tool for Clinical Trials Search
LangGraph tool for searching similar clinical trials using vector database
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import json
import logging

from .vector_db_manager import ClinicalTrialsVectorDB

logger = logging.getLogger(__name__)


class ClinicalTrialSearchInput(BaseModel):
    """Input schema for clinical trial search tool"""
    search_query: str = Field(
        description="Search query for finding similar clinical trials - can include drug names, conditions, treatment types, or any combination"
    )
    n_results: int = Field(
        default=5,
        description="Number of similar studies to return (1-20)",
        ge=1,
        le=20
    )
    conditions: Optional[List[str]] = Field(
        default=None,
        description="Optional list of medical conditions to filter by (e.g., ['lung cancer', 'breast cancer', 'HIV'])"
    )


class ClinicalTrialSearchTool(BaseTool):
    """
    Tool for searching similar clinical trials by drug/intervention name
    """
    name: str = "search_similar_clinical_trials"
    description: str = """
    Search for similar clinical trials based on a flexible search query.
    This tool finds clinical trials that match the search criteria and returns
    relevant studies with their NCT IDs, titles, and ClinicalTrials.gov URLs.
    
    Use this tool when users ask about:
    - Similar studies using specific drugs or drug combinations
    - Other trials for particular conditions or treatments
    - Comparable treatments or interventions
    - Related clinical research across different therapeutic areas
    
    The tool accepts flexible search queries including:
    - Drug names (brand names, generic names, abbreviations)
    - Drug combinations (e.g., "DTG + TAF + FTC")
    - Treatment modalities (e.g., "immunotherapy", "chemotherapy")
    - Medical conditions (e.g., "HIV", "lung cancer")
    - Any combination of the above
    
    The tool returns structured information about each study including:
    - NCT ID and official study URL
    - Study title and brief description
    - Treatment arms and interventions
    - Study status and phase
    - Similarity score
    """
    args_schema: type[BaseModel] = ClinicalTrialSearchInput
    
    # Pydantic fields - must be defined as class attributes
    db_path: str = Field(default="db/clinical_trials_vectordb", description="Path to vector database")
    vector_db: Any = Field(default=None, description="Vector database instance", exclude=True)
    
    def __init__(self, db_path: str = "db/clinical_trials_vectordb", **kwargs):
        # Initialize Pydantic fields first
        super().__init__(db_path=db_path, vector_db=None, **kwargs)
        
        # Then initialize the vector database
        try:
            # Use object.__setattr__ to bypass Pydantic validation for internal field
            object.__setattr__(self, 'vector_db', ClinicalTrialsVectorDB(db_path=db_path))
            logger.info(f"Initialized ClinicalTrialSearchTool with database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            object.__setattr__(self, 'vector_db', None)
    
    def _run(
        self,
        search_query: str,
        n_results: int = 5,
        conditions: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Search for similar clinical trials using flexible search query
        """
        if not self.vector_db:
            return "Error: Clinical trials database not available. Please ensure the vector database is properly initialized."
        
        try:
            # Validate inputs
            if not search_query or not search_query.strip():
                return "Error: Please provide a valid search query."
            
            search_query = search_query.strip()
            n_results = max(1, min(n_results, 20))  # Clamp between 1 and 20
            
            logger.info(f"Searching for clinical trials with query: {search_query}")
            
            # Try to use search_by_query if available, fallback to search_by_drug for compatibility
            if hasattr(self.vector_db, 'search_by_query'):
                results = self.vector_db.search_by_query(
                    query=search_query,
                    n_results=n_results,
                    conditions=conditions
                )
            else:
                # Fallback to existing search_by_drug method
                results = self.vector_db.search_by_drug(
                    drug_name=search_query,
                    n_results=n_results,
                    conditions=conditions
                )
            
            if not results:
                return f"No similar clinical trials found for '{search_query}'. The database may not contain studies matching this search criteria."
            
            # Format results for LangGraph response
            response_parts = [
                f"Found {len(results)} similar clinical trials for '{search_query}':\\n"
            ]
            
            for i, study in enumerate(results, 1):
                nct_id = study['nct_id']
                title = study.get('brief_title') or study.get('title', 'No title available')
                similarity = study['similarity_score']
                status = study.get('status', 'Unknown')
                phases = study.get('phases', 'Unknown')
                interventions = study.get('interventions', 'Not specified')
                study_url = study['study_url']
                
                # Truncate long titles
                if len(title) > 100:
                    title = title[:97] + "..."
                
                study_info = [
                    f"{i}. **{nct_id}** - {title}",
                    f"   Similarity: {similarity:.2%}",
                    f"   Status: {status}",
                    f"   Phase: {phases}" if phases != 'Unknown' else "",
                    f"   Interventions: {interventions}" if len(interventions) < 100 else f"   Interventions: {interventions[:97]}...",
                    f"   Study URL: {study_url}",
                    ""
                ]
                
                response_parts.extend([part for part in study_info if part])
            
            # Add summary information
            if conditions:
                response_parts.append(f"Search was filtered for conditions: {', '.join(conditions)}")
            
            response_parts.extend([
                "",
                "**How to use this information:**",
                "- Click on the Study URL links to view full details on ClinicalTrials.gov",
                "- Higher similarity scores indicate more relevant studies",
                "- Check the study status to see if trials are currently recruiting",
                "- Review interventions to understand treatment approaches"
            ])
            
            return "\\n".join(response_parts)
            
        except Exception as e:
            error_msg = f"Error searching clinical trials: {str(e)}"
            logger.error(error_msg)
            return f"Error: Failed to search clinical trials database. {error_msg}"
    
    async def _arun(
        self,
        search_query: str,
        n_results: int = 5,
        conditions: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of the tool"""
        return self._run(search_query, n_results, conditions, run_manager)


def create_clinical_trials_rag_tool(db_path: str = "db/clinical_trials_vectordb") -> ClinicalTrialSearchTool:
    """
    Factory function to create the clinical trials RAG tool
    
    Args:
        db_path: Path to the vector database
        
    Returns:
        Configured ClinicalTrialSearchTool
    """
    return ClinicalTrialSearchTool(db_path=db_path)


# Additional helper functions for the tool

def format_study_summary(study: Dict[str, Any]) -> str:
    """
    Format a single study for display
    """
    nct_id = study['nct_id']
    title = study.get('brief_title') or study.get('title', 'No title')
    similarity = study['similarity_score']
    status = study.get('status', 'Unknown')
    phases = study.get('phases', 'Unknown')
    study_url = study['study_url']
    
    return f"""
**{nct_id}**: {title}
- Similarity: {similarity:.2%}
- Status: {status}
- Phase: {phases}
- URL: {study_url}
"""


def validate_drug_name(drug_name: str) -> bool:
    """
    Validate drug name input
    """
    if not drug_name or not drug_name.strip():
        return False
    
    # Check for minimum length
    if len(drug_name.strip()) < 2:
        return False
    
    return True


def get_tool_description() -> str:
    """
    Get detailed description of the tool for documentation
    """
    return """
    Clinical Trials RAG Tool
    
    This tool searches a vector database of cancer clinical trials to find studies
    that use similar drugs or interventions to the one specified by the user.
    
    Features:
    - Vector similarity search using embeddings
    - Filters by cancer conditions (optional)
    - Returns study metadata including NCT IDs and URLs
    - Similarity scoring for relevance ranking
    
    Use cases:
    - "Find studies similar to pembrolizumab"
    - "What other trials use checkpoint inhibitors?"
    - "Show me breast cancer studies with immunotherapy"
    
    Returns:
    - NCT ID and ClinicalTrials.gov URL
    - Study title and status
    - Treatment details and phase
    - Similarity score for ranking
    """


# Tool testing and validation functions

def test_tool_functionality(db_path: str = "db/clinical_trials_vectordb"):
    """
    Test the clinical trials RAG tool functionality
    """
    try:
        tool = create_clinical_trials_rag_tool(db_path)
        
        # Test basic search
        result = tool._run("pembrolizumab", n_results=3)
        print("Test Result:")
        print(result)
        print("\\n" + "="*50 + "\\n")
        
        return True
        
    except Exception as e:
        print(f"Tool test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the tool
    # Only configure basic logging when run as script
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Clinical Trials RAG Tool...")
    success = test_tool_functionality()
    
    if success:
        print("✅ Tool test completed successfully!")
    else:
        print("❌ Tool test failed!")