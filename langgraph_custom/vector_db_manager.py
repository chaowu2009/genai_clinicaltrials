"""
Vector Database Manager for Clinical Trials RAG
Uses ChromaDB for vector storage and retrieval of clinical trials data
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism to avoid conflicts
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)


class ClinicalTrialsVectorDB:
    """
    Vector database manager for clinical trials data using ChromaDB
    """
    
    def __init__(
        self,
        db_path: str = "db/clinical_trials_vectordb",
        collection_name: str = "cancer_clinical_trials",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with proper settings to avoid threading issues
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    allow_reset=True, 
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
        except Exception as e:
            logger.warning(f"Failed to create persistent client: {e}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
        
        # Initialize embedding function
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SentenceTransformer: {e}")
            # Fallback to default embedding
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection '{collection_name}'")
        except Exception as e:
            logger.info(f"Creating new collection '{collection_name}': {e}")
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Cancer clinical trials for RAG"}
                )
                logger.info(f"Created new collection '{collection_name}'")
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
                raise
    
    def _create_document_text(self, study_data: Dict[str, Any]) -> str:
        """
        Create searchable document text from clinical trial data
        """
        parsed_json = study_data.get("parsed_json", {})
        metadata = study_data.get("metadata", {})
        
        # Build comprehensive document text for embedding
        doc_parts = []
        
        # Study overview
        if parsed_json.get("study_overview"):
            doc_parts.append(f"STUDY: {parsed_json['study_overview']}")
        
        # Brief description
        if parsed_json.get("brief_description"):
            doc_parts.append(f"DESCRIPTION: {parsed_json['brief_description']}")
        
        # Interventions/drugs
        if parsed_json.get("treatment_arms_interventions"):
            doc_parts.append(f"TREATMENTS: {parsed_json['treatment_arms_interventions']}")
        
        # Objectives
        if parsed_json.get("primary_secondary_objectives"):
            doc_parts.append(f"OBJECTIVES: {parsed_json['primary_secondary_objectives']}")
        
        # Eligibility
        if parsed_json.get("eligibility_criteria"):
            eligibility = parsed_json["eligibility_criteria"]
            # Truncate if too long
            if len(eligibility) > 1000:
                eligibility = eligibility[:1000] + "..."
            doc_parts.append(f"ELIGIBILITY: {eligibility}")
        
        # Conditions and interventions from metadata
        if metadata.get("conditions"):
            doc_parts.append(f"CONDITIONS: {', '.join(metadata['conditions'])}")
        
        if metadata.get("interventions"):
            doc_parts.append(f"DRUGS: {', '.join(metadata['interventions'])}")
        
        # Keywords
        if metadata.get("keywords"):
            doc_parts.append(f"KEYWORDS: {', '.join(metadata['keywords'][:20])}")  # Limit keywords
        
        return "\\n\\n".join(doc_parts)
    
    def _create_metadata(self, study_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for ChromaDB storage
        """
        nct_id = study_data.get("nct_id", "")
        metadata_dict = study_data.get("metadata", {})
        parsed_json = study_data.get("parsed_json", {})
        
        # ChromaDB metadata (must be JSON serializable)
        chroma_metadata = {
            "nct_id": nct_id,
            "title": metadata_dict.get("title", "")[:500],  # Truncate long titles
            "brief_title": metadata_dict.get("brief_title", "")[:200],
            "status": metadata_dict.get("status", ""),
            "study_type": metadata_dict.get("study_type", ""),
            "phases": ",".join(metadata_dict.get("phases", [])),
            "conditions": ",".join(metadata_dict.get("conditions", []))[:300],
            "interventions": ",".join(metadata_dict.get("interventions", []))[:300],
            "sponsor": metadata_dict.get("sponsor", "")[:200],
            "enrollment_count": str(metadata_dict.get("enrollment_count", "")),
            "study_url": study_data.get("study_url", ""),
            "api_url": study_data.get("api_url", ""),
            # Store first few keywords
            "keywords": ",".join(metadata_dict.get("keywords", [])[:10])[:300]
        }
        
        # Remove empty values
        chroma_metadata = {k: v for k, v in chroma_metadata.items() if v}
        
        return chroma_metadata
    
    def add_studies(self, studies_data: List[Dict[str, Any]]) -> int:
        """
        Add clinical trials studies to the vector database
        
        Args:
            studies_data: List of study data from clinical_trials_downloader
            
        Returns:
            Number of studies successfully added
        """
        if not studies_data:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        
        for study_data in studies_data:
            try:
                nct_id = study_data.get("nct_id")
                if not nct_id:
                    continue
                
                # Create document text
                doc_text = self._create_document_text(study_data)
                if not doc_text.strip():
                    logger.warning(f"Empty document text for {nct_id}")
                    continue
                
                # Create metadata
                metadata = self._create_metadata(study_data)
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(nct_id)
                
            except Exception as e:
                logger.error(f"Error processing study {study_data.get('nct_id', 'unknown')}: {e}")
                continue
        
        if not documents:
            logger.warning("No valid documents to add")
            return 0
        
        try:
            # Check for existing documents and remove duplicates
            existing_ids = set()
            try:
                existing_docs = self.collection.get()
                existing_ids = set(existing_docs['ids'])
            except:
                pass  # Empty collection
            
            # Filter out existing documents
            new_documents = []
            new_metadatas = []
            new_ids = []
            
            for doc, metadata, doc_id in zip(documents, metadatas, ids):
                if doc_id not in existing_ids:
                    new_documents.append(doc)
                    new_metadatas.append(metadata)
                    new_ids.append(doc_id)
            
            if not new_documents:
                logger.info("All documents already exist in the database")
                return 0
            
            # Add to ChromaDB
            self.collection.add(
                documents=new_documents,
                metadatas=new_metadatas,
                ids=new_ids
            )
            
            logger.info(f"Added {len(new_documents)} new studies to vector database")
            return len(new_documents)
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def search_similar_studies(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar clinical trials studies
        
        Args:
            query: Search query (drug name, condition, etc.)
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar studies with metadata and similarity scores
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ["status", "study_type", "phases"]:
                        where_clause[key] = value
                    elif key == "conditions" and value:
                        # For conditions, we'll handle this in post-processing
                        # since ChromaDB doesn't support complex text searching in metadata
                        pass
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 100),  # Limit to prevent memory issues
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            
            # Format results
            formatted_results = []
            
            for i in range(len(results['ids'][0])):
                study_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                # Calculate similarity score (1 - distance for cosine similarity)
                similarity_score = max(0, 1 - distance)
                
                formatted_result = {
                    "nct_id": study_id,
                    "similarity_score": round(similarity_score, 3),
                    "title": metadata.get("title", ""),
                    "brief_title": metadata.get("brief_title", ""),
                    "status": metadata.get("status", ""),
                    "study_type": metadata.get("study_type", ""),
                    "phases": metadata.get("phases", ""),
                    "conditions": metadata.get("conditions", ""),
                    "interventions": metadata.get("interventions", ""),
                    "sponsor": metadata.get("sponsor", ""),
                    "study_url": metadata.get("study_url", f"https://clinicaltrials.gov/study/{study_id}"),
                    "document_text": document[:500] + "..." if len(document) > 500 else document
                }
                
                # Apply additional filters
                if filters and "conditions" in filters and filters["conditions"]:
                    condition_filter = filters["conditions"].lower()
                    study_conditions = metadata.get("conditions", "").lower()
                    if condition_filter not in study_conditions:
                        continue
                
                formatted_results.append(formatted_result)
            
            # Sort by similarity score
            formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def search_by_query(
        self,
        query: str,
        n_results: int = 10,
        conditions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for clinical trials by general query
        
        Args:
            query: General search query (drug names, conditions, treatments, etc.)
            n_results: Number of results to return
            conditions: Optional list of conditions to filter by
            
        Returns:
            List of similar studies
        """
        filters = {}
        if conditions:
            filters["conditions"] = " ".join(conditions).lower()
        
        return self.search_similar_studies(query, n_results, filters)
    
    def search_by_drug(
        self,
        drug_name: str,
        n_results: int = 10,
        conditions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for clinical trials by drug name
        
        Args:
            drug_name: Name of the drug/intervention
            n_results: Number of results to return
            conditions: Optional list of conditions to filter by
            
        Returns:
            List of similar studies
        """
        # Enhance query with drug-related terms
        query = f"drug intervention treatment {drug_name} therapy medication"
        
        filters = {}
        if conditions:
            # We'll handle condition filtering in the main search
            filters["conditions"] = " ".join(conditions).lower()
        
        return self.search_similar_studies(query, n_results, filters)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection
        """
        try:
            collection_data = self.collection.get()
            
            total_studies = len(collection_data['ids'])
            
            # Count by status
            status_counts = {}
            study_type_counts = {}
            phase_counts = {}
            
            for metadata in collection_data.get('metadatas', []):
                # Status counts
                status = metadata.get('status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Study type counts  
                study_type = metadata.get('study_type', 'Unknown')
                study_type_counts[study_type] = study_type_counts.get(study_type, 0) + 1
                
                # Phase counts
                phases = metadata.get('phases', '')
                if phases:
                    for phase in phases.split(','):
                        phase = phase.strip()
                        if phase:
                            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            return {
                "total_studies": total_studies,
                "status_distribution": status_counts,
                "study_type_distribution": study_type_counts,
                "phase_distribution": phase_counts,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def load_from_json(self, json_file_path: str) -> int:
        """
        Load clinical trials data from JSON file (from clinical_trials_downloader)
        
        Args:
            json_file_path: Path to the JSON file with clinical trials data
            
        Returns:
            Number of studies loaded
        """
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                studies_data = json.load(f)
            
            if not isinstance(studies_data, list):
                raise ValueError("JSON file should contain a list of studies")
            
            logger.info(f"Loaded {len(studies_data)} studies from {json_file_path}")
            
            return self.add_studies(studies_data)
            
        except Exception as e:
            logger.error(f"Error loading from JSON file: {e}")
            raise
    
    def reset_collection(self):
        """
        Reset/clear the collection (useful for testing)
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Cancer clinical trials for RAG"}
            )
            logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise


def main():
    """Example usage"""
    import logging
    # Only configure basic logging for standalone execution
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    
    # Initialize vector database
    vector_db = ClinicalTrialsVectorDB()
    
    # Load data from JSON file (if it exists)
    json_file = Path("data/cancer_clinical_trials.json")
    if json_file.exists():
        print("📚 Loading clinical trials data into vector database...")
        count = vector_db.load_from_json(str(json_file))
        print(f"✅ Loaded {count} studies into vector database")
        
        # Get stats
        stats = vector_db.get_collection_stats()
        print(f"📊 Collection stats: {stats}")
        
        # Test search
        print("🔍 Testing drug search...")
        results = vector_db.search_by_drug("pembrolizumab", n_results=5)
        print(f"Found {len(results)} similar studies for pembrolizumab:")
        for result in results[:3]:
            print(f"  - {result['nct_id']}: {result['brief_title']} (similarity: {result['similarity_score']})")
    else:
        print("ℹ️  No clinical trials JSON file found. Run clinical_trials_downloader.py first.")


if __name__ == "__main__":
    main()