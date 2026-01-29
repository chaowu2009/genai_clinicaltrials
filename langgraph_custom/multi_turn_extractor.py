"""
Multi-Turn PDF Extraction Strategy
====================================

Splits large PDFs into smaller chunks and processes them across multiple LLM calls
to avoid rate limiting while maintaining context and quality.

Key Features:
- Field-based chunking: Extract one field at a time
- Smart context windows: Include relevant surrounding text
- Rate limit management: Automatic delays and retries
- Progress tracking: Real-time updates for UI
- Result merging: Intelligent combination of partial results

Author: Clinical Trials AI Team
Version: 1.0.0
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_custom.extraction_schemas import (
    ExtractionResult,
    FIELD_CONFIGS_WITH_SCHEMA,
    get_extraction_result_schema_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    field_name: str
    start_char: int
    end_char: int
    estimated_tokens: int
    context_before: str
    context_after: str


class MultiTurnExtractor:
    """
    Extracts clinical trial data using multiple LLM calls to avoid rate limits.
    
    Strategy:
    1. Split document into field-specific chunks
    2. Process each chunk with its own LLM call
    3. Merge results intelligently
    4. Handle rate limits with automatic retries
    """
    
    # Field definitions with extraction hints (now sourced from extraction_schemas.py)
    FIELD_CONFIGS = {field: config for field, config in FIELD_CONFIGS_WITH_SCHEMA.items()}
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens_per_call: int = 180000,  # Leave buffer under 200k limit
        delay_between_calls: float = 2.0,  # Seconds
        llm_instance: Optional[ChatOpenAI] = None  # Accept existing LLM instance
    ):
        """Initialize the multi-turn extractor with shared LLM configuration."""
        
        # Use provided LLM instance or create a new simple one (no structured outputs)
        if llm_instance:
            self.llm = llm_instance
            logger.info("Using provided LLM instance for re-extraction")
        else:
            # Create simple LLM instance like the main workflow (no json_schema)
            self.llm = ChatOpenAI(
                model=model, 
                temperature=temperature,
                streaming=False,  # Disable streaming for re-extraction
                request_timeout=300,  # 5 minutes for large chunks (40k+ tokens)
                timeout=300  # Connection timeout
            )
            logger.info(f"Created new LLM instance for re-extraction: {model}")
            
        self.max_tokens_per_call = max_tokens_per_call
        self.delay_between_calls = delay_between_calls
        self.encoding = tiktoken.encoding_for_model(model)
        
    def extract_page_numbers_from_text(self, text: str, content: str) -> List[int]:
        """
        Extract page numbers where specific content appears in the text.
        
        Args:
            text: Full text with page markers
            content: Specific content to find pages for
            
        Returns:
            List of page numbers where content appears
        """
        import re
        
        pages = []
        current_page = 1
        
        # Split text by page markers and track content
        page_pattern = r'--- Page (\d+) ---'
        text_parts = re.split(page_pattern, text)
        
        # Process each part
        for i in range(0, len(text_parts)):
            if i % 2 == 0:  # Text content
                text_content = text_parts[i]
                # Check if any part of the extracted content appears in this page's text
                content_words = content.lower().split()[:10]  # Check first 10 words
                text_lower = text_content.lower()
                
                matches = sum(1 for word in content_words if len(word) > 3 and word in text_lower)
                if matches >= 3:  # If at least 3 significant words match
                    pages.append(current_page)
            else:  # Page number
                try:
                    current_page = int(text_parts[i])
                except ValueError:
                    continue
        
        return sorted(list(set(pages)))
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _safe_content_to_string(self, content: Any) -> str:
        """
        Safely convert any content to a string.
        
        Args:
            content: Content that might be string, dict, list, or other type
            
        Returns:
            String representation of the content
        """
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, dict):
            # If it's a dict, convert to formatted JSON
            return json.dumps(content, indent=2)
        elif isinstance(content, list):
            # If it's a list, join items or convert to JSON
            if all(isinstance(item, str) for item in content):
                return "\n".join(content)
            else:
                return json.dumps(content, indent=2)
        else:
            # For any other type, convert to string
            result = str(content) if content is not None else ""
            return result.strip() if isinstance(result, str) else result
    
    def analyze_document_chunking(self, full_text: str) -> Dict[str, Any]:
        """
        Analyze how the document would be chunked for each field.
        Returns statistics without actually performing extraction.
        
        Returns:
            Dictionary with chunking analysis for each field
        """
        analysis = {
            "total_document_chars": len(full_text),
            "total_document_tokens": self.count_tokens(full_text),
            "fields": {}
        }
        
        for field_name, config in self.FIELD_CONFIGS.items():
            relevant_sections = self.find_relevant_sections(full_text, field_name, config)
            
            # Calculate how many sections would fit
            sections_that_fit = 0
            total_tokens_used = 0
            max_tokens = config["max_tokens"]
            
            for start, end, score in relevant_sections:
                section_text = full_text[start:end]
                section_tokens = self.count_tokens(section_text)
                
                if total_tokens_used + section_tokens <= max_tokens:
                    sections_that_fit += 1
                    total_tokens_used += section_tokens
                else:
                    break
            
            analysis["fields"][field_name] = {
                "keywords": config["keywords"],
                "max_tokens_allowed": max_tokens,
                "sections_found": len(relevant_sections),
                "sections_that_fit": sections_that_fit,
                "tokens_used": total_tokens_used,
                "utilization_pct": (total_tokens_used / max_tokens) * 100 if max_tokens > 0 else 0,
                "top_relevance_scores": [s[2] for s in relevant_sections[:5]] if relevant_sections else []
            }
        
        return analysis
    
    def print_chunking_analysis(self, full_text: str):
        """Print a readable analysis of document chunking."""
        analysis = self.analyze_document_chunking(full_text)
        
        print("\n" + "="*80)
        print("📊 DOCUMENT CHUNKING ANALYSIS")
        print("="*80)
        print(f"📄 Total Document: {analysis['total_document_chars']:,} chars, {analysis['total_document_tokens']:,} tokens")
        print()
        
        for field_name, stats in analysis["fields"].items():
            print(f"\n🔹 {field_name.upper().replace('_', ' ')}")
            print(f"   Keywords: {', '.join(stats['keywords'])}")
            print(f"   Max tokens allowed: {stats['max_tokens_allowed']:,}")
            print(f"   Sections found: {stats['sections_found']}")
            print(f"   Sections that fit: {stats['sections_that_fit']}")
            print(f"   Tokens used: {stats['tokens_used']:,} ({stats['utilization_pct']:.1f}% utilization)")
            if stats['top_relevance_scores']:
                print(f"   Top relevance scores: {stats['top_relevance_scores']}")
        
        print("\n" + "="*80 + "\n")

    
    def find_relevant_sections(
        self,
        full_text: str,
        field_name: str,
        config: Dict[str, Any]
    ) -> List[Tuple[int, int, float]]:
        """
        Find sections of text relevant to a specific field.
        
        Returns:
            List of (start_pos, end_pos, relevance_score) tuples
        """
        # Ensure full_text is a string
        if isinstance(full_text, dict):
            full_text = full_text.get('content', '') or json.dumps(full_text)
        if not isinstance(full_text, str):
            full_text = str(full_text) if full_text else ''
        
        keywords = config["keywords"]
        sections = []
        
        # Split text into paragraphs
        paragraphs = full_text.split('\n\n')
        current_pos = 0
        
        logger.info(f"🔍 Searching for '{field_name}' using keywords: {keywords}")
        logger.info(f"📄 Total paragraphs to scan: {len(paragraphs)}")
        
        for para in paragraphs:
            para_lower = para.lower()
            
            # Calculate relevance score
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in para_lower:
                    count = para_lower.count(keyword.lower())
                    score += count
                    matched_keywords.append(f"{keyword}({count})")
            
            if score > 0:
                start = current_pos
                end = current_pos + len(para)
                sections.append((start, end, score))
                # Log first few high-scoring sections for debugging
                if score >= 3:
                    preview = para[:100].replace('\n', ' ')
                    logger.debug(f"  ✓ Found section (score={score}): {preview}... [keywords: {', '.join(matched_keywords)}]")
            
            current_pos += len(para) + 2  # +2 for \n\n
        
        # Sort by relevance score (descending)
        sections.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"✅ Found {len(sections)} relevant sections for '{field_name}'")
        if sections:
            top_5_scores = [s[2] for s in sections[:5]]
            logger.info(f"   Top 5 relevance scores: {top_5_scores}")
        
        return sections
    
    def create_field_chunk(
        self,
        full_text: str,
        field_name: str,
        max_tokens: int
    ) -> str:
        """
        Create a focused chunk for extracting a specific field.
        
        Strategy:
        1. Find sections with relevant keywords
        2. Include high-relevance sections first
        3. Add context until token limit reached
        """
        # Ensure full_text is a string
        if isinstance(full_text, dict):
            full_text = full_text.get('content', '') or json.dumps(full_text)
        if not isinstance(full_text, str):
            full_text = str(full_text) if full_text else ''
        
        config = self.FIELD_CONFIGS[field_name]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 Creating chunk for: {field_name}")
        logger.info(f"🎯 Max tokens allowed: {max_tokens:,}")
        
        # Find relevant sections
        relevant_sections = self.find_relevant_sections(full_text, field_name, config)
        
        if not relevant_sections:
            # If no keywords found, use first N tokens
            logger.warning(f"⚠️  No relevant sections found for {field_name}, using beginning of document")
            chunk_text = full_text
        else:
            # Combine relevant sections
            chunk_parts = []
            current_tokens = 0
            sections_included = 0
            
            logger.info(f"📊 Combining sections (highest relevance first)...")
            
            for start, end, score in relevant_sections:
                section_text = full_text[start:end]
                section_tokens = self.count_tokens(section_text)
                
                if current_tokens + section_tokens > max_tokens:
                    logger.info(f"   ⛔ Stopped at section {sections_included + 1}: would exceed token limit")
                    logger.info(f"      (needed {section_tokens:,} more tokens, only {max_tokens - current_tokens:,} available)")
                    break
                
                chunk_parts.append(section_text)
                current_tokens += section_tokens
                sections_included += 1
                
                # Log first few sections being included
                if sections_included <= 5:
                    preview = section_text[:80].replace('\n', ' ')
                    logger.info(f"   ✓ Section {sections_included} (score={score}, tokens={section_tokens:,}): {preview}...")
            
            chunk_text = "\n\n".join(chunk_parts)
            
            logger.info(f"✅ Combined {sections_included} sections into chunk")
            logger.info(f"📏 Initial chunk size: {current_tokens:,} tokens ({len(chunk_text):,} chars)")
        
        # Truncate if still too long
        truncation_count = 0
        while self.count_tokens(chunk_text) > max_tokens:
            # Remove last 10% of text
            chunk_text = chunk_text[:int(len(chunk_text) * 0.9)]
            truncation_count += 1
        
        if truncation_count > 0:
            logger.warning(f"⚠️  Truncated chunk {truncation_count} times to fit token limit")
        
        final_tokens = self.count_tokens(chunk_text)
        logger.info(f"📦 Final chunk: {final_tokens:,} tokens ({len(chunk_text):,} chars)")
        logger.info(f"   Utilization: {(final_tokens/max_tokens)*100:.1f}% of allowed tokens")
        logger.info(f"{'='*60}\n")
        
        return chunk_text
    
    def extract_field(
        self,
        chunk_text: str,
        field_name: str,
        field_description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract a single field using LLM with enhanced page references and exact text preservation.
        
        Args:
            chunk_text: Focused text chunk for this field
            field_name: Name of the field to extract
            field_description: Description of what to extract
            
        Returns:
            Dictionary with 'content' and 'page_references' or None
        """
        system_prompt = f"""You are an expert clinical trial document analyst extracting information from a clinical trial PDF.

You MUST extract: **{field_name}**
What to look for: {field_description}

IMPORTANT INSTRUCTIONS:
1. **Extract ALL relevant information** you find in the document related to this field
2. Copy text VERBATIM - preserve exact wording, numbers, formatting, bullet points
3. Include ALL details: numbers, percentages, timeframes, dosages, criteria, names
4. Combine information from multiple sections if needed
5. Look for page markers like "--- Page X ---" to identify source pages
6. Return comprehensive content - be thorough, not minimal

FOR STRUCTURED DATA:
- Eligibility: Include BOTH inclusion AND exclusion criteria in full
- Interventions: Include drug names, dosages, schedules, routes, duration
- Objectives: Include PRIMARY and SECONDARY endpoints with definitions
- Adverse Events: Include event names, frequencies, grades, percentages

OUTPUT REQUIREMENTS:
- Return a JSON object with exactly two fields: "content" and "page_references"
- "content" MUST be a single string containing all extracted information
- "page_references" MUST be an array of integers (page numbers where info was found)
- Extract page numbers from "--- Page X ---" markers in the text

ONLY return {{"content": "Not found in provided text", "page_references": []}} if you genuinely cannot find ANY relevant information in the document.

Otherwise, extract everything you can find and return it in the content field."""

        user_prompt = f"""Extract **{field_name}** from the clinical trial document below.

{field_description}

Document text:
{chunk_text}

---

Extract ALL relevant content for {field_name}. Include complete details and preserve exact wording.
Return JSON with "content" (string with extracted info) and "page_references" (array of page numbers):"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            logger.info(f"🔄 Calling LLM for {field_name} (chunk: {len(chunk_text)} chars)...")
            start_time = time.time()
            response = self.llm.invoke(messages)
            elapsed = time.time() - start_time
            logger.info(f"✅ LLM response received for {field_name} (took {elapsed:.1f}s)")
            # Safely handle response.content which could be str, dict, list, or other types
            raw_content = response.content
            
            # Convert raw_content to string safely
            if isinstance(raw_content, str):
                content = raw_content.strip()
            elif isinstance(raw_content, dict):
                content = json.dumps(raw_content)
            elif isinstance(raw_content, list):
                # Handle list of content blocks (some models return this)
                text_parts = []
                for item in raw_content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        # Extract text from dict (e.g., {"type": "text", "text": "..."})
                        if "text" in item:
                            text_parts.append(item["text"])
                        else:
                            text_parts.append(json.dumps(item))
                    else:
                        text_parts.append(str(item))
                content = "".join(text_parts).strip()
            else:
                # Fallback for any other type
                content = str(raw_content).strip() if raw_content else ""
            
            # Clean up response - remove markdown formatting
            if content.startswith("```"):
                # Remove code block markers
                lines = content.split("\n")
                start_idx = 0
                end_idx = len(lines)
                
                # Find start of JSON (skip ```json or just ```)
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        start_idx = i
                        break
                
                # Find end of JSON (before closing ```)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().endswith("}"):
                        end_idx = i + 1
                        break
                
                content = "\n".join(lines[start_idx:end_idx])
            
            # Try to parse as JSON
            try:
                result = json.loads(content)
                
                # Validate structure
                if "content" in result:
                    extracted_content = result["content"]
                    
                    # Use safe conversion to string
                    extracted_content = self._safe_content_to_string(extracted_content)
                    
                    # Check if extraction failed - but still return dict structure
                    # We'll filter these later in calculate_metrics
                    
                    # Store the string content back in result
                    result["content"] = extracted_content
                    
                    # Ensure page_references exists and is valid
                    if "page_references" not in result:
                        result["page_references"] = []
                    elif not isinstance(result["page_references"], list):
                        result["page_references"] = []
                    
                    # Validate page numbers are integers
                    valid_pages = []
                    for page in result["page_references"]:
                        try:
                            valid_pages.append(int(page))
                        except (ValueError, TypeError):
                            continue
                    result["page_references"] = sorted(list(set(valid_pages)))  # Remove duplicates and sort
                    
                    # If page_references is empty, try to extract them from the chunk_text
                    if not result["page_references"] and extracted_content and extracted_content != "Not found in provided text":
                        detected_pages = self.extract_page_numbers_from_text(chunk_text, extracted_content)
                        if detected_pages:
                            result["page_references"] = detected_pages
                            logger.info(f"📄 Detected pages from content: {detected_pages}")
                    
                    logger.info(f"✅ Successfully extracted {field_name} ({len(extracted_content)} chars, pages: {result['page_references']})")
                    return result
                    return result
                else:
                    # Fallback to old format
                    return {"content": self._safe_content_to_string(content), "page_references": []}
                    
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON decode error for {field_name}: {json_error}")
                # Fallback: treat entire response as content
                safe_content = self._safe_content_to_string(content)
                # Always return a dict, even if content seems empty
                # The metrics calculation will handle filtering
                return {"content": safe_content, "page_references": []}
            
        except Exception as e:
            logger.error(f"Error extracting {field_name}: {e}")
            return None
    
    def extract_all_fields(
        self,
        full_text: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """
        Extract all fields using multi-turn strategy.
        
        Args:
            full_text: Complete PDF text
            progress_callback: Function to call with (field_name, progress, total)
            
        Returns:
            Dictionary of extracted fields
        """
        results = {}
        total_fields = len(self.FIELD_CONFIGS)
        
        # Field descriptions (same as in langgraph_workflow.py)
        field_descriptions = {
            "study_overview": "Study title, NCT ID, protocol number, phase, study type (randomized/non-randomized, controlled/uncontrolled, blinded/open-label), disease/condition, and study duration",
            "brief_description": "Concise 2-3 sentence summary of the study purpose, rationale, and overall design",
            "primary_secondary_objectives": "PRIMARY objectives (clearly labeled) with full definitions and timeframes, followed by SECONDARY objectives (clearly labeled) with their definitions and timeframes.",
            "treatment_arms_interventions": "All treatment arms with arm names, interventions/drugs for each arm, exact dosing schedules, routes of administration, treatment duration, and any comparator or combination therapy details",
            "eligibility_criteria": "Complete inclusion criteria (required characteristics for enrollment) and exclusion criteria (disqualifying factors) for study participants",
            "enrollment_participant_flow": "Target sample size, actual enrollment numbers, randomization methodology, screening process, participant allocation, and flow through study phases",
            "adverse_events_profile": "Reported adverse events with frequencies/percentages, serious adverse events (SAEs), toxicity grades, and overall safety profile data",
            "study_locations": "All study sites, countries, institutions, and names of principal investigators or site coordinators",
            "sponsor_information": "Primary sponsor organization, collaborating institutions, funding sources, and contact information"
        }
        
        # Sort fields by priority
        sorted_fields = sorted(
            self.FIELD_CONFIGS.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for idx, (field_name, config) in enumerate(sorted_fields, 1):
            logger.info(f"Extracting field {idx}/{total_fields}: {field_name}")
            
            if progress_callback:
                progress_callback(field_name, idx, total_fields)
            
            # Create focused chunk for this field
            chunk_text = self.create_field_chunk(
                full_text,
                field_name,
                config["max_tokens"]
            )
            
            # Extract the field
            field_value = self.extract_field(
                chunk_text,
                field_name,
                field_descriptions[field_name]
            )
            
            if field_value and isinstance(field_value, dict):
                results[field_name] = field_value
                content_len = len(field_value.get("content", ""))
                pages = field_value.get("page_references", [])
                logger.info(f"✅ Extracted {field_name}: {content_len} chars, pages: {pages}")
            else:
                results[field_name] = {"content": "", "page_references": []}
                logger.warning(f"⚠️  No data found for {field_name}")
            
            # Rate limit management: wait between calls
            if idx < total_fields:
                time.sleep(self.delay_between_calls)
        
        return results
    
    def extract_field_with_feedback(
        self,
        full_text: str,
        field_name: str,
        feedback: str,
        max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Re-extract a specific field with user feedback for refinement.
        
        This is used when the user requests refinement of a specific field
        after reviewing the initial extraction results.
        
        Args:
            full_text: Complete PDF text
            field_name: Name of the field to re-extract
            feedback: User feedback on what's wrong or what to focus on
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary with 'content' and 'page_references' or None if failed
        """
        # Ensure full_text is a string
        if isinstance(full_text, dict):
            full_text = full_text.get('content', '') or json.dumps(full_text)
        if not isinstance(full_text, str):
            full_text = str(full_text) if full_text else ''
        
        if field_name not in self.FIELD_CONFIGS:
            logger.error(f"Unknown field: {field_name}")
            return None
        
        # Field descriptions
        field_descriptions = {
            "study_overview": "Study title, NCT ID, protocol number, phase, study type (randomized/non-randomized, controlled/uncontrolled, blinded/open-label), disease/condition, and study duration",
            "brief_description": "Concise 2-3 sentence summary of the study purpose, rationale, and overall design",
            "primary_secondary_objectives": "PRIMARY objectives (clearly labeled) with full definitions and timeframes, followed by SECONDARY objectives (clearly labeled) with their definitions and timeframes.",
            "treatment_arms_interventions": "All treatment arms with arm names, interventions/drugs for each arm, exact dosing schedules, routes of administration, treatment duration, and any comparator or combination therapy details",
            "eligibility_criteria": "Complete inclusion criteria (required characteristics for enrollment) and exclusion criteria (disqualifying factors) for study participants",
            "enrollment_participant_flow": "Target sample size, actual enrollment numbers, randomization methodology, screening process, participant allocation, and flow through study phases",
            "adverse_events_profile": "Reported adverse events with frequencies/percentages, serious adverse events (SAEs), toxicity grades, and overall safety profile data",
            "study_locations": "All study sites, countries, institutions, and names of principal investigators or site coordinators",
            "sponsor_information": "Primary sponsor organization, collaborating institutions, funding sources, and contact information"
        }
        
        config = self.FIELD_CONFIGS[field_name]
        
        for attempt in range(max_retries):
            try:
                # Create focused chunk for this field
                chunk_text = self.create_field_chunk(
                    full_text,
                    field_name,
                    config["max_tokens"]
                )
                
                # Enhanced extraction with feedback
                system_prompt = f"""You are an expert at extracting clinical trial information from PDF documents.

A user has requested refinement of the **{field_name}** field with this feedback:
"{feedback}"

Extract the following field from the provided text, paying special attention to the user's feedback:
**{field_name}**: {field_descriptions[field_name]}

USER FEEDBACK TO ADDRESS: {feedback}

CRITICAL INSTRUCTIONS:
1. Extract EXACTLY what is asked for - be precise and comprehensive
2. Address the user's feedback directly - fix any issues they mentioned
3. Include ALL relevant details from the text that match this field
4. Look for page markers in the format "--- Page X ---" to identify page numbers
5. List ALL page numbers where this information appears
6. If information is not found, return "Not found in provided text"
7. Use the exact terminology and structure from the source document
8. Include specific metrics, time frames, and definitions when present

CRITICAL JSON OUTPUT REQUIREMENTS:
- You MUST return ONLY a valid JSON object
- The "content" field MUST be a STRING (never a dict or list)
- Do NOT return markdown code blocks or any text outside the JSON
- Do NOT wrap the JSON in backticks or code markers
- Return ONLY the raw JSON object

Return your response in this EXACT JSON format:
{{"content": "The extracted information here (preserve all details, bullet points, and structure)", "page_references": [1, 2, 3]}}

IMPORTANT for page_references:
- Scan the text for markers like "--- Page 5 ---" 
- Extract the page number from each marker where relevant content appears
- Return page numbers as a list of integers
- If no page markers found, return empty list []"""

                user_prompt = f"""Extract **{field_name}** from this clinical trial document, addressing the user feedback:

{chunk_text}

Return the extracted information in JSON format with content and page_references:Ensure the content field is always a string, not dict."""

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                logger.info(f"🔄 Calling LLM for {field_name} with feedback (chunk: {len(chunk_text)} chars)...")
                start_time = time.time()
                response = self.llm.invoke(messages)
                elapsed = time.time() - start_time
                logger.info(f"✅ LLM response received for {field_name} with feedback (took {elapsed:.1f}s)")
                # Safely handle response.content which could be str, dict, list, or other types
                raw_content = response.content
                
                # Convert raw_content to string safely
                if isinstance(raw_content, str):
                    content = raw_content.strip()
                elif isinstance(raw_content, dict):
                    content = json.dumps(raw_content)
                elif isinstance(raw_content, list):
                    # Handle list of content blocks (some models return this)
                    text_parts = []
                    for item in raw_content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            # Extract text from dict (e.g., {"type": "text", "text": "..."})
                            if "text" in item:
                                text_parts.append(item["text"])
                            else:
                                text_parts.append(json.dumps(item))
                        else:
                            text_parts.append(str(item))
                    content = "".join(text_parts).strip()
                else:
                    # Fallback for any other type
                    content = str(raw_content).strip() if raw_content else ""
                
                # Try to parse as JSON
                try:
                    # Remove markdown code blocks if present
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()
                    
                    result = json.loads(content)
                    
                    # Validate structure
                    if "content" in result:
                        extracted_content = result["content"]
                        
                        # Use safe conversion to string
                        extracted_content = self._safe_content_to_string(extracted_content)
                        
                        # Check if extraction failed
                        if extracted_content.lower() in ["not found in provided text"]:
                            logger.warning(f"No content found for {field_name} with feedback")
                            if attempt < max_retries - 1:
                                continue
                            # Return empty result instead of None
                            return {"content": "", "page_references": []}
                        
                        # Store the string content back in result
                        result["content"] = extracted_content
                        
                        # Ensure page_references exists and is valid
                        if "page_references" not in result:
                            result["page_references"] = []
                        elif not isinstance(result["page_references"], list):
                            result["page_references"] = []
                        
                        # Validate page numbers are integers
                        valid_pages = []
                        for page in result["page_references"]:
                            try:
                                valid_pages.append(int(page))
                            except (ValueError, TypeError):
                                continue
                        result["page_references"] = sorted(list(set(valid_pages)))  # Remove duplicates and sort
                        
                        logger.info(f"✅ Successfully re-extracted {field_name} with feedback")
                        return result
                    else:
                        # Fallback to old format
                        safe_content = self._safe_content_to_string(content)
                        # Always return content, even if seems empty
                        return {"content": safe_content, "page_references": []}
                        
                except json.JSONDecodeError as json_error:
                    logger.warning(f"JSON decode error for {field_name} with feedback: {json_error}")
                    # Fallback: treat entire response as content
                    safe_content = self._safe_content_to_string(content)
                    # Always return content, let metrics decide if it's valid
                    return {"content": safe_content, "page_references": []}
                
            except Exception as e:
                logger.error(f"Error re-extracting {field_name} with feedback (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retry
                    wait_time = self.delay_between_calls * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to re-extract {field_name} after {max_retries} attempts")
                    # Return empty result instead of None
                    return {"content": "", "page_references": []}
        
        # Return empty result if we get here
        return {"content": "", "page_references": []}
    
    def extract_with_retry(
        self,
        full_text: str,
        progress_callback: Optional[callable] = None,
        max_retries: int = 3
    ) -> Dict[str, str]:
        """
        Extract all fields with automatic retry on failures.
        
        Args:
            full_text: Complete PDF text
            progress_callback: Progress update function
            max_retries: Maximum retry attempts per field
            
        Returns:
            Dictionary of extracted fields
        """
        # Ensure full_text is a string at the entry point
        if isinstance(full_text, dict):
            full_text = full_text.get('content', '') or json.dumps(full_text)
        if not isinstance(full_text, str):
            full_text = str(full_text) if full_text else ''
        
        results = {}
        
        for field_name in self.FIELD_CONFIGS.keys():
            for attempt in range(max_retries):
                try:
                    # Get existing results to pass to callback
                    total = len(self.FIELD_CONFIGS)
                    current = len(results) + 1
                    
                    if progress_callback:
                        progress_callback(field_name, current, total)
                    
                    # Create chunk and extract
                    config = self.FIELD_CONFIGS[field_name]
                    chunk_text = self.create_field_chunk(
                        full_text,
                        field_name,
                        config["max_tokens"]
                    )
                    
                    field_descriptions = {
                        "study_overview": "Study title, NCT ID, protocol number, phase, study type (randomized/non-randomized, controlled/uncontrolled, blinded/open-label), disease/condition, and study duration",
                        "brief_description": "Concise 2-3 sentence summary of the study purpose, rationale, and overall design",
                        "primary_secondary_objectives": "PRIMARY objectives (clearly labeled) with full definitions and timeframes, followed by SECONDARY objectives (clearly labeled) with their definitions and timeframes.",
                        "treatment_arms_interventions": "All treatment arms with arm names, interventions/drugs for each arm, exact dosing schedules, routes of administration, treatment duration, and any comparator or combination therapy details",
                        "eligibility_criteria": "Complete inclusion criteria (required characteristics for enrollment) and exclusion criteria (disqualifying factors) for study participants",
                        "enrollment_participant_flow": "Target sample size, actual enrollment numbers, randomization methodology, screening process, participant allocation, and flow through study phases",
                        "adverse_events_profile": "Reported adverse events with frequencies/percentages, serious adverse events (SAEs), toxicity grades, and overall safety profile data",
                        "study_locations": "All study sites, countries, institutions, and names of principal investigators or site coordinators",
                        "sponsor_information": "Primary sponsor organization, collaborating institutions, funding sources, and contact information"
                    }
                    
                    field_value = self.extract_field(
                        chunk_text,
                        field_name,
                        field_descriptions[field_name]
                    )
                    
                    # Store result with debug info
                    if field_value and isinstance(field_value, dict):
                        content = field_value.get("content", "")
                        pages = field_value.get("page_references", [])
                        word_count = len(content.split()) if content else 0
                        logger.info(f"✅ {field_name}: {word_count} words, pages {pages}")
                        results[field_name] = field_value
                    else:
                        logger.warning(f"⚠️  {field_name}: No result from extraction")
                        results[field_name] = {"content": "", "page_references": []}
                    break  # Success, move to next field
                    
                except Exception as e:
                    logger.warning(f"⚠️  Attempt {attempt + 1} failed for {field_name}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = self.delay_between_calls * (2 ** attempt)
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        # All retries failed
                        results[field_name] = {"content": "", "page_references": []}
                        logger.error(f"❌ Failed to extract {field_name} after {max_retries} attempts")
            
            # Small delay between fields
            time.sleep(self.delay_between_calls)
        
        return results


def estimate_extraction_cost(full_text: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Estimate the cost and time for multi-turn extraction.
    
    Returns:
        Dictionary with cost, time, and token estimates
    """
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = len(encoding.encode(full_text))
    
    # Cost per million tokens (as of Nov 2024)
    costs = {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00}
    }
    
    # Estimate: each field uses ~20% of doc + generates ~500 tokens
    num_fields = 9
    input_tokens_per_field = int(total_tokens * 0.3)  # 30% overlap for context
    output_tokens_per_field = 500
    
    total_input_tokens = input_tokens_per_field * num_fields
    total_output_tokens = output_tokens_per_field * num_fields
    
    model_costs = costs.get(model, costs["gpt-4o-mini"])
    
    input_cost = (total_input_tokens / 1_000_000) * model_costs["input"]
    output_cost = (total_output_tokens / 1_000_000) * model_costs["output"]
    total_cost = input_cost + output_cost
    
    # Time estimate: ~3-5 seconds per field + delays
    estimated_time = (num_fields * 4) + (num_fields * 2)  # processing + delays
    
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "estimated_time_seconds": estimated_time,
        "estimated_time_minutes": estimated_time / 60,
        "num_fields": num_fields,
        "model": model
    }
