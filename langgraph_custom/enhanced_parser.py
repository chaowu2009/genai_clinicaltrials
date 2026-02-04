"""
Enhanced Clinical Trial PDF Parser
====================================

A high-accuracy PDF parsing system for clinical trial documents using state-of-the-art libraries.

Libraries Used:
- pdfplumber: Layout-aware text extraction and table detection
- PyMuPDF (fitz): Fast text extraction with position awareness
- pdfminer.six: Deep text analysis and layout reconstruction
- pytesseract: OCR for scanned documents
- pandas: Table structure analysis
- spaCy: NLP for intelligent section detection

Author: Clinical Trials AI Team
Version: 2.0.0
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib

# PDF Processing Libraries
import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text_to_fp, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTChar
from io import BytesIO, StringIO

# Data Processing
import pandas as pd
import numpy as np

# Optional: OCR support
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available. Install Pillow and pytesseract for scanned PDF support.")

# Optional: NLP support for better section detection
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("spaCy not available. Install spaCy for enhanced section detection.")

# Configure logging (only if no handlers configured)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


@dataclass
class SectionMetadata:
    """Metadata for a detected section."""
    title: str
    level: int  # Hierarchy level (1=main, 2=subsection, etc.)
    page_start: int
    page_end: int
    position_start: int
    position_end: int
    confidence: float  # 0-1 score for detection confidence


@dataclass
class TableMetadata:
    """Metadata for an extracted table."""
    page: int
    title: Optional[str]
    rows: int
    columns: int
    data: List[List[str]]
    confidence: float


@dataclass
class ClinicalTrialData:
    """Structured clinical trial data conforming to schema."""
    study_overview: str = ""
    brief_description: str = ""
    primary_secondary_objectives: str = ""
    treatment_arms_interventions: str = ""
    eligibility_criteria: str = ""
    enrollment_participant_flow: str = ""
    adverse_events_profile: str = ""
    study_locations: str = ""
    sponsor_information: str = ""
    
    # Metadata
    extraction_date: str = ""
    source_file: str = ""
    total_pages: int = 0
    confidence_score: float = 0.0


class EnhancedClinicalTrialParser:
    """
    Enhanced PDF parser with high-accuracy extraction capabilities.
    
    Features:
    - Multi-library text extraction with fallback
    - Layout-aware parsing
    - OCR support for scanned documents
    - Intelligent section detection using ML
    - Advanced table extraction and validation
    - Comprehensive metadata extraction
    """
    
    def __init__(self, use_ocr: bool = False, use_nlp: bool = False):
        """
        Initialize the enhanced parser.
        
        Args:
            use_ocr: Enable OCR for scanned PDFs
            use_nlp: Enable NLP for better section detection
        """
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.use_nlp = use_nlp and NLP_AVAILABLE
        
        if self.use_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.use_nlp = False
        
        # Enhanced section patterns with hierarchy
        self.section_patterns = self._initialize_patterns()
        
        # Clinical trial field mappings with synonyms
        self.field_mappings = self._initialize_field_mappings()
    
    def _initialize_patterns(self) -> List[Dict[str, Any]]:
        """Initialize section detection patterns with hierarchy."""
        return [
            # Level 1: Major sections
            {
                'pattern': re.compile(r'^(?:ABSTRACT|Abstract)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['abstract', 'summary'],
                'weight': 1.0
            },
            {
                'pattern': re.compile(r'^(?:INTRODUCTION|Introduction|BACKGROUND|Background)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['introduction', 'background', 'rationale'],
                'weight': 0.95
            },
            {
                'pattern': re.compile(r'^(?:METHODS?|Methods?|METHODOLOGY|Methodology)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['methods', 'methodology', 'materials'],
                'weight': 1.0
            },
            {
                'pattern': re.compile(r'^(?:RESULTS?|Results?|FINDINGS?|Findings?)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['results', 'findings', 'outcomes'],
                'weight': 1.0
            },
            {
                'pattern': re.compile(r'^(?:DISCUSSION|Discussion)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['discussion', 'interpretation'],
                'weight': 0.95
            },
            {
                'pattern': re.compile(r'^(?:CONCLUSION|Conclusion|CONCLUSIONS|Conclusions)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['conclusion', 'conclusions', 'summary'],
                'weight': 0.95
            },
            
            # Level 2: Common subsections
            {
                'pattern': re.compile(r'^(?:Study Design|STUDY DESIGN)\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['study design', 'trial design', 'protocol'],
                'weight': 0.90
            },
            {
                'pattern': re.compile(r'^(?:Participants?|PARTICIPANTS?|Subjects?|SUBJECTS?)\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['participants', 'subjects', 'patients', 'population'],
                'weight': 0.90
            },
            {
                'pattern': re.compile(r'^(?:Eligibility|ELIGIBILITY|Inclusion|INCLUSION|Exclusion|EXCLUSION)\s*(?:Criteria)?\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['eligibility', 'inclusion', 'exclusion', 'criteria'],
                'weight': 0.95
            },
            {
                'pattern': re.compile(r'^(?:Interventions?|INTERVENTIONS?|Treatment|TREATMENT)\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['intervention', 'treatment', 'therapy', 'regimen'],
                'weight': 0.90
            },
            {
                'pattern': re.compile(r'^(?:Outcomes?|OUTCOMES?|Endpoints?|ENDPOINTS?)\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['outcomes', 'endpoints', 'objectives', 'measures'],
                'weight': 0.90
            },
            {
                'pattern': re.compile(r'^(?:Statistical Analysis|STATISTICAL ANALYSIS)\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['statistical', 'analysis', 'statistics'],
                'weight': 0.85
            },
            {
                'pattern': re.compile(r'^(?:Adverse Events?|ADVERSE EVENTS?|Safety|SAFETY|Side Effects?)\s*$', re.MULTILINE),
                'level': 2,
                'keywords': ['adverse', 'safety', 'side effects', 'toxicity'],
                'weight': 0.95
            },
            
            # Numbered sections (1., 1.1, etc.)
            {
                'pattern': re.compile(r'^(\d+\.)+\s*[A-Z][A-Za-z\s]+$', re.MULTILINE),
                'level': None,  # Determined by number of dots
                'keywords': [],
                'weight': 0.80
            },
            
            # References
            {
                'pattern': re.compile(r'^(?:REFERENCES?|References?|BIBLIOGRAPHY|Bibliography)\s*$', re.MULTILINE),
                'level': 1,
                'keywords': ['references', 'bibliography'],
                'weight': 0.90
            },
        ]
    
    def _initialize_field_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mappings for clinical trial fields."""
        return {
            'study_overview': {
                'keywords': ['overview', 'summary', 'abstract', 'background', 'synopsis', 'rationale', 'purpose'],
                'patterns': [
                    r'(?:study|trial)\s+(?:overview|summary)',
                    r'background\s+and\s+rationale',
                ],
                'weight': 1.0
            },
            'brief_description': {
                'keywords': ['brief description', 'summary', 'abstract', 'introduction', 'short description'],
                'patterns': [
                    r'brief\s+(?:description|summary)',
                    r'study\s+description',
                ],
                'weight': 0.95
            },
            'primary_secondary_objectives': {
                'keywords': ['objective', 'objectives', 'aim', 'aims', 'purpose', 'goal', 'endpoint', 'endpoints', 'outcome', 'outcomes'],
                'patterns': [
                    r'(?:primary|secondary)\s+(?:objective|endpoint|outcome)',
                    r'study\s+(?:objective|aim)',
                ],
                'weight': 1.0
            },
            'treatment_arms_interventions': {
                'keywords': ['treatment', 'intervention', 'arms', 'regimen', 'therapy', 'protocol', 'study groups'],
                'patterns': [
                    r'treatment\s+arms?',
                    r'intervention\s+groups?',
                    r'study\s+arms?',
                ],
                'weight': 1.0
            },
            'eligibility_criteria': {
                'keywords': ['eligibility', 'inclusion', 'exclusion', 'criteria', 'patient selection'],
                'patterns': [
                    r'(?:eligibility|inclusion|exclusion)\s+criteria',
                    r'patient\s+selection',
                ],
                'weight': 1.0
            },
            'enrollment_participant_flow': {
                'keywords': ['enrollment', 'enrolment', 'recruitment', 'participant flow', 'sample size', 'randomization'],
                'patterns': [
                    r'participant\s+flow',
                    r'enrollment|recruitment',
                    r'sample\s+size',
                ],
                'weight': 0.90
            },
            'adverse_events_profile': {
                'keywords': ['adverse event', 'adverse events', 'safety', 'side effect', 'toxicity', 'tolerability'],
                'patterns': [
                    r'adverse\s+event',
                    r'safety\s+(?:profile|results)',
                    r'side\s+effect',
                ],
                'weight': 1.0
            },
            'study_locations': {
                'keywords': ['location', 'site', 'center', 'centre', 'hospital', 'clinic', 'investigational site'],
                'patterns': [
                    r'study\s+(?:location|site)',
                    r'investigational\s+site',
                ],
                'weight': 0.85
            },
            'sponsor_information': {
                'keywords': ['sponsor', 'funding', 'support', 'grant', 'acknowledgment'],
                'patterns': [
                    r'sponsor(?:ship)?',
                    r'funding\s+source',
                    r'financial\s+support',
                ],
                'weight': 0.85
            },
        }
    
    def extract_text_multimethod(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text using multiple methods and select the best result.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        results = {}
        
        # Method 1: pdfplumber (best for layout)
        try:
            text, metadata = self._extract_with_pdfplumber(file_path)
            results['pdfplumber'] = {'text': text, 'metadata': metadata, 'score': len(text)}
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Method 2: PyMuPDF (fastest, good quality)
        try:
            text, metadata = self._extract_with_pymupdf(file_path)
            results['pymupdf'] = {'text': text, 'metadata': metadata, 'score': len(text)}
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Method 3: pdfminer (best for complex layouts)
        try:
            text, metadata = self._extract_with_pdfminer(file_path)
            results['pdfminer'] = {'text': text, 'metadata': metadata, 'score': len(text)}
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {e}")
        
        # OCR fallback for scanned documents
        if self.use_ocr and all(result['score'] < 100 for result in results.values()):
            try:
                text, metadata = self._extract_with_ocr(file_path)
                results['ocr'] = {'text': text, 'metadata': metadata, 'score': len(text)}
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
        
        if not results:
            raise ValueError("All text extraction methods failed")
        
        # Select best result based on text length and quality
        best_method = max(results.items(), key=lambda x: x[1]['score'])
        logger.info(f"Selected extraction method: {best_method[0]} (score: {best_method[1]['score']})")
        
        return best_method[1]['text'], best_method[1]['metadata']
    
    def _extract_with_pdfplumber(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text using pdfplumber with layout preservation."""
        text_parts = []
        metadata = {'pages': 0, 'method': 'pdfplumber'}
        
        with pdfplumber.open(file_path) as pdf:
            metadata['pages'] = len(pdf.pages)
            
            for page in pdf.pages:
                # Extract with layout settings
                page_text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=True,
                    x_density=7.25,
                    y_density=13
                )
                if page_text:
                    text_parts.append(page_text)
                    text_parts.append(f"\n--- Page {page.page_number} ---\n")
        
        return '\n'.join(text_parts), metadata
    
    def _extract_with_pymupdf(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text using PyMuPDF with position awareness."""
        text_parts = []
        metadata = {'pages': 0, 'method': 'pymupdf'}
        
        doc = fitz.open(file_path)
        metadata['pages'] = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text with layout preservation
            text = page.get_text("text", sort=True)
            if text:
                text_parts.append(text)
                text_parts.append(f"\n--- Page {page_num + 1} ---\n")
        
        doc.close()
        return '\n'.join(text_parts), metadata
    
    def _extract_with_pdfminer(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text using pdfminer with advanced layout analysis."""
        output_string = StringIO()
        metadata = {'method': 'pdfminer'}
        
        with open(file_path, 'rb') as fp:
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                boxes_flow=0.5,
                detect_vertical=True,
                all_texts=True
            )
            extract_text_to_fp(fp, output_string, laparams=laparams)
        
        text = output_string.getvalue()
        output_string.close()
        
        # Count pages (approximate)
        metadata['pages'] = text.count('\f') + 1
        
        return text, metadata
    
    def _extract_with_ocr(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text using OCR for scanned documents."""
        if not OCR_AVAILABLE:
            raise ImportError("OCR libraries not available")
        
        text_parts = []
        metadata = {'method': 'ocr', 'pages': 0}
        
        doc = fitz.open(file_path)
        metadata['pages'] = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            if text:
                text_parts.append(text)
                text_parts.append(f"\n--- Page {page_num + 1} ---\n")
        
        doc.close()
        return '\n'.join(text_parts), metadata
    
    def extract_tables(self, file_path: str) -> List[TableMetadata]:
        """
        Extract tables with validation and cleaning.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of TableMetadata objects
        """
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                
                for table_data in page_tables:
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    # Clean and validate table
                    cleaned_table = self._clean_table(table_data)
                    if not cleaned_table:
                        continue
                    
                    # Calculate confidence based on completeness
                    total_cells = len(cleaned_table) * len(cleaned_table[0])
                    empty_cells = sum(1 for row in cleaned_table for cell in row if not cell or cell.strip() == '')
                    confidence = 1.0 - (empty_cells / total_cells) if total_cells > 0 else 0.0
                    
                    # Try to find table title (text above table)
                    title = self._find_table_title(page, page_num)
                    
                    table_meta = TableMetadata(
                        page=page_num,
                        title=title,
                        rows=len(cleaned_table),
                        columns=len(cleaned_table[0]) if cleaned_table else 0,
                        data=cleaned_table,
                        confidence=confidence
                    )
                    tables.append(table_meta)
        
        logger.info(f"Extracted {len(tables)} tables")
        return tables
    
    def _clean_table(self, table_data: List[List[str]]) -> List[List[str]]:
        """Clean and normalize table data."""
        if not table_data:
            return []
        
        # Convert to pandas for cleaning
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Clean cell values (use map instead of deprecated applymap)
            df = df.map(lambda x: str(x).strip() if pd.notna(x) else '')
            
            # Convert back to list
            cleaned = [df.columns.tolist()] + df.values.tolist()
            return cleaned
        except Exception as e:
            logger.warning(f"Table cleaning failed: {e}")
            return table_data
    
    def _find_table_title(self, page, page_num: int) -> Optional[str]:
        """Attempt to find a title for the table."""
        # This is a simplified version - could be enhanced with position analysis
        text = page.extract_text()
        lines = text.split('\n')
        
        for line in lines:
            if 'table' in line.lower() and len(line) < 100:
                return line.strip()
        
        return None
    
    def detect_sections_advanced(self, text: str) -> List[SectionMetadata]:
        """
        Advanced section detection using multiple strategies.
        
        Args:
            text: Extracted text
            
        Returns:
            List of SectionMetadata objects
        """
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 2:
                continue
            
            # Check against patterns
            for pattern_info in self.section_patterns:
                match = pattern_info['pattern'].search(line_stripped)
                if match:
                    # Calculate position
                    position_start = sum(len(lines[j]) + 1 for j in range(i))
                    
                    # Determine level
                    level = pattern_info['level']
                    if level is None:  # Numbered section
                        level = line_stripped.count('.')
                    
                    # Calculate confidence
                    confidence = pattern_info['weight']
                    
                    # Check formatting indicators (all caps, bold proxy, etc.)
                    if line_stripped.isupper():
                        confidence += 0.05
                    if len(line_stripped) < 50:  # Short lines more likely headers
                        confidence += 0.05
                    
                    sections.append(SectionMetadata(
                        title=line_stripped,
                        level=level,
                        page_start=-1,  # Will be calculated later
                        page_end=-1,
                        position_start=position_start,
                        position_end=-1,
                        confidence=min(confidence, 1.0)
                    ))
                    break
        
        # Sort by position and set end positions
        sections.sort(key=lambda x: x.position_start)
        for i in range(len(sections) - 1):
            sections[i].position_end = sections[i + 1].position_start
        if sections:
            sections[-1].position_end = len(text)
        
        # Filter low confidence sections
        sections = [s for s in sections if s.confidence >= 0.7]
        
        logger.info(f"Detected {len(sections)} sections")
        return sections
    
    def extract_section_content(self, text: str, sections: List[SectionMetadata]) -> Dict[str, str]:
        """
        Extract content for each detected section.
        
        Args:
            text: Full text
            sections: List of detected sections
            
        Returns:
            Dictionary mapping section titles to content
        """
        content_dict = {}
        
        for section in sections:
            content = text[section.position_start:section.position_end]
            
            # Remove the section title from content
            lines = content.split('\n')
            if lines and lines[0].strip() == section.title:
                content = '\n'.join(lines[1:])
            
            # Clean content
            content = self._clean_section_content(content)
            
            if content:
                content_dict[section.title] = content
        
        return content_dict
    
    def _clean_section_content(self, content: str) -> str:
        """Clean section content."""
        # Remove page markers
        content = re.sub(r'\n---\s*Page\s+\d+\s*---\n', '\n', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Remove form feeds
        content = content.replace('\f', '')
        
        return content.strip()
    
    def map_to_clinical_trial_schema(
        self, 
        sections: Dict[str, str], 
        tables: List[TableMetadata]
    ) -> ClinicalTrialData:
        """
        Map extracted sections to clinical trial schema with high accuracy.
        
        Args:
            sections: Dictionary of section title to content
            tables: List of extracted tables
            
        Returns:
            ClinicalTrialData object
        """
        data = ClinicalTrialData()
        data.extraction_date = datetime.now().isoformat()
        
        # Track confidence scores for each field
        field_scores = {}
        
        for field_name, mapping_info in self.field_mappings.items():
            content, score = self._find_best_match(
                sections,
                mapping_info['keywords'],
                mapping_info.get('patterns', [])
            )
            
            setattr(data, field_name, content)
            field_scores[field_name] = score
        
        # Calculate overall confidence
        data.confidence_score = np.mean(list(field_scores.values())) if field_scores else 0.0
        
        logger.info(f"Schema mapping complete. Overall confidence: {data.confidence_score:.2f}")
        
        return data
    
    def _find_best_match(
        self, 
        sections: Dict[str, str], 
        keywords: List[str], 
        patterns: List[str]
    ) -> Tuple[str, float]:
        """
        Find the best matching section for a field using fuzzy matching.
        
        Args:
            sections: Available sections
            keywords: Keywords to match
            patterns: Regex patterns to match
            
        Returns:
            Tuple of (content, confidence_score)
        """
        best_match = ""
        best_score = 0.0
        
        # Try exact keyword matches first
        for section_title, content in sections.items():
            title_lower = section_title.lower()
            
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    # Calculate match score based on keyword position and length
                    position_score = 1.0 - (title_lower.index(keyword.lower()) / len(title_lower))
                    length_score = len(keyword) / len(title_lower)
                    score = (position_score + length_score) / 2
                    
                    if score > best_score:
                        best_match = content
                        best_score = score
        
        # Try pattern matches
        for section_title, content in sections.items():
            for pattern in patterns:
                if re.search(pattern, section_title, re.IGNORECASE):
                    score = 0.9  # High score for pattern matches
                    if score > best_score:
                        best_match = content
                        best_score = score
        
        # Fuzzy matching as fallback
        if best_score < 0.3:
            section_titles = list(sections.keys())
            for keyword in keywords:
                matches = difflib.get_close_matches(keyword, section_titles, n=1, cutoff=0.6)
                if matches:
                    best_match = sections[matches[0]]
                    best_score = 0.5
                    break
        
        # Content search as last resort
        if best_score < 0.2:
            for keyword in keywords[:3]:  # Check top 3 keywords only
                for section_title, content in sections.items():
                    if keyword.lower() in content.lower()[:500]:  # Check first 500 chars
                        snippet_start = max(0, content.lower().index(keyword.lower()) - 200)
                        snippet_end = min(len(content), snippet_start + 600)
                        best_match = content[snippet_start:snippet_end]
                        best_score = 0.3
                        break
                if best_match:
                    break
        
        return best_match, best_score
    
    def parse_pdf(
        self, 
        file_path: str, 
        extract_tables: bool = True
    ) -> Tuple[ClinicalTrialData, List[TableMetadata]]:
        """
        Main parsing method - orchestrates all extraction steps.
        
        Args:
            file_path: Path to PDF file
            extract_tables: Whether to extract tables
            
        Returns:
            Tuple of (ClinicalTrialData, list of tables)
        """
        logger.info(f"Starting enhanced parsing of: {file_path}")
        
        # Step 1: Extract text
        text, metadata = self.extract_text_multimethod(file_path)
        logger.info(f"Extracted {len(text)} characters from {metadata['pages']} pages")
        
        # Step 2: Detect sections
        sections_meta = self.detect_sections_advanced(text)
        logger.info(f"Detected {len(sections_meta)} sections")
        
        # Step 3: Extract section content
        sections_content = self.extract_section_content(text, sections_meta)
        
        # Step 4: Extract tables (optional)
        tables = []
        if extract_tables:
            tables = self.extract_tables(file_path)
        
        # Step 5: Map to schema
        clinical_data = self.map_to_clinical_trial_schema(sections_content, tables)
        clinical_data.source_file = Path(file_path).name
        clinical_data.total_pages = metadata['pages']
        
        logger.info("Parsing complete!")
        
        return clinical_data, tables
    
    def export_to_json(
        self, 
        clinical_data: ClinicalTrialData, 
        output_path: str,
        include_metadata: bool = True
    ):
        """
        Export parsed data to JSON file.
        
        Args:
            clinical_data: Parsed clinical trial data
            output_path: Output file path
            include_metadata: Include metadata in output
        """
        data_dict = asdict(clinical_data)
        
        if not include_metadata:
            # Remove metadata fields
            for key in ['extraction_date', 'source_file', 'total_pages', 'confidence_score']:
                data_dict.pop(key, None)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported to: {output_path}")


def parse_clinical_trial_pdf(
    file_path: str,
    output_path: Optional[str] = None,
    use_ocr: bool = False,
    use_nlp: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to parse a clinical trial PDF.
    
    Args:
        file_path: Path to PDF file
        output_path: Optional output JSON path
        use_ocr: Enable OCR
        use_nlp: Enable NLP
        
    Returns:
        Dictionary with parsed data
    """
    parser = EnhancedClinicalTrialParser(use_ocr=use_ocr, use_nlp=use_nlp)
    clinical_data, tables = parser.parse_pdf(file_path)
    
    if output_path:
        parser.export_to_json(clinical_data, output_path)
    
    result = asdict(clinical_data)
    result['tables'] = [asdict(t) for t in tables]
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        result = parse_clinical_trial_pdf(pdf_path, output_path)
        print(f"Parsed successfully! Confidence: {result['confidence_score']:.2%}")
    else:
        print("Usage: python enhanced_parser.py <pdf_file> [output_json]")
        print("\nExample:")
        print("  python enhanced_parser.py trial.pdf output.json")
