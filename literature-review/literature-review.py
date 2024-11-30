from typing import List, Dict, Any
import logging
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIError
import PyPDF2
import re
import requests

class LiteratureReviewGenerator:
    """
    A class to generate literature reviews from PDF sources using PocketGroq's
    RAG and LLM capabilities.
    """
    
    def __init__(self, groq_provider: GroqProvider, model: str = "llama3-8b-8192"):
        """
        Initialize the literature review generator.
        
        Args:
            groq_provider: An initialized GroqProvider instance
            model: The LLM model to use for generation
        """
        self.groq = groq_provider
        self.model = model
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise

    def extract_metadata(self, pdf_text: str) -> Dict[str, str]:
        """
        Extract metadata like title and authors from PDF text.
        
        Args:
            pdf_text: The extracted text content
            
        Returns:
            Dictionary containing metadata
        """
        # Basic metadata extraction - can be enhanced
        lines = pdf_text.split('\n')
        metadata = {
            'title': lines[0] if lines else '',
            'authors': lines[1] if len(lines) > 1 else ''
        }
        return metadata

    def generate_review(self, pdf_paths: List[str], max_length: int = 1000) -> str:
        """
        Generate a literature review from multiple PDF sources.
        
        Args:
            pdf_paths: List of paths to PDF files
            max_length: Maximum length of generated review
            
        Returns:
            Generated literature review text
        """
        try:
            # Extract content from all PDFs
            documents = []
            for path in pdf_paths:
                text = self.extract_text_from_pdf(path)
                metadata = self.extract_metadata(text)
                documents.append({
                    'text': text,
                    'metadata': metadata
                })

            # Initialize RAG if not already done
            if not self.groq.rag_manager:
                self.groq.initialize_rag()

            # Load documents into RAG system
            for doc in documents:
                self.groq.load_documents(doc['text'], persistent=True)

            # Generate the review prompt
            prompt = """Generate a comprehensive literature review based on the provided documents.
            Focus on:
            1. Key themes and findings
            2. Methodological approaches
            3. Research gaps and future directions
            4. Relationships between the papers
            
            Format the review following academic standards and maintain coherent flow between papers.
            Maximum length: {} words.
            """.format(max_length)

            # Generate the review
            review = self.groq.query_documents(prompt)
            
            # Post-process and format
            review = self._format_review(review)
            
            return review

        except Exception as e:
            self.logger.error(f"Error generating literature review: {str(e)}")
            raise

    def _format_review(self, review: str) -> str:
        """
        Format and clean up the generated review.
        
        Args:
            review: Raw generated review text
            
        Returns:
            Formatted review text
        """
        # Remove any redundant whitespace
        review = re.sub(r'\s+', ' ', review).strip()
        
        # Add proper paragraph breaks
        review = review.replace(". ", ".\n\n")
        
        return review

    def evaluate_review(self, review: str, criteria: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the quality of the generated review.
        
        Args:
            review: Generated review text
            criteria: List of evaluation criteria
            
        Returns:
            Dictionary of evaluation scores
        """
        if criteria is None:
            criteria = [
                "coherence",
                "coverage",
                "academic_style",
                "synthesis"
            ]
            
        scores = {}
        for criterion in criteria:
            prompt = f"Evaluate the following literature review on {criterion} on a scale of 0-1:\n\n{review}"
            score = float(self.groq.generate(prompt, temperature=0))
            scores[criterion] = score
            
        return scores