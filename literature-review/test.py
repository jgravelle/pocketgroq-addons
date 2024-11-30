import os
import pytest
import tempfile
from pocketgroq import GroqProvider
from literature_review import LiteratureReviewGenerator

def create_sample_pdf(content: str) -> str:
    """Create a temporary PDF file with given content for testing."""
    import fpdf

    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Split content into lines and add to PDF
    for line in content.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        pdf_path = tmp.name
        pdf.output(pdf_path)
        
    return pdf_path

def test_literature_review_generation():
    """Test main literature review generation functionality."""
    
    # Initialize GroqProvider
    try:
        groq = GroqProvider()
    except Exception as e:
        pytest.skip(f"Skipping test due to GroqProvider initialization error: {str(e)}")
    
    # Create LiteratureReviewGenerator instance
    generator = LiteratureReviewGenerator(groq)
    
    # Create sample PDFs
    sample_papers = [
        {
            "title": "Deep Learning Advances in Computer Vision",
            "content": """
            Title: Deep Learning Advances in Computer Vision
            Authors: John Smith, Jane Doe
            
            Abstract
            This paper explores recent advances in deep learning applications for computer vision.
            We discuss convolutional neural networks and their impact on image recognition tasks.
            
            Introduction
            Computer vision has seen remarkable progress with the advent of deep learning...
            """
        },
        {
            "title": "Transformer Architecture Evolution",
            "content": """
            Title: Transformer Architecture Evolution
            Authors: Alice Johnson, Bob Wilson
            
            Abstract
            We present a comprehensive review of transformer architecture development.
            Recent innovations have led to significant improvements in NLP tasks.
            
            Introduction
            Since the introduction of the original transformer architecture...
            """
        }
    ]
    
    pdf_paths = []
    try:
        # Create temporary PDF files
        for paper in sample_papers:
            pdf_path = create_sample_pdf(paper["content"])
            pdf_paths.append(pdf_path)
        
        # Generate literature review
        review = generator.generate_review(
            pdf_paths=pdf_paths,
            max_length=500
        )
        
        # Basic assertions
        assert isinstance(review, str)
        assert len(review) > 0
        
        # Check for key content
        assert "deep learning" in review.lower()
        assert "transformer" in review.lower()
        
        # Test evaluation
        scores = generator.evaluate_review(review)
        assert isinstance(scores, dict)
        assert all(0 <= score <= 1 for score in scores.values())
        
        print("\nGenerated Review:")
        print("=" * 80)
        print(review)
        print("\nEvaluation Scores:")
        for criterion, score in scores.items():
            print(f"{criterion}: {score:.2f}")
            
    finally:
        # Cleanup temporary files
        for pdf_path in pdf_paths:
            try:
                os.unlink(pdf_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {pdf_path}: {str(e)}")

def test_error_handling():
    """Test error handling scenarios."""
    
    groq = GroqProvider()
    generator = LiteratureReviewGenerator(groq)
    
    # Test with non-existent PDF
    with pytest.raises(Exception):
        generator.generate_review(["nonexistent.pdf"])
    
    # Test with empty PDF list
    with pytest.raises(ValueError):
        generator.generate_review([])
    
    # Test with invalid PDF content
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(b"Not a valid PDF")
        tmp.flush()
        
        with pytest.raises(Exception):
            generator.generate_review([tmp.name])
        
        os.unlink(tmp.name)

if __name__ == "__main__":
    print("Running literature review generator tests...")
    test_literature_review_generation()
    test_error_handling()
    print("\nAll tests completed!")