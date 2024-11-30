# PocketGroq Literature Review Generator

This module extends PocketGroq to provide automated literature review generation capabilities, implementing approaches described in "Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation" (arXiv:2411.18583).

## Acknowledgements

This implementation is based on research by Nurshat Fateh Ali et al. from the Military Institute of Science and Technology, as detailed in their paper:

> Ali, N. F., Mosharrof, S., Mohtasim, M. M., & Krishna, T. G. (2024). Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation. arXiv:2411.18583.

Their work demonstrated that LLM-based approaches achieve superior results (ROUGE-1 score: 0.364) compared to traditional NLP methods for automated literature review generation.

## Features

- **PDF Processing**: Extract text and metadata from academic papers
- **RAG Integration**: Uses PocketGroq's Retrieval-Augmented Generation capabilities
- **Multi-Document Analysis**: Process multiple papers simultaneously
- **Quality Evaluation**: Built-in review quality assessment
- **Format Preservation**: Maintains academic writing standards

## Installation

```bash
pip install pocketgroq
```

## Usage

### Basic Example

```python
from pocketgroq import GroqProvider
from literature_review import LiteratureReviewGenerator

# Initialize
groq = GroqProvider()
generator = LiteratureReviewGenerator(groq)

# Generate review from PDFs
pdf_paths = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
review = generator.generate_review(pdf_paths, max_length=1000)

# Evaluate quality
scores = generator.evaluate_review(review)
```

### Advanced Usage

```python
# Custom evaluation criteria
criteria = ["methodology", "findings", "future_work"]
scores = generator.evaluate_review(review, criteria=criteria)

# Persist processed documents
groq.initialize_rag(rag_persistent=True)
generator.generate_review(pdf_paths, persistent=True)
```

## Configuration

The module uses these default settings:
- Model: `llama3-8b-8192`
- RAG: Enabled with persistence
- Max length: 1000 words

## Technical Details

### Architecture

1. **Document Processing**
   - PDF text extraction using PyPDF2
   - Metadata parsing
   - Content structuring

2. **RAG System**
   - Document embedding
   - Contextual retrieval
   - Knowledge integration

3. **Review Generation**
   - Content synthesis
   - Academic formatting
   - Citation handling

### Performance

Based on the original research:
- ROUGE-1: 0.364
- ROUGE-2: 0.123
- ROUGE-L: 0.181

## Testing

Run the test suite:

```bash
python test_literature_review.py
```

## Contributing

Contributions welcome! Please check our contribution guidelines.

## Known Limitations

- Currently optimized for academic papers in English
- PDF extraction may vary with different paper formats
- Quality dependent on input document clarity

## Future Development

Planned enhancements based on the original paper's recommendations:
- Integration with additional LLM models
- Enhanced metadata extraction
- Bibliography generation
- Multi-language support

## License

MIT License

## Citation

If you use this implementation in your research, please cite J. Gravelle, the original paper and this implementation:

```bibtex
@article{ali2024automated,
  title={Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation},
  author={Ali, Nurshat Fateh and Mosharrof, Shakil and Mohtasim, Md. Mahdi and Krishna, T. Gopi},
  journal={arXiv preprint arXiv:2411.18583},
  year={2024}
}
```

## Contact

For issues and support:
- GitHub Issues
- Email: [maintainer email]