# Probabilistic Text Generator

A probabilistic text generation system that combines semantic templates with n-gram language models and dynamic context handling. The system learns from input text to generate coherent and contextually relevant text continuations.

## Features

- Semantic template-based generation
- N-gram language modeling with KL divergence
- Dynamic context window adaptation
- Temperature-controlled creativity
- Probability boosting for rare words
- Part-of-speech aware text generation
- Efficient model serialization and loading

## Requirements

```bash
numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/probabilistic-text-generator.git
cd probabilistic-text-generator
```

2. Install dependencies:
```bash
pip install numpy
```

## Usage

The system operates in two modes:

### 1. Training Mode

Train a new model on your text data:

```bash
python text_generator.py
# Choose option 1
# Enter the path to your training text file
```

This will create two model files:
- `semantic_generator_probabilities.pkl`: Word probabilities and templates
- `language_model_dump.pkl`: Language models and context data

### 2. Generation Mode

Generate text continuations using a trained model:

```bash
python text_generator.py
# Choose option 2
# Enter your prompts
```

Example interaction:
```
User: The sun was setting behind the mountains
AI: The sun was setting behind the mountains in the optimal policies actually have no idea of the other rules.
```

## How It Works

### Text Processing

1. **Clean Text Processing**
   - Removes URLs, email addresses, and special characters
   - Normalizes spacing and punctuation
   - Standardizes sentence formatting

2. **Semantic Analysis**
   - Identifies parts of speech
   - Extracts sentence templates
   - Categorizes words (nouns, verbs, adjectives, etc.)

### Language Modeling

1. **N-gram Models**
   - Builds unigram and bigram probability distributions
   - Tracks word transition probabilities
   - Maintains context windows for coherent generation

2. **Context Handling**
   - Dynamic context window sizing
   - KL divergence-based context relevance
   - Temperature-controlled word selection

3. **Probability Adjustments**
   - Sigmoid-based probability smoothing
   - Divergence-based probability boosting
   - Dynamic temperature adaptation

## Model Components

### SemanticGenerator
- Handles text learning and template extraction
- Manages word categorization
- Creates initial probability distributions

### NaturalTextGenerator
- Manages text continuation generation
- Handles context and temperature control
- Applies probability adjustments and boosting

## Advanced Features

1. **Dynamic Context**
   - Adapts context window size based on input
   - Balances between coherence and creativity

2. **Temperature Control**
   - Adjusts generation randomness
   - Responds to context divergence

3. **Probability Boosting**
   - Enhances rare word selection
   - Maintains output diversity

## Limitations

- Requires sufficient training data for good results
- Memory usage scales with context window size
- Generation speed depends on context complexity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

George Wagenknecht

## Acknowledgments

- Built with inspiration from various NLP techniques
- Uses efficient numpy operations for probability calculations
- Implements advanced context tracking mechanisms
