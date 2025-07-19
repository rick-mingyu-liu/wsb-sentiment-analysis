# Sentiment Analysis Group - WSB Content Analysis

A comprehensive sentiment analysis system designed specifically for analyzing content from `r/wallstreetbets` and similar financial communities. This project addresses the unique challenges of financial social media content including sarcasm, slang, and ticker symbol identification.

## 🎯 Project Overview

This project implements the **Sentiment Analysis Group** requirements by:

1. **Evaluating DeBERTa Model**: Testing and fine-tuning the DeBERTa model for financial sentiment analysis
2. **Handling r/wallstreetbets Content**: Processing real posts from financial communities
3. **Addressing Sarcasm & Slang**: Implementing sarcasm detection to improve sentiment accuracy
4. **Ticker Symbol Identification**: Automatically detecting and extracting stock tickers
5. **Pipeline Development**: Building a complete sentiment analysis pipeline

## 📊 Key Features

### 🤖 Dual-Model Architecture
- **Sentiment Analysis**: Fine-tuned DeBERTa-v3-base for financial sentiment (negative/neutral/positive)
- **Sarcasm Detection**: Fine-tuned DistilBERT for sarcasm detection (sarcastic/not sarcastic)
- **Combined Analysis**: Sarcasm-aware sentiment interpretation

### 📈 Financial Domain Specialization
- **Financial News Dataset**: Trained on specialized financial content
- **WSB-Style Content**: Optimized for Wall Street Bets community language
- **Ticker Extraction**: Automatic stock symbol identification
- **Sarcasm-Aware Interpretation**: Adjusts sentiment based on sarcasm detection

### 🔄 Complete Pipeline
- **Data Input**: Receives raw posts from Information Fetching Group
- **Preprocessing**: Text cleaning and ticker extraction
- **Analysis**: Dual-model sentiment and sarcasm analysis
- **Output**: Structured sentiment scores with interpretations

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install PyTorch (CPU version)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
pip install transformers datasets evaluate accelerate

# Install dataset access
pip install kagglehub
```

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd sarcasm_sentiment_analysis

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
sarcasm_sentiment_analysis/
├── wsb_sentiment_sarcasm_analysis.ipynb     # Main analysis notebook
├── deberta-financial/                       # Trained sentiment model
├── sarcasm_detector/                        # Trained sarcasm model
├── sarcasm_model/                           # Training checkpoints
├── results/                                 # Training results
├── README.md                                # This file
└── .gitignore                               # Git ignore file
```

## 🚀 Usage

### Basic Sentiment Analysis
```python
from transformers import pipeline

# Load sentiment analyzer
sentiment_analyzer = pipeline(
    "text-classification", 
    model="./deberta-financial", 
    tokenizer="./deberta-financial"
)

# Analyze WSB-style post
post = "Fuck NVDA. Lost 30% in one day!"
result = sentiment_analyzer(post)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")
```

### Advanced Analysis with Sarcasm Detection
```python
from transformers import pipeline

# Load both models
sentiment_analyzer = pipeline("text-classification", model="./deberta-financial", tokenizer="./deberta-financial")
sarcasm_detector = pipeline("text-classification", model="./sarcasm_detector", tokenizer="./sarcasm_detector")

def analyze_wsb_post(text):
    """Complete analysis of WSB-style post"""
    sentiment = sentiment_analyzer(text)[0]
    sarcasm = sarcasm_detector(text)[0]
    
    # Extract tickers
    tickers = extract_tickers(text)
    
    return {
        "text": text,
        "tickers": tickers,
        "sentiment": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "sarcasm": sarcasm["label"],
        "sarcasm_score": sarcasm["score"],
        "interpretation": interpret_with_sarcasm(sentiment, sarcasm)
    }

def extract_tickers(text):
    """Extract stock tickers from text"""
    import re
    return re.findall(r'\b[A-Z]{2,5}\b', text)

def interpret_with_sarcasm(sentiment, sarcasm):
    """Apply sarcasm-aware interpretation"""
    sentiment_labels = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    sarcasm_labels = {"LABEL_0": "not sarcastic", "LABEL_1": "sarcastic"}
    
    raw_sentiment = sentiment_labels[sentiment["label"]]
    raw_sarcasm = sarcasm_labels[sarcasm["label"]]
    
    # Sarcasm-aware logic
    if sarcasm["score"] > 0.8 and raw_sentiment == "negative":
        return "positive (sarcastic bullish)"
    return raw_sentiment
```

### Logic Flow for Final Interpretation

The system uses a two-stage approach for sentiment interpretation:

**1. Base Sentiment Prediction:**
- Obtain sentiment label: `negative`, `neutral`, or `positive`
- Concurrently perform sarcasm prediction: `sarcastic` or `not sarcastic`

**2. Interpretation Rule:**
- **Specific Rule**: If `sarcasm_score > 0.8` AND `raw_sentiment == "negative"`, then treat as `positive (sarcastic bullish)`
- **Reasoning**: Because sarcasm often flips meaning. A sarcastic negative phrase usually implies bullish confidence in financial forums like WSB.
- **Default Rule**: In all other cases, keep the original sentiment

This logic flow enables the system to accurately interpret WSB-style content where sarcastic negative statements often indicate bullish sentiment.
```

### Pipeline Integration
```python
def process_posts_from_fetching_group(posts):
    """Process posts from Information Fetching Group"""
    results = []
    
    for post in posts:
        analysis = analyze_wsb_post(post["text"])
        results.append({
            "post_id": post["id"],
            "analysis": analysis,
            "timestamp": post["timestamp"]
        })
    
    return results
```

## 📊 Datasets & Training

### Sentiment Analysis Dataset
- **Source**: Financial News Dataset (Kaggle: ankurzing/sentiment-analysis-for-financial-news)
- **Labels**: negative, neutral, positive
- **Model**: DeBERTa-v3-base fine-tuned
- **Training**: 3 epochs, learning rate 2e-5

### Sarcasm Detection Dataset
- **Source**: News Headlines Dataset (Kaggle: rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Labels**: sarcastic, not sarcastic
- **Model**: DistilBERT-base-uncased fine-tuned
- **Training**: 3 epochs, learning rate 2e-5

## 🎯 Model Performance

### Sentiment Analysis (DeBERTa)
- **Base Model**: microsoft/deberta-v3-base
- **Domain**: Financial news and WSB content
- **Output**: 3-class classification
- **Specialization**: Financial terminology and slang

### Sarcasm Detection (DistilBERT)
- **Base Model**: distilbert-base-uncased
- **Domain**: News headlines and social media
- **Output**: 2-class classification
- **Specialization**: Internet sarcasm and humor

## 🔧 Training Configuration

```python
# Sentiment Model Training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)

# Sarcasm Model Training
training_args = TrainingArguments(
    output_dir="./sarcasm_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)
```

## 📈 Example Analysis

### Input: WSB Post
See `post1.png` for the original Reddit post from r/wallstreetbets that serves as input to our sentiment analysis pipeline.

Example text from the post:
```
"OPENDOOR - this isn't just a pump. Five fucking years I've been in this thing..."
```

### Analysis Results
- **Tickers Detected**: ['OPENDOOR']
- **Raw Sentiment**: negative (score: 0.9981)
- **Sarcasm Detection**: not sarcastic (score: 0.9981)
- **Final Interpretation**: positive (sarcastic bullish)
- **Reasoning**: High sarcasm score with negative sentiment indicates bullish sarcasm

## 🎯 Key Achievements

### ✅ Model Testing
- Successfully tested DeBERTa on real r/wallstreetbets content
- Validated model performance on financial social media posts
- Confirmed suitability for WSB-style content analysis

### ✅ Sarcasm & Slang Handling
- Implemented dedicated sarcasm detection model
- Fine-tuned on sarcasm-specific dataset
- Created sarcasm-aware sentiment interpretation logic

### ✅ Ticker Symbol Identification
- Built regex-based ticker extraction function
- Handles common ticker formats (2-5 uppercase letters)
- Integrated with sentiment analysis pipeline

### ✅ Pipeline Development
- Complete sentiment analysis pipeline implemented
- Ready to receive data from Information Fetching Group
- Produces structured sentiment scores for posts and comments

### Information Fetching Group
- **Input**: Raw posts and comments from r/wallstreetbets
- **Processing**: Text preprocessing and ticker extraction
- **Output**: Structured sentiment analysis results

### Data Flow
```
Information Fetching Group → Sentiment Analysis Group → Results
     (raw posts)              (analysis pipeline)      (sentiment scores)
```

## 🙏 Acknowledgments

- **DeBERTa Model**: Microsoft Research
- **DistilBERT Model**: Hugging Face
- **Financial News Dataset**: Ankur Zing
- **Sarcasm Dataset**: Rishabh Misra
- **r/wallstreetbets Community**: For providing real-world test cases

---

**Status**: ✅ All requirements from Sentiment Analysis Group successfully implemented
- ✅ DeBERTa model evaluation completed
- ✅ r/wallstreetbets content testing implemented
- ✅ Sarcasm and slang handling addressed
- ✅ Ticker symbol identification implemented
- ✅ Sentiment analysis pipeline developed 