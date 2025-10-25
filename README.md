# WallStreetBets Sentiment & Sarcasm Analysis

End-to-end fine-tuned Transformer classifiers for `financial sentiment` (3-class) and `sarcasm` (2-class).
Built with PyTorch + 🤗 Transformers, trained/evaluated on multiple datasets, and packaged for reuse and continued fine-tuning.

## deberta-financial/
Base: `microsoft/deberta-v3-base`

Labels: `{0: negative, 1: neutral, 2: positive}`

Kaggle Dataset for training: `ankurzing/sentiment-analysis-for-financial-news`, `cosmos98/twitter-and-reddit-sentimental-analysis-dataset`

## sarcasm_detector/
Base: `distilbert-base-uncased`

Labels: `{0: not sarcastic, 1: sarcastic}`

Kaggle Dataset for training: `danofer/sarcasm`, `rmisra/news-headlines-dataset-for-sarcasm-detection`

### Due to git repo capacity, we upload the model on `Hugging Face Hub`
https://huggingface.co/imnotrick/sentiment_sarcasm