# 🐦 Tweet Sentiment Analysis

A machine learning project that analyzes tweet sentiment using Logistic Regression with TF-IDF vectorization. Built with a Streamlit web interface for real-time sentiment prediction.

## 📊 Project Overview

This project classifies tweets as **Positive**, **Negative**, or **Neutral** using:
- **Dataset**: Sentiment140 (1.6M labeled tweets)
- **Model**: Logistic Regression with optimized hyperparameters
- **Accuracy**: ~82-85%
- **Frontend**: Interactive Streamlit web app

## 🚀 Features

- **Real-time sentiment analysis** with confidence scores
- **Interactive web interface** with text input
- **Data visualizations** showing dataset statistics
- **Advanced text preprocessing** (URL removal, lemmatization, etc.)
- **Pre-trained model** for instant predictions

## 📁 Project Structure

```
tweet_sentiment_analysis/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── utils.py              # Shared preprocessing functions
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
└── database/
    └── tweets.csv       # Dataset (not included in repo)
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Quasicut7/tweet_sentiment_analysis.git
cd tweet_sentiment_analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the Sentiment140 dataset and place `tweets.csv` in the `database/` folder:
- **Dataset**: [Sentiment140](http://help.sentiment140.com/for-students)
- **File**: `training.1600000.processed.noemoticon.csv`
- **Rename to**: `tweets.csv`

### 4. Train the Model (One-time)
```bash
python train_model.py
```
*This creates `sentiment_model.pkl` and `vectorizer.pkl` files*

### 5. Run the Application
```bash
streamlit run app.py
```

## 💻 Usage

### Web Interface
1. **Sentiment Analyzer**: Enter text to get real-time sentiment prediction
2. **Data Visualization**: View dataset statistics and distributions

### Model Training
Run `train_model.py` to see the complete training process with accuracy metrics.

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF (50K features, 1-3 grams)
- **Preprocessing**: URL removal, lemmatization, character normalization
- **Parameters**: C=1.0, max_iter=1000, solver='liblinear'

### Performance
- **Training Data**: 1.44M tweets (90%)
- **Test Data**: 160K tweets (10%)
- **Accuracy**: ~82-85%
- **Prediction Time**: <1 second

## 📦 Dependencies

```
pandas
matplotlib
nltk
scikit-learn
streamlit
```

## 🚫 .gitignore

The following files are excluded from version control:
- `database/tweets.csv` (227MB dataset)
- `sentiment_model.pkl` (trained model)
- `vectorizer.pkl` (TF-IDF vectorizer)
- `__pycache__/` (Python cache)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Repository**: [GitHub](https://github.com/Quasicut7/tweet_sentiment_analysis)
- **Dataset**: [Sentiment140](http://help.sentiment140.com/for-students)
- **Streamlit**: [Documentation](https://docs.streamlit.io)

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: Make sure to download the dataset and train the model before running the application for the first time.