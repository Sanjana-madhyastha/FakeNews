# Fake News Detection

## Overview
This project is a **Fake News Detection** web application that uses **Machine Learning (ML) and Natural Language Processing (NLP)** to classify news articles as **real or fake**. The application is built using **Streamlit** for the frontend and **scikit-learn** for training the model.

## Features
- **User-Friendly Interface**: Built with Streamlit for ease of use.
- **Machine Learning-Based Detection**: Uses a trained ML model to classify news articles.
- **TF-IDF Vectorization**: Converts text data into numerical format for model training.
- **Interactive and Fast**: Provides instant feedback on news credibility.

## Technologies Used
- Python
- Scikit-Learn
- Pandas
- NumPy
- Streamlit
- Joblib (for model saving/loading)

## Installation
### Prerequisites
Ensure you have Python installed (preferably 3.8 or later). You can create a virtual environment using the following:
```sh
python -m venv fakenews_env
source fakenews_env/bin/activate  # For Mac/Linux
fakenews_env\Scripts\activate  # For Windows
```

### Install Dependencies
Run the following command to install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Running the Web Application
```sh
streamlit run app.py
```

### Testing the Model
You can manually input a news article into the web app, and it will predict whether the news is **real or fake**.

## Files Structure
- `app.py` - Streamlit frontend for user interaction.
- `fake_news_model.pkl` - Trained machine learning model.
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text preprocessing.
- `requirements.txt` - List of required dependencies.

## Dataset
The model was trained on a dataset containing **real and fake news articles** collected from reliable sources. It uses **TF-IDF vectorization** for feature extraction and a **machine learning classifier** for prediction.

## Future Improvements
- Improve model accuracy with deep learning approaches.
- Integrate fact-checking APIs for enhanced detection.
- Deploy the model on cloud platforms for wider accessibility.

## License
This project is open-source and available under the **MIT License**.

## Contributors
Developed by **Sanjana**.
