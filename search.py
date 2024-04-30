import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from io import BytesIO

# Download NLTK stopwords corpus
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words_list = list(stop_words)

def load_texts():
    """Load text files."""
    text_folder = r'C:\Users\MSI\Documents\tpindexation\projet2\document'
    if os.path.exists(text_folder) and os.path.isdir(text_folder):
        text_files = [os.path.join(text_folder, file) for file in os.listdir(text_folder) if file.endswith('.txt')]
        return text_files
    else:
        print(f"Error: Text folder '{text_folder}' not found.")
        return []

def load_audio_files():
    """Load audio files."""
    audio_folder = r'C:\Users\MSI\Documents\tpindexation\projet2\static\audio'
    if os.path.exists(audio_folder) and os.path.isdir(audio_folder):
        audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith('.mp3')]
        return audio_files
    else:
        print(f"Error: Audio folder '{audio_folder}' not found.")
        return []

def text_search(query_text, text_files):
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def search_similar_text(query_text, text_files, top_n=5):
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        if not text_files:
            print("No text files found.")
            return []
        text_contents = [(file, read_text_file(file)) for file in text_files]
        preprocessed_texts = [(file, preprocess_text(text)) for file, text in text_contents]
        query_vector = tfidf_vectorizer.fit_transform([preprocess_text(query_text)])
        document_vectors = tfidf_vectorizer.transform([text for _, text in preprocessed_texts])
        similarities = cosine_similarity(query_vector, document_vectors)
        indices = similarities.argsort()[0][::-1][:top_n]
        similar_text = [(text_files[i], text_contents[i][1], similarities[0][i]) for i in indices if similarities[0][i] > 0]
        return similar_text

    def preprocess_text(text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    similar_text = search_similar_text(query_text, text_files)
    results = []
    for filename, content, similarity in similar_text:
        results.append({
            'type': 'text',
            'content': content,
            'similarity': similarity
        })
    return results


def audio_search(query, audio_files):
    def extract_features(audio_path):
        try:
            y, sr = librosa.load(audio_path)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  
            return np.mean(mfccs.T, axis=0)  
        except Exception as e:
            print(f"Error extracting features from audio: {e}")
            return None

    def search_similar_audio(query_audio_path, audio_paths, top_n=3):
        query_features = extract_features(query_audio_path)
        if query_features is None:
            print("Error: Unable to extract features from query audio.")
            return []
        database_features = []
        database_audio_paths = []
        for audio_path in audio_paths:
            features = extract_features(audio_path)
            if features is not None:
                database_audio_paths.append(audio_path)
                database_features.append(features)
        if not database_features:
            print("Error: No audio files found.")
            return []
        database_features = np.array(database_features)
        similarities = cosine_similarity([query_features], database_features)
        indices = np.argsort(similarities[0])[::-1][:top_n]
        similar_audio = [(database_audio_paths[i], similarities[0][i]) for i in indices]
        return similar_audio
    similar_audio = search_similar_audio(query, load_audio_files())
    results = []
    for audio_path, similarity in similar_audio:
        results.append({
            'type': 'audio',
            'path':  '\\static\\audio\\'+os.path.basename(audio_path) , 
            'similarity': similarity
        })
    return results

def extract_features(image):
    try:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        return hist / np.sum(hist)
    except Exception as e:
        print(f"Error extracting features from image: {e}")
        return None
def search_similar_images(query_image, image_paths, top_n=3):
    query_features = extract_features(query_image)
    if query_features is None:
        print("Error: Unable to extract features from query image.")
        return []
    database_features = []
    database_image_paths = []
    for image_path in image_paths:
        features = extract_features(cv2.imread(image_path))
        if features is not None:
            database_image_paths.append(image_path)
            database_features.append(features)
    if not database_features:
        print("Error: No image files found.")
        return []
    database_features = np.array(database_features)
    similarities = cosine_similarity([query_features], database_features)
    indices = np.argsort(similarities[0])[::-1][:top_n]
    similar_images = [(database_image_paths[i], similarities[0][i]) for i in indices]
    return similar_images
def image_search(query):
    image_folder = r'C:\Users\MSI\Documents\tpindexation\projet2\static\image'
    if isinstance(query, BytesIO):
        try:
            query_image = Image.open(query)
        except Exception as e:
            print(f"Error: Unable to read uploaded image: {e}")
            return ""
    else:
        query_image_path = query
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
                   file.lower().endswith(('.jpg', '.png'))]
    if isinstance(query, BytesIO):
        similar_images = search_similar_images(query_image, image_paths)
    else:
        similar_images = search_similar_images(Image.open(query_image_path), image_paths)
    results = []
    for image_path, similarity in similar_images:
        results.append({
            'type': 'image',
            'path': '\\static\\image\\'+os.path.basename(image_path) , 
            'similarity': similarity
        })

    return results