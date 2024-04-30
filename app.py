from flask import Flask, render_template, request
from search import text_search, audio_search, image_search, load_texts, load_audio_files
import os
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/search', methods=['POST'])
def search():
    query_type = request.form['query_type']
    if query_type == 'text':
        query = request.form['text_query']
        text_results =  text_search(query, load_texts())
        audio_query= rf'C:\Users\MSI\Documents\tpindexation\projet2\static\audio\{query}.mp3'
        if os.path.exists(audio_query):
            audio_results = audio_search(audio_query, load_audio_files())
        else:
            audio_results = []
        image_query=rf'C:\Users\MSI\Documents\tpindexation\projet2\static\image\{query}.jpg'
        if os.path.exists(image_query):
            image_results = image_search(image_query)
        else:
            image_results = []
        results = text_results + audio_results + image_results
        print(results)
    elif query_type == 'audio':
        query = request.files['audio_query']
        audio_results = audio_search(query, load_audio_files())
        results = audio_results
    elif query_type == 'image':
        query = request.files['image_query']
        image_results = image_search(query)
        results = image_results
    else :
        results ="not found"
    return render_template('results.html', results=results)
if __name__ == '__main__':
    app.run(debug=True)