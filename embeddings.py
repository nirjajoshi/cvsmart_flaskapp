from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os
import PyPDF2
import docx
import gc

app = Flask(__name__)

# Load the pre-trained transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create a temp directory for storing uploaded files
temp_dir = 'temp'
os.makedirs(temp_dir, exist_ok=True)

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''  # Handle None values
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

def chunk_text(text, max_tokens=512):
    """Breaks the text into chunks of a maximum number of tokens."""
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

@app.route('/get-embedding', methods=['POST'])
def get_embedding():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_path = os.path.join(temp_dir, file.filename)
    
    # Save the file to the temp directory
    file.save(file_path)

    # Extract text based on file type
    try:
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            os.remove(file_path)  # Clean up on error
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        os.remove(file_path)  # Clean up on error
        return jsonify({'error': f'Error extracting text: {str(e)}'}), 500

    if not text:
        os.remove(file_path)  # Clean up on error
        return jsonify({'error': 'No text extracted from file'}), 400

    # Calculate embeddings for the extracted text in chunks
    embeddings = []
    try:
        for chunk in chunk_text(text):
            chunk_embeddings = model.encode([chunk]).tolist()
            embeddings.extend(chunk_embeddings)
            gc.collect()  # Force garbage collection after processing each chunk
    except Exception as e:
        os.remove(file_path)  # Clean up on error
        return jsonify({'error': f'Error generating embeddings: {str(e)}'}), 500

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
