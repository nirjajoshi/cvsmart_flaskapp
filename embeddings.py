from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os
import PyPDF2
import docx

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

@app.route('/get-embedding', methods=['POST'])
def get_embedding():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Save the file to the temp directory
    file_path = os.path.join(temp_dir, file.filename)  # Save to temp folder
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

    # Calculate embeddings for the extracted text
    embeddings = model.encode([text]).tolist()
    print("Embeddings:", embeddings)  # Log the embeddings to see their shape and size
    print("Embedding length:", len(embeddings[0])) 

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    app.run(port=5000)
 

