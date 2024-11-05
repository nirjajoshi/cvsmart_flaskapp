from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''  # Handle None values
    return text

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = docx.Document(file)
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

@app.route('/get-embedding', methods=['POST'])
def get_embedding():
    """Get embeddings from the uploaded file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_stream = BytesIO(file.read())  # Read the file into a memory stream

    # Extract text based on file type
    try:
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_stream)
        elif file.filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file_stream)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    except Exception as e:
        return jsonify({'error': f'Error extracting text: {str(e)}'}), 500

    if not text:
        return jsonify({'error': 'No text extracted from file'}), 400

    # Calculate embeddings for the extracted text
    try:
        embeddings = model.encode([text]).tolist()
    except Exception as e:
        return jsonify({'error': f'Error calculating embeddings: {str(e)}'}), 500

    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    app.run(port=5000)
