from flask import Flask, render_template, request, jsonify, send_from_directory
from chatbot.rag_chatbot import MedicalChatbot
import os

app = Flask(__name__)

# Initialize chatbot
PDF_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'pdfs')
CSV_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'csvs')
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)
chatbot = MedicalChatbot(pdf_folder=PDF_FOLDER, csv_folder=CSV_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        response = chatbot.get_response(user_message)
        return jsonify({
            'response': response['answer'],
            'sources': response['sources']
        })
    except Exception as e:
        import traceback
        print("Error in /api/chat:", e)
        traceback.print_exc()
        return jsonify({
            'response': f"Sorry, there was an error: {str(e)}",
            'sources': {'images': [], 'tables': [], 'text': [], 'doctors': []}
        }), 200

# Serve static files (images and tables)
@app.route('/static/extracted_images/<path:filename>')
def serve_image(filename):
    return send_from_directory(chatbot.image_dir, filename)

@app.route('/static/extracted_tables/<path:filename>')
def serve_table(filename):
    return send_from_directory(chatbot.table_dir, filename)

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    if chatbot.memory is not None:
        chatbot.memory.clear()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True) 