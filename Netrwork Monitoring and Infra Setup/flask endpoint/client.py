from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {"error": "No file selected"}, 400
    
    save_path = rf"./model.pth"
    file.save(save_path)
    
    return {"message": f"File {file.filename} uploaded successfully!"}

if __name__ == '__main__':
    app.run(host='192.168.85.144', port=5001, debug=True)
