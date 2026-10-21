import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/vagas')
def get_vagas():
    try:
        with open('parking_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"erro": f"Não foi possível ler o arquivo: {str(e)}"}), 500

if __name__ == '__main__':
    print("Servidor iniciado em http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
