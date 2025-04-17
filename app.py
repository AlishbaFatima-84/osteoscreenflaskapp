from flask import Flask, request, jsonify
from File1 import integration
import numpy as np
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
from supporting_functions import xray_image
from dotenv import load_dotenv
load_dotenv()
import os


app = Flask(__name__)
CORS(app)

# Load from environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_server = os.getenv('DB_SERVER')
db_name = os.getenv('DB_NAME')

# Encode special characters like @ in password
from urllib.parse import quote_plus
db_password_encoded = quote_plus(db_password)

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f'mssql+pyodbc://{db_user}:{db_password_encoded}@{db_server}:1433/{db_name}?driver=ODBC+Driver+17+for+SQL+Server'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
print("✅ Connected to Azure SQL")

# ✅ Patient Model
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30))
    email = db.Column(db.String(40))
    age = db.Column(db.Integer)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    bmi = db.Column(db.Float)
    blood_group = db.Column(db.String(10))
    potassium = db.Column(db.Float)
    alkaline_phosphatase = db.Column(db.Float)
    bmd = db.Column(db.Float)
    t_score = db.Column(db.Float)
    z_score = db.Column(db.Float)
    who_classification = db.Column(db.String(50))
    fracture_risk = db.Column(db.String(50))

# ✅ Create tables (if not exist)
with app.app_context():
    db.create_all()
    print("✅ Tables created")

# ✅ Utility to convert model output safely
def to_float(value):
    return float(value[0]) if isinstance(value, np.ndarray) else float(value)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Flask is working on Azure!"})

# ✅ Route to handle prediction and saving
@app.route('/api', methods=['POST'])
def sendDataToAPI():
    try:
        data = request.get_json()
        age = data.get('age')
        name = data.get('name')
        email = data.get('email')
        height = data.get('height')
        weight = data.get('weight')
        bmi = data.get('bmi')
        blood_group = data.get('blood_group')
        potassium = data.get('potassium')
        alkaline_phosphatase = data.get('alkaline_phosphatase')
        xray_base64 = data.get('xrayBytes')

        blood_group_encoded = 0
        if blood_group in ['O+', 'O-']:
            blood_group_encoded = 0
        elif blood_group in ['A+', 'A-']:
            blood_group_encoded = 1
        elif blood_group in ['B+', 'B-']:
            blood_group_encoded = 2
        else:
            blood_group_encoded = 3

        Features = np.array(
            [[age, weight, bmi, blood_group_encoded, potassium, alkaline_phosphatase]],
            dtype=np.float32
        )

        if xray_base64:
            pre_image = xray_image(xray_base64)

        bmd_value, T_Score_value, Z_Score_value, WHO_value, Fracture_risk_value = integration(pre_image, Features)

        # ✅ Fix: Convert NumPy values to native Python float
        bmd_value = to_float(bmd_value)
        T_Score_value = to_float(T_Score_value)
        Z_Score_value = to_float(Z_Score_value)

        WHO_Classification = "Normal" if WHO_value == 0 else "Osteopenia" if WHO_value == 1 else "Osteoporosis"

        patient = Patient(
            name=name,
            age=age,
            email=email,
            height=height,
            weight=weight,
            bmi=bmi,
            blood_group=blood_group,
            potassium=potassium,
            alkaline_phosphatase=alkaline_phosphatase,
            bmd=bmd_value,
            t_score=T_Score_value,
            z_score=Z_Score_value,
            who_classification=WHO_Classification,
            fracture_risk=Fracture_risk_value
        )
        db.session.add(patient)
        db.session.commit()

        return jsonify({
            "bmd": round(bmd_value, 4),
            "T_score": round(T_Score_value, 4),
            "Z_score": round(Z_Score_value, 4),
            "WHO Classification": WHO_Classification,
            "Fracture_risk": Fracture_risk_value
        })

    except Exception as e:
        print("❗ Server Error:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

# ✅ Retrieve patient history by email
@app.route('/api/patients', methods=['GET'])
def get_patient_history():
    try:
        email = request.args.get('email')
        if email:
            email = email.strip().lower()
            patients = Patient.query.filter(Patient.email.ilike(email)).all()
        else:
            patients = []

        patient_history = [{
            "id": p.id,
            "name": p.name,
            "email": p.email,
            "age": p.age,
            "height": p.height,
            "weight": p.weight,
            "bmi": p.bmi,
            "blood_group": p.blood_group,
            "potassium": p.potassium,
            "alkaline_phosphatase": p.alkaline_phosphatase,
            "bmd": p.bmd,
            "t_score": p.t_score,
            "z_score": p.z_score,
            "who_classification": p.who_classification,
            "fracture_risk": p.fracture_risk
        } for p in patients]

        return jsonify(patient_history)

    except Exception as e:
        print("❗ Server Error:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)