import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate(r'C:\Users\paulf\Documents\EPF\Data sources\API---Webscrapping\TP2 and  3\api-fourdrinoy-firebase-adminsdk-z3rmx-a05fdb9edf.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
parameters_ref = db.collection('parameters').document('parameters')

data = {
    'n_estimators': 100,
    'criterion': 'gini'
}
parameters_ref.set(data)
