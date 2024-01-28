import random
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, storage, firestore
import json, os, dotenv
from dotenv import load_dotenv
load_dotenv()

os.environ["FIREBASE_CREDENTIAL"] = dotenv.get_key(dotenv.find_dotenv(), "FIREBASE_CREDENTIAL")
cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_CREDENTIAL")))
firebase_admin.initialize_app(cred,{'storageBucket': 'healthhack-store.appspot.com'}) # connecting to firebase
db = firestore.client()

class Clinical_scores:
    def __init__(self, name, date, score, total_attempts):
        self.name = name
        self.date = date
        self.score = score
        self.total_attempts = total_attempts

    def to_dict(self):
        clinical_scores_dict = {
            "name": self.name,
            "date": self.date,
            "score": self.score,
            "total_attempts": self.total_attempts
        }
        return clinical_scores_dict

    def __repr__(self):
        return f"Clinical_scores(\
                name={self.name}, \
                date={self.date}, \
                score={self.score}, \
                total_attempts={self.total_attempts}\
            )"

# Function to generate random dates
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))

# Start and end dates for the random date generation
start_date = date(2023, 11, 1)
end_date = date(2023, 12, 31)

# Names and scores lists
names = ["Tan Wei Ming", "Lim Yu Yan", "Ong Kai Wen", "Ng Zi Xuan", "Koh Hui Wen", "Lee Jia Yi", "Tan Kai Xin", "Goh Wei Ning", "Chen Yi Ling", "Toh Zhen Yu", "Sim Li Ting", "Wong Shu Ting", "Ng Jun Kai", "Tan Jing Yi", "Chua Xue Li", "Ho Kai Lin", "Lim Jia Hui", "Teo Wei Lin", "Chen Xiu Ting", "Ang Li Hui"]
scores = ['A', 'B', 'C', 'D', 'E']

clinical_scores_ref = db.collection("clinical_scores")

# Generating and storing instances
for name in names:
    total_attempts = random.randint(1, 10)  # Randomly assigning total attempts between 1 to 10
    for attempt in range(total_attempts):
        date_of_attempt = random_date(start_date, end_date).isoformat()
        score = random.choice(scores)
        clinical_score = Clinical_scores(name, date_of_attempt, score, total_attempts)
        
        # Storing to Firebase (assuming `clinical_scores_ref` is already defined)
        clinical_scores_ref.document().set(clinical_score.to_dict())

# Sample Firebase retrieval code (assuming the document ID is known)
doc_ref = db.collection("clinical_scores").document("some_document_id")

doc = doc_ref.get()
if doc.exists:
    print(f"Document data: {doc.to_dict()}")
else:
    print("No such document!")
