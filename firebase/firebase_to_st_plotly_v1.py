import random
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, storage, firestore
import streamlit as st
import pandas as pd
import plotly.express as px
import json, os, dotenv
from dotenv import load_dotenv
load_dotenv()

os.environ["FIREBASE_CREDENTIAL"] = dotenv.get_key(dotenv.find_dotenv(), "FIREBASE_CREDENTIAL")
cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_CREDENTIAL")))

# Initialize Firebase (if not already initialized)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {'storageBucket': 'healthhack-store.appspot.com'})

#firebase_admin.initialize_app(cred,{'storageBucket': 'healthhack-store.appspot.com'}) # connecting to firebase
db = firestore.client()

docs = db.collection("clinical_scores").stream()

# for doc in docs:
#     print(f"{doc.id} => {doc.to_dict()}")

# Create a list of dictionaries from the documents
data = []
for doc in docs:
    doc_dict = doc.to_dict()
    doc_dict['document_id'] = doc.id  # In case you need the document ID later
    data.append(doc_dict)

# Create a DataFrame
df = pd.DataFrame(data)

# Set 'name' as the index of the DataFrame
# df.set_index('name', inplace=True)

# Now, 'df' is the DataFrame with 'name' as the index
print(df)

# Load your DataFrame (assuming it's named df)
# df = pd.read_csv('your_file.csv')

# Convert date from string to datetime if it's not already in datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Streamlit page configuration
st.set_page_config(page_title="Interactive Data Dashboard", layout="wide")

# Sidebar - Selection
st.sidebar.header("Filter here:")
selected_name = st.sidebar.multiselect(
    "Select the Name:",
    options=df['name'].unique(),
    default=df['name'].unique()
)

df_selection = df[df['name'].isin(selected_name)]


# Main panel
st.title(":bar_chart: Student Performance Dashboard")
st.markdown("##")

# Total attempts by name
total_attempts_by_name = df.groupby("name")['date'].count().reset_index()
total_attempts_by_name.columns = ['name', 'total_attempts']
fig_total_attempts = px.bar(
    total_attempts_by_name,
    x="name",
    y="total_attempts",
    title="<b>Total Attempts by Name</b>",
    color_discrete_sequence=["#0083B8"] * len(total_attempts_by_name),
    template="plotly_white",
)
st.plotly_chart(fig_total_attempts, use_container_width=True)

# Score distribution
df['score'] = pd.Categorical(df['score'], categories=['A', 'B', 'C', 'D', 'E'], ordered=True)
fig_score_distribution = px.histogram(
    df, 
    x="score", 
    title="<b>Score Distribution</b>",
    color_discrete_sequence=["#33CFA5"]
)
st.plotly_chart(fig_score_distribution, use_container_width=True)

# Students with <5 attempts
students_with_less_than_5_attempts = total_attempts_by_name[total_attempts_by_name['total_attempts'] < 5]
fig_less_than_5_attempts = px.bar(
    students_with_less_than_5_attempts,
    x="name",
    y="total_attempts",
    title="<b>Students with <5 Attempts</b>",
    color_discrete_sequence=["#D62728"] * len(students_with_less_than_5_attempts),
    template="plotly_white",
)
st.plotly_chart(fig_less_than_5_attempts, use_container_width=True)

# Selection of a student for detailed view (<5 attempts)
selected_student_less_than_5 = st.selectbox("Select a student with less than 5 attempts to view details:", students_with_less_than_5_attempts['name'])
if selected_student_less_than_5:
    st.write(df[df['name'] == selected_student_less_than_5])

# Students with at least one score of 'C', 'D', 'E'
students_with_cde = df[df['score'].isin(['C', 'D', 'E'])].groupby("name")['date'].count().reset_index()
students_with_cde.columns = ['name', 'total_attempts']
fig_students_with_cde = px.bar(
    students_with_cde,
    x="name",
    y="total_attempts",
    title="<b>Students with at least one score of 'C', 'D', 'E'</b>",
    color_discrete_sequence=["#FF7F0E"] * len(students_with_cde),
    template="plotly_white",
)
st.plotly_chart(fig_students_with_cde, use_container_width=True)

# Selection of a student for detailed view (score of 'C', 'D', 'E')
selected_student_cde = st.selectbox("Select a student with at least one score of 'C', 'D', 'E' to view details:", students_with_cde['name'])
if selected_student_cde:
    st.write(df[df['name'] == selected_student_cde])
