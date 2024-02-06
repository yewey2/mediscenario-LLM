from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components
import datetime, time
from dataclasses import dataclass
import math
import base64

## Firestore ??
import os
# import sys
# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

import openai
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
import tiktoken
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables import chain

import langchain_community.embeddings.huggingface
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory #, ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory

import os, dotenv
from dotenv import load_dotenv
load_dotenv()

import firebase_admin, json
from firebase_admin import credentials, storage, firestore
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


if not os.path.isdir("./.streamlit"):
    os.mkdir("./.streamlit")
    print('made streamlit folder')
if not os.path.isfile("./.streamlit/secrets.toml"):
    with open("./.streamlit/secrets.toml", "w") as f:
        f.write(os.environ.get("STREAMLIT_SECRETS"))
    print('made new file')
    
import os, dotenv
from dotenv import load_dotenv
load_dotenv()

if not os.path.isdir("./.streamlit"):
    os.mkdir("./.streamlit")
    print('made streamlit folder')
if not os.path.isfile("./.streamlit/secrets.toml"):
    with open("./.streamlit/secrets.toml", "w") as f:
        f.write(os.environ.get("STREAMLIT_SECRETS"))
    print('made new file')
    

import db_firestore as db

## Load from streamlit!!
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") or st.secrets["HF_TOKEN"]
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["FIREBASE_CREDENTIAL"] = os.environ.get("FIREBASE_CREDENTIAL") or st.secrets["FIREBASE_CREDENTIAL"]


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

## Hardcode indexes for now
## TODO: Move indexes to firebase
indexes = """Bleeding
ChestPain
Dysphagia
Headache
ShortnessOfBreath
Vomiting
Weakness
Weakness2""".split("\n")

model_name = "bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceBgeEmbeddings(
        # model_name=model_name, 
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs)
embeddings = st.session_state.embeddings

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
llm = st.session_state.llm
if "llm_i" not in st.session_state:
    st.session_state.llm_i = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
llm_i = st.session_state.llm_i
if "llm_gpt4" not in st.session_state:
    st.session_state.llm_gpt4 = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
llm_gpt4 = st.session_state.llm_gpt4


if "TEMPLATE" not in st.session_state:
    with open('templates/patient.txt', 'r') as file: 
        TEMPLATE = file.read()
    st.session_state.TEMPLATE = TEMPLATE
TEMPLATE = st.session_state.TEMPLATE

prompt = PromptTemplate(
    input_variables = ["question", "context"],
    template = st.session_state.TEMPLATE
)

def format_docs(docs):
    return "\n--------------------\n".join(doc.page_content for doc in docs)


sp_mapper = {"human":"student","ai":"patient", "user":"student","assistant":"patient"}

if "TEMPLATE2" not in st.session_state:
    with open('templates/grader.txt', 'r') as file: 
        TEMPLATE2 = file.read()
    st.session_state.TEMPLATE2 = TEMPLATE2
TEMPLATE2 = st.session_state.TEMPLATE2

prompt2 = PromptTemplate(
    input_variables = ["question", "context", "history"],
    template = st.session_state.TEMPLATE2
)

@chain
def get_patient_chat_history(_):
    return st.session_state.get("patient_chat_history")


if not st.session_state.get("scenario_list", None):
    st.session_state.scenario_list = indexes

def init_patient_llm():
    index_name = f"indexes/{st.session_state.scenario_list[st.session_state.selected_scenario]}/QA"
    if "store" not in st.session_state:
        st.session_state.store = db.get_store(index_name, embeddings=embeddings)
    if "retriever" not in st.session_state:
        st.session_state.retriever = st.session_state.store.as_retriever(search_type="similarity", search_kwargs={"k":2})
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            llm=llm, memory_key="chat_history", input_key="question", 
            k=5, human_prefix="student", ai_prefix="patient",)
    
    if ("chain" not in st.session_state
        or 
        st.session_state.TEMPLATE != TEMPLATE):
        st.session_state.chain = (
        RunnableParallel({
            "context": st.session_state.retriever | format_docs, 
            "question": RunnablePassthrough()
            }) | 
        LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory, verbose=False)
    )

def init_grader_llm():
    st.session_state["patient_chat_history"] = "History\n" + '\n'.join([(sp_mapper.get(i.type, i.type) + ": "+ i.content) for i in st.session_state.memory.chat_memory.messages])
    ## Grader
    index_name = f"indexes/{st.session_state.scenario_list[st.session_state.selected_scenario]}/Rubric"
    
    ## Reset time
    st.session_state.start_time = False

    if "store2" not in st.session_state:
        st.session_state.store2 = db.get_store(index_name, embeddings=embeddings)
    if "retriever2" not in st.session_state:
        st.session_state.retriever2 = st.session_state.store2.as_retriever(search_type="similarity", search_kwargs={"k":2})

    ## Re-init history
    st.session_state["patient_chat_history"] = "History\n" + '\n'.join([(sp_mapper.get(i.type, i.type) + ": "+ i.content) for i in st.session_state.memory.chat_memory.messages])

    if ("chain2" not in st.session_state
        or 
        st.session_state.TEMPLATE2 != TEMPLATE2):
        st.session_state.chain2 = (
        RunnableParallel({
            "context": st.session_state.retriever2 | format_docs, 
            # "history": RunnableLambda(lambda _: "History\n" + '\n'.join([(sp_mapper.get(i.type, i.type) + ": "+ i.content) for i in st.session_state.memory.chat_memory.messages])),
            "history": (get_patient_chat_history),
            "question": RunnablePassthrough(),
            }) | 

            # LLMChain(llm=llm_i, prompt=prompt2, verbose=False ) #|
            LLMChain(llm=llm_gpt4, prompt=prompt2, verbose=False ) #|
            | {
                "json": itemgetter("text"),
                "text": (
                    LLMChain(
                        llm=llm, 
                        prompt=PromptTemplate(
                            input_variables=["text"],
                            template="Interpret the following JSON of the student's grades, and do a write-up for each section.\n\n```json\n{text}\n```"),
                            verbose=False)
                    )
        }
    )


login_info = {
    "bob":"builder",
    "student1": "password",
    "admin":"admin"
}

def set_username(x):
    st.session_state.username = x

def validate_username(username, password):
    if login_info.get(username) == password:
        set_username(username)
    else:
        st.warning("Wrong username or password")
    return None

if not st.session_state.get("username"):
    ## ask to login
    st.title("Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    login_button = st.button("Login", on_click=validate_username, args=[username, password])
    ll, rr = st.columns(2)
    ## TODO: Sync login info usernames to firebase, and remove this portion
    ll.header("Admin Login")
    ll.write("Username: admin")
    ll.write("Password: admin")
    rr.header("Student Login")
    rr.write("Username: student1")
    rr.write("Password: password")

else:
    if True: ## Says hello and logout 
        col_1, col_2 = st.columns([1,3])
        col_2.title(f"Hello there, {st.session_state.username}")
        # Display logout button
        if col_1.button('Logout'):
            # Remove username from session state
            del st.session_state.username
            # Rerun the app to go back to the login view
            st.rerun()

    scenario_tab, dashboard_tab = st.tabs(["Training", "Dashboard"])

    class ScenarioTabIndex:
        SELECT_SCENARIO = 0
        PATIENT_LLM = 1
        GRADER_LLM = 2

    def set_scenario_tab_index(x):
        st.session_state.scenario_tab_index=x
        return None
    
    def go_to_patient_llm():
        selected_scenario = st.session_state.get('selected_scenario')
        if selected_scenario is None or selected_scenario < 0:
            st.warning("Please select a scenario!")
        else:
            st.session_state.start_time = datetime.datetime.now()
            states = ["store", "store2","retriever","retriever2","chain","chain2"]
            for state_to_del in states:
                if state_to_del in st.session_state:
                    del st.session_state[state_to_del]
            init_patient_llm()
            set_scenario_tab_index(ScenarioTabIndex.PATIENT_LLM)
    if not st.session_state.get("scenario_tab_index"):
        set_scenario_tab_index(ScenarioTabIndex.SELECT_SCENARIO)
        
    with scenario_tab:
        ## 
        if True:
            ## Check in select scenario
            if st.session_state.scenario_tab_index == ScenarioTabIndex.SELECT_SCENARIO:
                def change_scenario(scenario_index):
                    st.session_state.selected_scenario = scenario_index
                if st.session_state.get("selected_scenario", None) is None:
                    st.session_state.selected_scenario = -1
                
                total_cols = 3
                rows = list()
                # for _ in range(0, number_of_indexes, total_cols):
                #     rows.extend(st.columns(total_cols))

                st.header(f"Selected Scenario: {st.session_state.scenario_list[st.session_state.selected_scenario] if st.session_state.selected_scenario>=0 else 'None'}")
                st.button("Generate a new scenario")
                for i, scenario in enumerate(st.session_state.scenario_list):
                    if i % total_cols == 0:
                        rows.extend(st.columns(total_cols))
                    curr_col = rows[(-total_cols + i % total_cols)]
                    tile = curr_col.container(height=120)
                    ## TODO: Implement highlight box if index is selected
                    # if st.session_state.selected_scenario == i:
                    #     tile.markdown("<style>background: pink !important;</style>", unsafe_allow_html=True)
                    tile.write(":balloon:")
                    tile.button(label=scenario, on_click=change_scenario, args=[i])

                select_scenario_btn = st.button("Select Scenario", on_click=go_to_patient_llm, args=[])
                    
            elif st.session_state.scenario_tab_index == ScenarioTabIndex.PATIENT_LLM:
                st.header("Patient info")
                ## TODO: Put the patient's info here, from SCENARIO
                # st.write("Pull the info here!!!")
                col1, col2, col3 = st.columns([1,3,1])
                with col1:
                    back_to_scenario_btn = st.button("Back to selection", on_click=set_scenario_tab_index, args=[ScenarioTabIndex.SELECT_SCENARIO])
                # with col3: 
                #     start_timer_button = st.button("START")

                with col2:
                    TIME_LIMIT = 60*10 ## to change to 10 minutes
                    time.sleep(1)
                    # if start_timer_button:
                    #     st.session_state.start_time = datetime.datetime.now()
                    # st.session_state.time = -1 if not st.session_state.get('time') else st.session_state.get('time') 
                    st.session_state.start_time = False if not st.session_state.get('start_time') else st.session_state.start_time
                        
                    from streamlit.components.v1 import html

                    
                    html(f"""
                        <style>
                        @import url('https://fonts.googleapis.com/css2?family=Pixelify+Sans&display=swap');
                        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
                        @import url('https://fonts.googleapis.com/css2?family=Monofett&display=swap');
                </style>

                <style>
                    html {{
                        font-family: 'Pixelify Sans', monospace, serif;
                        font-family: 'VT323', monospace, sans-serif;
                        font-family: 'Monofett', monospace, sans-serif;
                        font-family: 'Times New Roman', sans-serif;
                        background-color: #0E1117 !important;
                        color: RGB(250,250,250);
                        // border-radius: 25%;
                        // border: 1px solid #0E1117;
                    }}
                    html, body {{
                        // background-color: transparent !important;
                        // margin: 10px;
                        // border: 1px solid pink;
                        text-align: center;
                    }}
                    body {{
                        background-color: #0E1117;
                        // margin: 10px;
                        // border: 1px solid pink;
                    }}
                    
                    body #ttime {{
                        font-weight: bold;
                        font-family: 'VT323', monospace, sans-serif;
                        // font-family: 'Pixelify Sans', monospace, serif;
                    }}
                </style>

                <div>
                    <h1>Time left</h1>
                    <h1 id="ttime"> </h1>
                </div>


                <script>

                var x = setInterval(function() {{
                    var start_time_str = "{st.session_state.start_time}";
                    var start_date = new Date(start_time_str);
                    var curr_date = new Date();
                    var time_difference = curr_date - start_date;
                    var time_diff_secs = Math.floor(time_difference / 1000);
                    var time_left = {TIME_LIMIT} - time_diff_secs;
                    var mins = Math.floor(time_left / 60);
                    var secs = time_left % 60;
                    var fmins = mins.toString().padStart(2, '0');
                    var fsecs = secs.toString().padStart(2, '0');
                    console.log("run");

                    if (start_time_str == "False") {{
                        document.getElementById("ttime").innerHTML = 'Press "Start" to start!';
                        clearInterval(x);
                    }}
                    else if (time_left <= 0) {{
                        document.getElementById("ttime").innerHTML = "Time's Up!!!";
                        clearInterval(x);
                    }}
                    else {{
                        document.getElementById("ttime").innerHTML = `${{fmins}}:${{fsecs}}`;
                    }}
                }}, 999)

                </script>
                        """,
                        )

                with open("./public/chars/Female_talk.gif", "rb") as f:
                    contents = f.read()
                student_url = base64.b64encode(contents).decode("utf-8")
                    
                with open("./public/chars/Male_talk.gif", "rb") as f:
                    contents = f.read()
                patient_url = base64.b64encode(contents).decode("utf-8")
                interactive_container = st.container()
                user_input_col ,r = st.columns([4,1])
                def to_grader_llm():
                    init_grader_llm()
                    set_scenario_tab_index(ScenarioTabIndex.GRADER_LLM)

                with r:
                    to_grader_btn = st.button("To Grader", on_click=to_grader_llm)
                with user_input_col:
                    user_inputs = st.text_input("", placeholder="Chat with the patient here!", key="user_inputs")
                    if user_inputs:
                        response = st.session_state.chain.invoke(user_inputs).get("text")
                        st.session_state.patient_response = response
                with interactive_container:
                    html(f"""
    <style>
        body {{
            font-family: 'VT323', monospace, sans-serif;
        }}

        .conversation-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 10px;
            width: calc(100% - 20px);
            height: calc(100% - 20px);
            background-color: #add8e6; /* Soothing blue background */
            padding: 10px;
        }}
        
        .doctor-image {{
            grid-column: 1;
            grid-row: 2;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .patient-image {{
            grid-column: 2;
            grid-row: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .doctor-input {{
            grid-column: 2;
            grid-row: 2;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .patient-input {{
            grid-column: 1;
            grid-row: 1;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px; /* Rounded corners for the images */
        }}

        input[type="text"] {{
            width: 90%;
            padding: 10px;
            margin: 10px;
            border: none;
            border-radius: 5px;
        }}
    </style>
    </head>
    <body>
        <div class="conversation-container">
            <div class="doctor-image">
                <img src="data:image/png;base64,{student_url}" alt="Doctor" />
            </div>
            <div class="patient-image">
                <img src="data:image/gif;base64,{patient_url}" alt="Patient" />
            </div>
            <div class="doctor-input">
                    <span id="doctor_message">You: {st.session_state.get('user_inputs') or ''}</span>
            </div>
            <div class="patient-input">
                    <span id="patient_message">{'Patient: '+st.session_state.get('patient_response') if st.session_state.get('patient_response') else '...'}</span>
            </div>
        </div>
    </body>
    </html>

    """, height=500)
                
            elif st.session_state.scenario_tab_index == ScenarioTabIndex.GRADER_LLM:
                st.session_state.grader_output = "" if not st.session_state.get("grader_output") else st.session_state.grader_output
                def get_grades():
                    st.session_state["patient_chat_history"] = "History\n" + '\n'.join([(sp_mapper.get(i.type, i.type) + ": "+ i.content) for i in st.session_state.get("memory").chat_memory.messages])
                    txt = f"""
    <summary>
        {st.session_state.diagnosis}
    </summary>
    <differential-1>
        {st.session_state.differential_1}
    </differential-1>
    <differential-2>
        {st.session_state.differential_2}
    </differential-2>
    <differential-3>
        {st.session_state.differential_3}
    </differential-3>
    """
                    response = st.session_state.chain2.invoke(txt)
                    st.session_state.grader_output = response
                st.session_state.has_llm_output = bool(st.session_state.get("grader_output"))
                ## TODO: False for now, need check llm output!
                with st.expander("Your Diagnosis and Differentials", expanded=not st.session_state.has_llm_output):
                    st.session_state.diagnosis = st.text_area("Input your case summary and **main** diagnosis:", placeholder="This is a young gentleman with significant family history of stroke, and medical history of poorly-controlled hypertension. He presents with acute onset of bitemporal headache associated with dysarthria and meningism symptoms. Important negatives include the absence of focal neurological deficits, ataxia, and recent trauma.")
                    st.divider()
                    st.session_state.differential_1 = st.text_input("Differential 1")
                    st.session_state.differential_2 = st.text_input("Differential 2")
                    st.session_state.differential_3 = st.text_input("Differential 3")
                    with st.columns(6)[5]:
                        send_for_grading = st.button("Get grades!", on_click=get_grades)
                with st.expander("Your grade", expanded=st.session_state.has_llm_output):
                    if st.session_state.grader_output:
                        st.write(st.session_state.grader_output.get("text").get("text"))
                
                # back_btn = st.button("back to LLM?", on_click=set_scenario_tab_index, args=[ScenarioTabIndex.PATIENT_LLM])
                back_btn = st.button("New Scenario?", on_click=set_scenario_tab_index, args=[ScenarioTabIndex.SELECT_SCENARIO])
        else:
            pass
    with dashboard_tab:
        cred = db.cred
        # cred = credentials.Certificate(json.loads(os.environ.get("FIREBASE_CREDENTIAL")))

        # Initialize Firebase (if not already initialized)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'storageBucket': 'healthhack-store.appspot.com'})

        #firebase_admin.initialize_app(cred,{'storageBucket': 'healthhack-store.appspot.com'}) # connecting to firebase
        db_client = firestore.client()

        docs = db_client.collection("clinical_scores").stream()

        # Create a list of dictionaries from the documents
        data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            doc_dict['document_id'] = doc.id  # In case you need the document ID later
            data.append(doc_dict)

        # Create a DataFrame
        df = pd.DataFrame(data)

        username = st.session_state.get("username")
        st.title("Dashboard")
        
        # Convert date from string to datetime if it's not already in datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Streamlit page configuration
        #st.set_page_config(page_title="Interactive Data Dashboard", layout="wide")

        # Use df_selection for filtering data based on authenticated user
        if username != 'admin':
            df_selection = df[df['name'] == username]
        else:
            df_selection = df  # Admin sees all data

        # Chart Title: Student Performance Dashboard
        st.title(":bar_chart: Student Performance Dashboard")
        st.markdown("##")

        # Chart 1: Total attempts
        if df_selection.empty:
            st.error("No data available to display.")
        else:
            # Total attempts by name (filtered)
            total_attempts_by_name = df_selection.groupby("name")['date'].count().reset_index()
            total_attempts_by_name.columns = ['name', 'total_attempts']
            
            # For a single point or multiple points, use a scatter plot
            fig_total_attempts = px.scatter(
                total_attempts_by_name,
                x="name",
                y="total_attempts",
                title="<b>Total Attempts</b>",
                size='total_attempts',  # Adjust the size of points
                color_discrete_sequence=["#0083B8"] * len(total_attempts_by_name),
                template="plotly_white",
                text='total_attempts'  # Display total_attempts as text labels
            )
            
            # Add text annotation for each point
            for line in range(0, total_attempts_by_name.shape[0]):
                fig_total_attempts.add_annotation(
                    text=str(total_attempts_by_name['total_attempts'].iloc[line]),
                    x=total_attempts_by_name['name'].iloc[line],
                    y=total_attempts_by_name['total_attempts'].iloc[line],
                    showarrow=True,
                    font=dict(family="Courier New, monospace", size=18, color="#ffffff"),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=20,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                )
            
            # Update traces for styling
            fig_total_attempts.update_traces(marker=dict(size=12), selector=dict(mode='markers+text'))
            
            # Display the scatter plot in Streamlit
            st.plotly_chart(fig_total_attempts, use_container_width=True)

        # Chart 2 (students only): Personal scores over time
        if username != 'admin':
            # Sort the DataFrame by 'date' in chronological order
            df_selection = df_selection.sort_values(by='date')
            #fig = px.bar(df_selection, x='date', y='global_score', title='Your scores!')
    
            if len(df_selection) > 1:
                # # If more than one point, use a bar chart
                # fig = px.bar(df_selection, x='date', y='global_score', title='Global Score Over Time')
                # # fig.update_yaxes(
                # #     tickmode='array',
                # #     tickvals=[1, 2, 3, 4, 5], # Reverse the order of tickvals
                # #     ticktext=['A', 'B','C','D','E'] # Reverse the order of ticktext
                # # )
                # Mapping dictionary
                grade_to_score = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20}

                # Apply mapping to convert letter grades to numerical scores
                df_selection['numeric_score'] = df_selection['global_score'].map(grade_to_score)

                # Sort the DataFrame by 'date' in chronological order
                df_selection = df_selection.sort_values(by='date')

                # Check if there's more than one point in the DataFrame
                if len(df_selection) > 1:
                    # Create a bar chart using Plotly Express
                    fig = px.bar(df_selection, x='date', y='numeric_score', title='Your scores over time')
                else:
                    # Create a bar chart with just one point
                    fig = px.bar(df_selection, x='date', y='numeric_score', title='Global Score')

                # Manually set the y-axis ticks and labels
                fig.update_yaxes(
                    tickmode='array',
                    tickvals=list(grade_to_score.values()),  # Positions for the ticks
                    ticktext=list(grade_to_score.keys()),  # Text labels for the ticks
                    range=[0, 120]  # Extend the range a bit beyond 100 to accommodate 'A'
                )

                # # Use st.plotly_chart to display the chart in Streamlit
                # st.plotly_chart(fig, use_container_width=True)

            else:
                # For a single point, use a scatter plot
                fig = px.scatter(df_selection, x='date', y='global_score', title='Global Score',
                                text='global_score', size_max=60)
                # Add text annotation
                for line in range(0,df_selection.shape[0]):
                    fig.add_annotation(text=df_selection['global_score'].iloc[line],
                                        x=df_selection['date'].iloc[line], y=df_selection['global_score'].iloc[line],
                                        showarrow=True, font=dict(family="Courier New, monospace", size=18, color="#ffffff"),
                                        align="center", arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363",
                                        ax=20, ay=-30, bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ff7f0e",
                                        opacity=0.8)
                fig.update_traces(marker=dict(size=12), selector=dict(mode='markers+text'))

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Show students their scores over time 
            st.dataframe(df_selection[['date', 'global_score', 'name']])
        

        # Chart 3 (admin only): Global score chart    
        # Define the order of categories explicitly
        order_of_categories = ['A', 'B', 'C', 'D', 'E']

        # Convert global_score to a categorical type with the specified order
        df_selection['global_score'] = pd.Categorical(df_selection['global_score'], categories=order_of_categories, ordered=True)

        # Plot the histogram
        fig_score_distribution = px.histogram(
            df_selection, 
            x="global_score", 
            title="<b>Global Score Distribution</b>",
            color_discrete_sequence=["#33CFA5"],
            category_orders={"global_score": ["A", "B", "C", "D", "E"]}
        )
        if username == 'admin':
            st.plotly_chart(fig_score_distribution, use_container_width=True)
        

        # Chart 4 (admin only): Students with <5 attempts (filtered)
        if username == 'admin':
            students_with_less_than_5_attempts = total_attempts_by_name[total_attempts_by_name['total_attempts'] < 5]
            fig_less_than_5_attempts = px.bar(
                students_with_less_than_5_attempts,
                x="name",
                y="total_attempts",
                title="<b>Students with <5 Attempts</b>",
                color_discrete_sequence=["#D62728"] * len(students_with_less_than_5_attempts),
                template="plotly_white",
            )

        if username == 'admin':
            st.plotly_chart(fig_less_than_5_attempts, use_container_width=True)


        # Selection of a student for detailed view (<5 attempts) - based on filtered data
        if username == 'admin':    
            selected_student_less_than_5 = st.selectbox("Select a student with less than 5 attempts to view details:", students_with_less_than_5_attempts['name'])
            if selected_student_less_than_5:
                st.write(df_selection[df_selection['name'] == selected_student_less_than_5])

        # Chart 5 (admin only): Students with at least one global score of 'C', 'D', 'E' (filtered)
        if username == 'admin':  
            students_with_cde = df_selection[df_selection['global_score'].isin(['C', 'D', 'E'])].groupby("name")['date'].count().reset_index()
            students_with_cde.columns = ['name', 'total_attempts']
            fig_students_with_cde = px.bar(
                students_with_cde,
                x="name",
                y="total_attempts",
                title="<b>Students with at least one global score of 'C', 'D', 'E'</b>",
                color_discrete_sequence=["#FF7F0E"] * len(students_with_cde),
                template="plotly_white",
            )
            st.plotly_chart(fig_students_with_cde, use_container_width=True)

        # Selection of a student for detailed view (score of 'C', 'D', 'E') - based on filtered data
        if username == 'admin':
            selected_student_cde = st.selectbox("Select a student with at least one score of 'C', 'D', 'E' to view details:", students_with_cde['name'])
            if selected_student_cde:
                st.write(df_selection[df_selection['name'] == selected_student_cde])

    # Chart 7 (all): Radar Chart

        # Mapping grades to numeric values
        grade_to_numeric = {'A': 90, 'B': 70, 'C': 50, 'D': 30, 'E': 10}
        df.replace(grade_to_numeric, inplace=True)

        # Calculate average numeric scores for each category
        average_scores = df.groupby('name')[['hx_PC_score', 'hx_AS_score', 'hx_others_score', 'differentials_score']].mean().reset_index()

        if username == 'admin':
            st.title('Average Scores Radar Chart')
        else:
            st.title('Performance in each segment as compared to your friends!')

        # Categories for the radar chart
        categories = ['Presenting complaint', 'Associated symptoms', '(Others)', 'Differentials']

        st.markdown("""
        ###
        Double click on the names in the legend to include/exclude them from the plot.
        """)


        # Custom colors for better contrast
        colors = ['gold', 'cyan', 'magenta', 'green']

        # Plotly Radar Chart
        fig = go.Figure()

        for index, row in average_scores.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['hx_PC_score'], row['hx_AS_score'], row['hx_others_score'], row['differentials_score']],
                theta=categories,
                fill='toself',
                name=row['name'],
                line=dict(color=colors[index % len(colors)])
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],  # Numeric range
                    tickvals=[10, 30, 50, 70, 90],  # Positions for the grade labels
                    ticktext=['E', 'D', 'C', 'B', 'A']  # Grade labels
                )),
            showlegend=True,
            height=600,  # Set the height of the figure
            width=600    # Set the width of the figure
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
            
