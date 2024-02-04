import random
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, storage, firestore
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Create a list of dictionaries from the documents
data = []
for doc in docs:
    doc_dict = doc.to_dict()
    doc_dict['document_id'] = doc.id  # In case you need the document ID later
    data.append(doc_dict)

# Create a DataFrame
df = pd.DataFrame(data)

#print(df)

# Exception handling for irregular grading, e.g. A-, B+
def standardize_grade(value):
    if pd.isna(value):
        return value
    value = str(value).upper().strip()  # Convert to string, uppercase and remove leading/trailing spaces
    if value and value[0] in ['A', 'B', 'C', 'D', 'E']:
        return value[0]  # Return the first character if it's A, B, C, D, or E
    return value  # Return the original value if no match

# Columns to check
columns_to_check = ['hx_others_score', 'hx_AS_score', 'differentials_score', 'global_score']

# Apply the function to the specified columns
df[columns_to_check] = df[columns_to_check].applymap(standardize_grade)

login_info = {
    "student1": "password",
    "student2": "password",
    "student3": "password",
    "admin":"admin"
}
# Initialize username variable
username = None

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

if st.session_state.get("username"):
    username = st.session_state.get("username")
    st.title(f"Hello there, {st.session_state.username}")

    # Display logout button
    if st.button('Logout'):
        # Remove username from session state
        del st.session_state.username
        # Rerun the app to go back to the login view
        st.rerun()

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

