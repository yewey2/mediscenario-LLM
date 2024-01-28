from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components
import datetime


# client = OpenAI(api_key=st.secrets['openai']["OPENAI_API_KEY"])

from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=key)

st.title("ChatGPT-like clone")


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages_1" not in st.session_state:
    st.session_state.messages_1 = []

if "messages_2" not in st.session_state:
    st.session_state.messages_2 = []

if "start_time" not in st.session_state:
    st.session_state.start_time = None

## Testing HTML
# html_string = """
# <canvas></canvas>


# <script>
#     canvas = document.querySelector('canvas');
#     canvas.width = 1024;
#     canvas.height = 576;
#     console.log(canvas);

#     const c = canvas.getContext('2d');
#     c.fillStyle = "green";
#     c.fillRect(0,0,canvas.width,canvas.height);

#     const img = new Image();
#     img.src = "./tksfordumtrive.png";
#     c.drawImage(img,  10, 10);
# </script>

# <style>
#     body {
#         margin: 0;
#     }
# </style>
# """
# components.html(html_string,
#                 width=1280,
#                 height=640)


# Create two columns for the two chat interfaces
col1, col2 = st.columns(2)

# First chat interface
with col1:
    st.subheader("Chat 1")
    for message in st.session_state.messages_1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Second chat interface
with col2:
    st.subheader("Chat 2")
    for message in st.session_state.messages_2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Timer and Input
time_left = None
if st.session_state.start_time:
    time_elapsed = datetime.datetime.now() - st.session_state.start_time
    time_left = datetime.timedelta(minutes=10) - time_elapsed
    st.write(f"Time left: {time_left}")

if time_left is None or time_left > datetime.timedelta(0):
    # Chat 1 is active
    prompt = st.text_input("Enter your message for Chat 1:")
    active_chat = 1
    messages = st.session_state.messages_1
elif time_left and time_left <= datetime.timedelta(0):
    # Chat 2 is active
    prompt = st.text_input("Enter your message for Chat 2:")
    active_chat = 2
    messages = st.session_state.messages_2

if prompt:
    if st.session_state.start_time is None:
        st.session_state.start_time = datetime.datetime.now()
    
    messages.append({"role": "user", "content": prompt})
    
    with (col1 if active_chat == 1 else col2):
        with st.chat_message("user"):
            st.markdown(prompt)
    
    with (col1 if active_chat == 1 else col2):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=messages,
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            messages.append({"role": "assistant", "content": full_response})


# import streamlit as st
# import time
# def count_down(ts):
#     with st.empty():
#         while ts:
#             mins, secs = divmod(ts, 60)
#             time_now = '{:02d}:{:02d}'.format(mins, secs)
#             st.header(f"{time_now}")
#             time.sleep(1)
#             ts -= 1
# st.write("Time Up!")
# def main():
#     st.title("Pomodoro")
#     time_minutes = st.number_input('Enter the time in minutes ', min_value=1, value=25)
#     time_in_seconds = time_minutes * 60
#     if st.button("START"):
#             count_down(int(time_in_seconds))
# if __name__ == '__main__':
#     main()