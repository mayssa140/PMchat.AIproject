from pymongo import MongoClient
import bcrypt
import json
import requests
import streamlit as st
import pm4py
import os
from tempfile import NamedTemporaryFile
import sys
import io
from PIL import Image
import base64
from io import BytesIO
import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Adjust the URI if needed
db = client['PMChatAI']
users_collection = db['users']
conversations_collection = db['conversations']


# Function to get the next incremental user ID
def get_next_user_id():
    # Find the current max user_id in the users collection
    last_user = users_collection.find_one(
        {"user_id": {"$exists": True}},  # Only consider documents that have a user_id field
        sort=[("user_id", -1)]  # Sort by user_id in descending order
    )

    if last_user and "user_id" in last_user:
        return last_user["user_id"] + 1  # Increment the max user_id by 1
    else:
        return 1  # If no users exist or no user has a user_id, start with user_id 1


# Function to register user with incremental user ID
def register_user(username, password):
    # Generate the next incremental user ID
    user_id = get_next_user_id()

    # Hash the user's password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Create the user data
    user_data = {
        'user_id': user_id,  # Assign the incremental user ID
        'username': username,
        'password': hashed_password.decode('utf-8')
    }
    st.session_state["user_id"] = user_id
    # Insert the new user into the users collection
    users_collection.insert_one(user_data)
    st.success(f"User {username} registered successfully with user_id: {user_id}")


# Function to authenticate user


def authenticate_user(username, password):
    user = users_collection.find_one({'username': username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        # Store user ID and username in session
        st.session_state["user_id"] = user["user_id"]
        st.session_state["username"] = username

        # Retrieve stored conversations for the user
        st.session_state["conversations"] = list(conversations_collection.find({'user_id': user["_id"]}))
        return True
    return False

def save_conversation_to_db(user_inquiry,paradigm, assistant_message1, assistant_message2, assistant_message3, code):
    if 'user_id' in st.session_state:
        conversation_data = {
            "timestamp": datetime.datetime.now(),
            "session_id": st.session_state.get("session_id", "default_session"),  # Optional session tracking
            "user_id": st.session_state['user_id'],  # Logged user's ID
            "username": st.session_state['username'],  # Optional: username
            "user_inquiry": user_inquiry,
            "paradigm": paradigm,
            "assistant_message1": assistant_message1,
            "assistant_message2": assistant_message2,
            "assistant_message3": assistant_message3,
            "code":code
        }
        # Insert conversation into MongoDB (assuming conversations_collection is already defined)
        conversations_collection.insert_one(conversation_data)



# Dashboard function
def dashboard(username):
    st.title(f"Welcome , {username}!")
    # Logout Button
    if st.button("Logout"):
        # Clear session state and redirect to login page
        st.session_state.clear()  # Clear all session state
        st.rerun()
def make_api_call(prompt) -> str:
    api_key = "sk-proj-XCNHA2CuMuWek7AeD2iaT3BlbkFJVwM87YrZeH8G4vu7atzJ"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    content = r.json()["choices"][0]["message"]["content"]

    return content

    # Function to create the routing prompt


def create_prompt_routing(inquiry):
    prompt = """
       You should choose the LLM implementation paradigm for the following user inquiry:
   <<inquiry>>

   You should use "Direct Provision of Insights" in the following situations:
   - Domain knowledge and/or semantic understanding are needed
   - ...

   You should use "Code Generation" in the following situations:
   - Numerical queries
   - Queries that requires the processing of the entire event log
   - ...

   Please make a choice for the user inquiry.

   Please express your choice as a JSON. For example, if you choose "Direct Provision of Insights", then you should produce the following JSON:
   {
       "implementationParadigm": "direct"
   }

   Otherwise, if you choose "Code Generation", then you should produce the following JSON:
   {
       "implementationParadigm": "code"
   }
       """
    prompt = prompt.replace("<<inquiry>>", inquiry)
    return prompt

    # Function to determine the routing based on the inquiry


def make_routing(inquiry):
    routing_prompt = create_prompt_routing(inquiry)

    # Execute the prompt to get the response
    llm_response = make_api_call(routing_prompt)
    print(llm_response)

    required_json = llm_response.split("```json")[-1].split("```")[0]
    required_json = json.loads(required_json)

    return required_json["implementationParadigm"]  # Either 'direct' or 'code'

    # Function for direct provision of insights


def direct_provision_of_insight(log, inquiry):
    # Convert the DataFrame to a string
    # log_str = log.to_string()
    log_str = pm4py.llm.abstract_variants(log) + pm4py.llm.abstract_dfg(log)

    prompt = """
       

           Below is the context log and an inquiry. Please follow the steps outlined:
           1.Provide the initial response to the inquiry in the context of the event log.
           Context log: <<pm4py.llm.abstract_variants(log)>>
           Inquiry: <<inquiry>>
           """

    prompt = prompt.replace("<<inquiry>>", inquiry)
    prompt = prompt.replace("<<pm4py.llm.abstract_variants(log)>>", log_str)

    llm_response = make_api_call(prompt)

    return llm_response

    # Function for evaluating direct provision of insights


def direct_provision_of_insight_evaluation(initial_text):
    prompt = """
               Please don't generate long responses, just give the essential informations
              2. Evaluate the combination of the inquiry and the initial response "text" using LLM-as-a-Judge (self-reflection). 
              Specifically, assess the response on the following criteria:
                 - **Accuracy**: How correct is the information provided?
                 - **Clarity**: How clear and understandable is the response?
                 - **Relevance**: How well does the response address the inquiry?
                 - **Efficiency**: How concise and to the point is the response?
              3. Assign a grade and score (out of 10) for the initial response "text" based on the above criteria.
                 text: <<text>>

              """
    prompt = prompt.replace("<<text>>", initial_text)
    llm_response = make_api_call(prompt)

    return llm_response

    # Function for improving direct provision of insights


def direct_provision_of_insight_improvement(text_evaluation):
    prompt = """

             4. Provide an improved response to the inquiry to improve the initial response (text) . This response should:
              - Respect the points in text_evaluation
              - Correct any inaccuracies from the initial response.
              - Be more concise .
              - Be clearer and easier to understand.
              - Achieve a higher score than the initial response.   
             5. Assign a grade and score (out of 10) for the improved response.
                text: <<text>>
                 Inquiry: <<inquiry>>

              """
    prompt = prompt.replace("<<text>>", text_evaluation)
    llm_response = make_api_call(prompt)

    return llm_response

    # Function for code generation


def code_generation(inquiry, log):
    log_str = pm4py.llm.abstract_variants(log) + pm4py.llm.abstract_dfg(log)

    prompt = """
   I have an event log stored as a Pandas dataframe inside the "log" variable.

   The case ID is stored inside 'case:concept:name' , 
   the activity is stored inside 'concept:name'. 
   The timestamp is stored inside 'time:timestamp'.
   I would like you to generate Python code that can be executed to compute the following inquiry on this "log" object:
   [Include a specific inquiry or task here, for example, inquiry ]
   Assume to work with Python/Pandas.
   If you need any process mining algorithm, please feel free to use the pm4py library too.
   Please produce some code that I can execute with the "exec()" function in Python in the context of th event log.

   Include the code between the tags ```python and ```
   Context log: <<pm4py.llm.abstract_variants>>
           Inquiry: <<inquiry>>

           """

    prompt = prompt.replace("<<inquiry>>", inquiry)
    prompt = prompt.replace("<<pm4py.llm.abstract_variants>>", log_str)

    llm_response = make_api_call(prompt)

    return llm_response


def code_generation_evaluation(code_initial):
    prompt = """
   The same  event log stored as a Pandas dataframe inside the "log" variable.
   Can you give me your evaluation to the code_initial (please  don't generate the code just I need you feedback)
   for example:
   Please evaluate the following:
   1. The code conform with the following inquiry
   1. The code is written using Python and Pandas, and you may also use the pm4py library if needed for process mining tasks.
   2. The code includes comments explaining each step to ensure clarity and understanding.
   3. The code should avoid any use of system-level commands, file I/O, network access, or any other potentially harmful operations.
   4. The code should handle potential errors gracefully (e.g., missing values, unexpected data types).
   5. The code should be modular, with key parts encapsulated in functions.
   6. the code contain the main to be execute it with exec() function

       Context log: <<log>>
               Inquiry: <<inquiry>>
               code_initial:<<code_i>>
               """

    prompt = prompt.replace("<<code_i>>", code_initial)

    llm_response = make_api_call(prompt)

    return llm_response


def code_generation_safety_ensuring(evaluation_code):
    prompt = """
       - Start with "here is the safe code"
       - I would like you to regenerate the Python code that can be executed to compute the following inquiry and  that incorporates the points in the evaluation_code.
       - The code that  I can execute with the "exec()" function in Python in the context of the event log and have result ( don't put the main in comment).
       - Include a final review to ensure correctness, efficiency.
       - The same event log stored as a Pandas dataframe inside the "log" variable (don't generate any example of data I need the same "log").
          The case ID is stored inside 'case:concept:name' , 
   the activity is stored inside 'concept:name'. 
   The timestamp is stored inside 'time:timestamp'.
       - Include the code between the tags ```python and ``` 
       Context log: <<log>>
               Inquiry: <<inquiry>>
         evaluation_code: <<evaluation_code>>
       """

    prompt = prompt.replace("<<evaluation_code>>", evaluation_code)

    llm_response = make_api_call(prompt)

    return llm_response

    # Header with API URL and LLM Model selection
    # Custom CSS to style the title
    # Custom CSS to style the header
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Streamlit UI
st.set_page_config(page_title="PmChat.ai", page_icon="🤖")
st.markdown("""
       <style>
       .header {
           position: fixed; /* Fixed position */
           top: 30px; /* Aligns with the top */
           right: 0; /* Aligns with the right */
           width: 100%; /* Full width */
           height: 30px
           background-color: white; /* Background color for visibility */
           padding: 5px; /* Padding for aesthetics */
           z-index: 1000; /* Ensure it appears above other elements */
           text-align: left; /* Align text to the right */
       }
       .header h3 {
           margin: 10px; /* Remove default margin from the header */
           padding-left: 300px; /* Optional: Adjust left padding */
       }
   .rounded-image {
           border-radius: 15px;  /* Adjust this value for more or less rounding */
           width: 100px;         /* Set the width of the image */
           height: 100px;        /* Set the height of the image */
           object-fit: cover;    /* Ensure the image is not distorted */
           margin: -30px 50px 100px 30px;  /* top right bottom left */
           box-shadow: 5px 4px 15px rgb(115, 147, 179); /* Blue shadow */


       }

       </style>
   """, unsafe_allow_html=True)

# Streamlit UI
import datetime


def get_and_display_conversations():
    if 'user_id' in st.session_state:
        user_id = st.session_state['user_id']

        # Fetch conversations from both collections
        conversations = list(conversations_collection.find({"user_id": user_id}))



        # Display each conversation in the sorted order
        for conversation in conversations:
            if conversation["paradigm"] =="The routing is 'Direct Provision of Insights'":
                st.chat_message("user").markdown(conversation["user_inquiry"])
                st.chat_message("assistant").markdown(conversation["paradigm"])
                st.chat_message("assistant").markdown(conversation["assistant_message1"])
                st.chat_message("assistant").markdown(conversation.get("assistant_message2"))
                st.chat_message("assistant").markdown(conversation.get("assistant_message3"))

            else:
                st.chat_message("user").markdown(conversation["user_inquiry"])
                st.chat_message("assistant").markdown(conversation["paradigm"])
                st.chat_message("assistant").markdown(conversation["assistant_message1"])
                st.chat_message("assistant").markdown(conversation.get("assistant_message2"))
                st.chat_message("assistant").markdown(conversation.get("assistant_message3"))
                st.chat_message("assistant").markdown(conversation.get("code"))


# Make sure to replace `conversations_code_collection` with the actual name of your second MongoDB collection


# Check if user is logged in
if 'username' in st.session_state and st.session_state.username:
    dashboard(st.session_state.username)
    get_and_display_conversations()
    # Load the image
    image_path = "C:/Users/LENOVO/PycharmProjects/pythonProject/logo.JPG"
    image = Image.open(image_path)
    # Convert the image to base64
    img_base64 = image_to_base64(image)
    st.sidebar.markdown(f"""
           <div style="display: flex; justify-content: center;">
               <img src="data:image/png;base64,{img_base64}" class="rounded-image"/>
           </div>
       """, unsafe_allow_html=True)

    st.sidebar.selectbox("Select API URL", ["OpenAI", "Ollama", "DeepInfra"])
    st.sidebar.selectbox("Select LLM Model", ["gpt-4o-mini", "gpt-4o"])
    # Title
    # Header with the title
    st.markdown('<div class="header"><h3>PMchat.AI</h3></div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # File uploader for large XES files
    uploaded_file = st.file_uploader("Upload Event Log", type=["xes"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file on disk
        with NamedTemporaryFile(delete=False, suffix=".xes") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filepath = temp_file.name

        # Display progress
        st.write("Processing the XES file...")

        # Read XES file using pm4py
        try:
            log = pm4py.read_xes(temp_filepath)
            st.write("File successfully loaded!")
            st.write(f"Number of events: {len(log)}")
            st.write("First few events:", log[:5])  # Display first 5 events for preview
        except Exception as e:
            st.write(f"Error reading XES file: {e}")

        # Clean up temporary file
        os.remove(temp_filepath)

    # Chat-style interface for the inquiry
    if prompt := st.chat_input("Type your inquiry here..."):
        # Display user message in chat history
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the inquiry
        if uploaded_file:
            routing = make_routing(prompt)  # Determine the routing based on inquiry

            if routing == "direct":
                st.chat_message("assistant").markdown("The routing is 'Direct Provision of Insights'")
                direct_insight = direct_provision_of_insight(log, prompt)
                st.chat_message("assistant").markdown(direct_insight)
                evaluation_insights = direct_provision_of_insight_evaluation(direct_insight)
                st.chat_message("assistant").markdown(evaluation_insights)
                improved_result = direct_provision_of_insight_improvement(direct_insight)
                st.chat_message("assistant").markdown(improved_result)
                save_conversation_to_db(prompt, "The routing is 'Direct Provision of Insights'", direct_insight, evaluation_insights, improved_result, 0 )
            elif routing == "code":
                st.chat_message("assistant").markdown("The routing is 'Code Generation'!")
                generated_code = code_generation(prompt, log)
                st.chat_message("assistant").markdown(f"Generated Code:\n```python\n{generated_code}\n```")

                # Evaluate and execute the generated code
                self_evaluation = code_generation_evaluation(generated_code)
                st.chat_message("assistant").markdown(f"The evaluation of code is here.\n{self_evaluation}\n")

                code_ensuring = code_generation_safety_ensuring(self_evaluation)
                st.chat_message("assistant").markdown(f"Ensured code safety and evaluation.\n{code_ensuring}\n")

                # Extract and run the code
                code = generated_code.split("```python")[-1].split("```")[0]


                buffer = io.StringIO()
                sys.stdout = buffer
                try:
                    exec(code)
                    output = buffer.getvalue()  # Get the captured output
                except Exception as e:
                    output = f"Error executing code: {e}"
                finally:
                    sys.stdout = sys.__stdout__  # Reset stdout
                st.chat_message("assistant").markdown(f"Execution Output:\n```\n{output}\n```")
                save_conversation_to_db(prompt, "The routing is 'Code Generation'!", generated_code, self_evaluation, code_ensuring, output)


        else:
            st.error("Please upload an event log file.")


else:
    st.title("User Authentication")
    # Tabs for Registration and Login
    tab1, tab2 = st.tabs(["Register", "Login"])


    # Common Form Style Function
    def common_form(title, username_placeholder, password_placeholder, button_text, on_submit, username_key,
                    password_key, confirm_password_key=None):
        st.subheader(title)
        username = st.text_input(username_placeholder, key=username_key)
        password = st.text_input(password_placeholder, type='password', key=password_key)

        # Only show confirm password field if it exists
        if confirm_password_key:
            confirm_password = st.text_input("Confirm Password", type='password', key=confirm_password_key)
        else:
            confirm_password = None

        if st.button(button_text):
            on_submit(username, password, confirm_password)


    # Registration Tab
    with tab1:
        def register_action(username, password, confirm_password):
            if username and password and confirm_password:
                # Check if username already exists
                if users_collection.find_one({'username': username}):
                    st.warning("Username already exists.")
                elif password != confirm_password:
                    st.warning("Passwords do not match.")
                else:
                    register_user(username, password)
                    st.session_state.username = username  # Set session variable
                    st.success("User registered successfully! Redirecting to dashboard...")
                    st.rerun()  # Redirect to dashboard
            else:
                st.warning("Please fill in all fields.")


        common_form("Register", "Enter Username", "Enter Password", "Register", register_action, "register_username",
                    "register_password", "register_confirm_password")

    # Login Tab
    with tab2:
        def login_action(username, password, confirm_password=None):
            if username and password:
                if authenticate_user(username, password):
                    st.session_state.username = username  # Set session variable
                    st.success("Login successful! Redirecting to dashboard...")
                    st.rerun()  # Redirect to dashboard
                else:
                    st.warning("Invalid username or password.")
            else:
                st.warning("Please enter both username and password.")


        common_form("Login", "Enter Username", "Enter Password", "Login", login_action, "login_username",
                    "login_password")