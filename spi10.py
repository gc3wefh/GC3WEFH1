from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import streamlit as st
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
from langchain_groq import ChatGroq

def get_df_code(llm, question):
    prompt = PromptTemplate(
        template="""system
        We have a dataframe df with the following columns:
            Code	
            Station_ID	
            JMD_code	
            Station_Name	
            Altitude_m	
            Latitude	
            Longitude	
            Time	
            SPI

        The following is the request from a user:    
        {question}

        Generate the python code for the request as one statement st.session_state.df = ... only without any explanation.

        Answer:assistant
        """,
        input_variables=["question"],
    )

    df_code_chain = prompt | llm | StrOutputParser()
    return df_code_chain.invoke({"question": question})

title = "Jordan Standardized Precipitation Index"
st.set_page_config(layout="wide", page_title=title)
st.markdown(f"### {title}")

# Set up LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Create a Kepler map
map1 = KeplerGl(height=400)

config = {
    "version": "v1",
    "config": {
        "mapState": {
            "bearing": 0,
            "latitude": 32.24,
            "longitude": 35.35,
            "pitch": 0,
            "zoom": 6,
        },
        "visState": {
            'layerBlending': "additive",
        }
    },
}
map1.config = config

# Load CSV file
df = pd.read_csv('dataset/SPI/Jordan Standardized Precipitation Index.csv')

# Add data type check and debugging prints
if "df" in st.session_state:
    st.write(f"Data type: {type(st.session_state.df)}")
    if isinstance(st.session_state.df, pd.DataFrame):
        st.write(st.session_state.df.head())
        map1.add_data(data=st.session_state.df, name=title)
    else:
        st.error("Data is not a valid DataFrame")
else:
    st.write(f"Data type: {type(df)}")
    if isinstance(df, pd.DataFrame):
        st.write(df.head())
        map1.add_data(data=df, name=title)
    else:
        st.error("Loaded CSV is not a valid DataFrame")

# Set up two columns for the map and chat interface
col1, col2 = st.columns([3, 2])

with col1:
    keplergl_static(map1)

# Set up the chat interface
with col2:
    # Create a container for the chat messages
    chat_container = st.container()

    # Show the chat history
    for message in st.session_state.chat:
        with chat_container:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    # Add guidance text
    st.markdown("""
    ### Guidance
    You can ask questions such as:
    - What is the average SPI for Jarash?
    - Show entire table
    - What is the altitude of Salt?
    - Aggregate precipitation across all stations in Amman
    - Show dynamics of SPI in Jarash
    - Create a chart of dynamics of SPI in Jarash
    """)

    # Get user input
    user_input = st.chat_input("Type your question. ")
    if user_input:
        with chat_container:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("We are in the process of your request"):
                    try:
                        result = get_df_code(llm, user_input)
                        exec(result)
                        # Ensure that st.session_state.df is a DataFrame
                        if isinstance(st.session_state.df, pd.Series):
                            st.session_state.df = st.session_state.df.to_frame().T
                        if isinstance(st.session_state.df, pd.DataFrame):
                            response = f"Your request was processed. {st.session_state.df.shape[0]} rows are found and displayed"
                        else:
                            response = "Processed data is not a valid DataFrame. Please refine your request and try again."
                    except Exception as e:
                        response = f"We are not able to process your request. Please refine your request and try again. Error: {e}"
                    st.session_state.chat.append({"role": "assistant", "content": response})
                    st.rerun()

if "df" in st.session_state:
    st.dataframe(st.session_state.df)
else:
    st.dataframe(df)
