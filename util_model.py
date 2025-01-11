from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the API key
api_key = os.getenv('API_KEY_GROQ')

def query_clean_message(text):
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    prompt = f"""
Please clean up the following message by correcting any grammar, spelling, and punctuation errors, and improving clarity while preserving the original meaning. 
Return only the cleaned version of the message **without enclosing it in quotes**. Here's the message:

{text}
"""
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content

# Example usage
# print(query_clean_message("thsi is a sampple text with speling erors."))
