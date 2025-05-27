import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
# from langchain_ollama import ChatOllama  # ❌ Commented: Ollama not used
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
import streamlit as st

# ✅ Add OpenAI imports
from langchain_openai import ChatOpenAI

# Load env
load_dotenv()
# ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")  # ❌ Commented: Ollama not used
# ollama_model = os.getenv("OLLAMA_MODEL3")  # ❌ Commented: Ollama not used
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

# Initialize OpenAI LLM
llm = ChatOpenAI(
    api_key=openai_api_key,
    model=openai_model,
    temperature=0
)

# DB paths
db_path = r"D:/My_project/DB-AGENTS/db/salaries_2023.db"
csv_path = r"D:/My_project/DB-AGENTS/Data/salaries_2023.csv"

# Create DB from CSV if missing
if not os.path.exists(db_path):
    df = pd.read_csv(csv_path).fillna(0)
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("salaries_2023", engine, if_exists="replace", index=False)

# Connect to DB and show schema
db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    include_tables=["salaries_2023"],
    sample_rows_in_table_info=5
)
# print("✅ Connected tables:", db.get_usable_table_names())
# print("✅ Table schema:\n", db.get_table_info(["salaries_2023"]))

MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.
- as part of your final answer, please include the SQL query you used in json format or code format

## Tools:

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [base_salary], [grade] 
FROM salaries_2023

WHERE state = 'Division'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 workers making 100,000.

Explanation:
I queried the `xyz` table for the `salary` column where the department
is 'IGM' and the date starts with '2020'. The query returned a list of tuples
with the bazse salary for each day in 2020. To answer the question,
I took the sum of all the salaries in the list, which is 27437.
I used the following query

```sql
SELECT [salary] FROM xyztable WHERE department = 'IGM' AND date LIKE '2020%'"
```
===> End of Example

"""


# 🧠 Inject system instructions directly to the LLM (i took out the instructions)
# llm_with_prompt = ChatOllama(
#     base_url=ollama_endpoint,
#     model=ollama_model,
#     temperature=0,
#     enable_thinking=False,
#     )
llm_with_prompt = llm  # Use OpenAI LLM

# Setup agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm_with_prompt)
agent = create_sql_agent(
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=llm_with_prompt, 
    toolkit=toolkit,
    top_k=30, 
    verbose=True,
    handle_parsing_errors=True,
    )


#QUESTION = (
#    "what is the highest average salary by department, and give me the number?"
#)


# Run query
#result = agent.invoke(QUESTION)
#print("\n🎯 Final Answer:\n", result)



st.title("SQL Query AI Agent")

question = st.text_input("Enter your question:")

if st.button("Run Query"):
    if question:
        # Run the query
        result = agent.invoke(question)
        
        st.markdown(f"🎯 Final Answer:\n{result['output']}")
else:
    st.error("Please enter a query.")
