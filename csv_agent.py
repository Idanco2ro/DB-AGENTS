import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
import contextlib
from io import StringIO
import streamlit as st

# Load environment variables
load_dotenv()
ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
ollama_model = os.getenv("OLLAMA_MODEL1")

# Load and normalize the DataFrame
csv_path = r'D:\My_project\DB-AGENTS\Data\salaries_2023.csv'
df = pd.read_csv(csv_path)

def normalize_column_names(df):
    """
    Normalize column names to lowercase with underscores and remove special characters.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w\s]', '', regex=True)
    )
    return df

df = normalize_column_names(df)

# Gender normalization with regex
def map_gender(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if re.match(r'^f(emale)?$', value):
        return 'Female'
    elif re.match(r'^m(ale)?$', value):
        return 'Male'
    return None

df['gender'] = df['gender'].apply(map_gender)

# Display how many rows were dropped
invalid_gender_rows = df['gender'].isna().sum()

# Define a safe Python executor with better error handling
@contextlib.contextmanager
def capture_output():
    new_out = StringIO()
    with contextlib.redirect_stdout(new_out):
        yield new_out

def safe_python_executor(code: str) -> str:
    try:
        code = code.strip().strip("`")
        with capture_output() as out:
            local_vars = {"df": df, "pd": pd}
            exec(code, {}, local_vars)
        return out.getvalue().strip()
    except NameError as ne:
        return f"Execution error (NameError): {ne}. Did you forget to define this variable?"
    except Exception as e:
        return f"Execution error: {e}"

# Tool list
tools = [
    Tool(
        name="Python Executor",
        func=safe_python_executor,
        description="Executes Python code on the 'df' DataFrame to compute and return numeric results. Use this tool for any data analysis or computation tasks."
    )
]

# Connect to Ollama LLM
llm = ChatOllama(
    base_url=ollama_endpoint,
    model=ollama_model,
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)

# Updated CSV prompt template with Claude-style discipline
CSV_PROMPT_TEMPLATE = """
You are a professional data analyst working with a pandas DataFrame named 'df'.

Follow these strict rules when answering:

1. Start your reasoning inside a <think>...</think> block.
2. Use the Python Executor tool ONLY if exact numeric values are required.
3. Every Python code block must:
   - Be wrapped in triple backticks: ```python ... ```
   - Define all variables explicitly (e.g., male_avg, female_avg), even if used in reasoning before.
   - Use print() for all results, formatted with commas and two decimals.
4. Never assume variables are available unless defined in the same code block.
5. Perform any key comparison twice using different methods and explain it.
6. After all calculations, output the final result clearly and concisely and explain how you get to that result Humanly so that i can understand.

Question: {question}
"""

# LangGraph agent setup
agent_executor = create_react_agent(llm_with_tools, tools, prompt=CSV_PROMPT_TEMPLATE)

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    next: str

graph = StateGraph(AgentState)

def run_agent(state: AgentState) -> AgentState:
    response = agent_executor.invoke({"messages": state["messages"]})
    assistant_message = response["messages"][-1].content
    state["messages"].append({"role": "assistant", "content": assistant_message})
    state["next"] = "end"

    # Display reasoning
    print("\n🧠 Agent's Reasoning:\n")
    print(assistant_message)

    # Execute all code blocks safely
    code_blocks = re.findall(r"```python(.*?)```", assistant_message, re.DOTALL)
    full_output = []
    for idx, code in enumerate(code_blocks, 1):
        print(f"\n🔧 Executing Code Block {idx}:\n{code.strip()}\n")
        output = safe_python_executor(code)
        print(f"📊 Output of Code Block {idx}:\n{output}\n")
        full_output.append(f"🔧 Code Block {idx} Output:\n{output}")

    if full_output:
        state["messages"].append({
            "role": "assistant",
            "content": "\n\n".join(full_output)
        })

    return state

graph.add_node("agent", run_agent)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
app = graph.compile()

# Streamlit UI
st.title("📊 Database AI Agent with Ollama + LangChain")

st.write("### 🧾 Dataset Preview")
st.dataframe(df.head())

# Warn if gender rows were dropped
if invalid_gender_rows > 0:
    st.warning(f"{invalid_gender_rows} rows had unrecognized gender and were excluded.")

# User input
st.write("### ❓ Ask a Question")
user_question = st.text_input(
    "Enter your question about the dataset:",
    "Which grade has the highest average base salary, and compare the average female pay with male pay?"
)

if st.button("Run Query") and user_question:
    QUERY = CSV_PROMPT_TEMPLATE.format(question=user_question)
    initial_state = {
        "messages": [{"role": "user", "content": QUERY}],
        "next": "agent"
    }
    with st.spinner("Processing your query..."):
        try:
            result = app.invoke(initial_state)
            final_answer = result["messages"][-1]["content"]
            st.write("### ✅ Final Answer")
            st.markdown(final_answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

