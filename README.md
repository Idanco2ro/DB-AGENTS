# DB-AGENTS Project Documentation

## Overview

This project demonstrates various approaches to building AI-powered agents that interact with structured data (SQL databases and CSV files) using modern LLMs (OpenAI, Ollama, LangChain, and OpenAI Assistants API). The goal is to enable natural language querying and analysis of salary data, leveraging both direct SQL execution and function-calling paradigms.

## Project Structure

- **sql_db_agent.py**:  
  Uses LangChain's SQL agent toolkit to allow users to ask questions about a SQLite database via a Streamlit UI. The agent generates SQL queries, executes them, and explains the results in Markdown.

- **csv_agent.py**:  
  Provides a data analysis agent for a CSV file using pandas and LangChain/Ollama. The agent can execute Python code on the DataFrame to answer user questions, with strict prompt discipline for reproducibility and clarity.

- **fun_calling.py**:  
  Demonstrates OpenAI's function calling with a simple weather example, showing how LLMs can call backend functions based on user queries.

- **fun_call_db_agent.py**:  
  Extends function calling to the salary database, mapping user questions to Python functions that execute SQL queries and return structured results.

- **assis_api_sql_db.py**:  
  Explores the OpenAI Assistants API for persistent, multi-turn conversations with tool/function calling, integrating with the salary database.

- **helpers.py**:  
  Contains Python functions that encapsulate common SQL queries on the salary database, used by the function-calling agents.

- **first_agent.py**:  
  A simple conversational agent using Ollama, demonstrating basic LLM interaction.

- **requirements.txt / pyproject.toml**:  
  Dependency management for Python packages.

## Why These Approaches?

- **Natural Language to SQL**:  
  Many users are not SQL experts. By using LLMs to translate questions into SQL, we democratize access to structured data.

- **Function Calling**:  
  LLMs can call backend functions for complex or sensitive operations, ensuring correctness and security (e.g., only allowing certain queries).

- **Prompt Engineering**:  
  Strict prompt templates (especially in `csv_agent.py`) ensure reproducibility, clarity, and safety when executing code.

- **Streamlit UI**:  
  Provides an accessible web interface for users to interact with the agents.

- **Multiple LLM Backends**:  
  Supports both OpenAI and Ollama for flexibility and cost control.

## How to Use

1. **Set up environment variables** in a `.env` file (API keys, model names, etc.).
2. **Install dependencies** using `pip install -r requirements.txt` or via `pyproject.toml`.
3. **Prepare the data**: Place the salary CSV in the expected location.
4. **Run the desired agent**:
   - For SQL agent UI: `python sql_db_agent.py`
   - For CSV agent UI: `python csv_agent.py`
   - For function-calling demos: `python fun_calling.py` or `python fun_call_db_agent.py`
   - For OpenAI Assistants API: `python assis_api_sql_db.py`

## Security and Safety

- All code execution is sandboxed and restricted to the provided DataFrame.
- SQL agents are instructed not to perform DML operations (no INSERT/UPDATE/DELETE).
- Function-calling agents only expose whitelisted functions.

## Future Directions

- Add authentication and user management.
- Expand the set of supported queries and functions.
- Integrate with more advanced LLMs or local models.
- Improve error handling and user feedback.

---
