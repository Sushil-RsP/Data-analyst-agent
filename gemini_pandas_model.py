import os
import ast
import pandas as pd

from dotenv import load_dotenv
import google.generativeai as genai



load_dotenv()

api_key = os.getenv("MY_API_KEY")

def get_schema_context(df):
    
    return f"""
Columns:
{list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Shape:
{df.shape}

Sample Rows:
{df.head(5).to_string()}
"""


def generate_pandas_code(question, df):
    
    schema = get_schema_context(df)

    prompt = f"""
You are a senior data analyst.

You are given a pandas DataFrame called df.

Dataset Information:

{schema}

User Question:
{question}

Rules:
1. Return ONLY valid pandas expression code.
2. Do NOT explain.
3. Do NOT use markdown.
4. Do NOT import anything.
5. DataFrame variable name is df.
6. Return a single expression that can be evaluated.

Examples:

Question:
What is the maximum salary?

Answer:
df["Salary"].max()

Question:
Which employee has highest salary?

Answer:
df.loc[df["Salary"].idxmax()]
"""
    
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
    "gemini-2.5-flash"
    )

    response = model.generate_content(prompt)

    return response.text.strip()


BLACKLIST = [
    "import",
    "open",
    "exec",
    "eval",
    "__",
    "os.",
    "subprocess"
]

def validate(code):
    return not any(word in code for word in BLACKLIST)


def execute_code(code, df):
    if not validate(code):
        raise ValueError("Unsafe code detected")

    return eval(
        code,
        {"df": df, "pd": pd},
        {}
    )


"""def format_result(result):

    if isinstance(result, pd.DataFrame):
        return result.to_string()

    if isinstance(result, pd.Series):
        return result.to_string()

    return str(result)"""
    
    
def format_result(result):

    if isinstance(result, pd.DataFrame):
        return result.to_html(
            classes="table table-striped table-bordered",
            index=False
        )

    if isinstance(result, pd.Series):
        return result.reset_index().to_html(
            classes="table table-striped table-bordered",
            index=False
        )

    return str(result)