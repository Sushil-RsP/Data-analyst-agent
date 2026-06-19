# Data Analyst AI Agent

An AI-powered Data Analyst Agent built using Flask, Pandas, Matplotlib, Seaborn, and Google Gemini. The application allows users to upload a dataset, perform automated exploratory data analysis (EDA), clean data, generate visualizations, and ask questions in natural language.

## Features

### Automated Dataset Analysis

After uploading a dataset, the system automatically:

* Displays the first 5 rows of the dataset (df.head())
* Shows dataset shape (rows and columns)
* Lists all column names
* Detects data types for each column
* Generates descriptive statistics using df.describe()
* Performs basic Exploratory Data Analysis (EDA)
* Identifies missing values
* Performs automatic data cleaning
* Fills missing values using suitable methods

### AI-Powered Querying

Users can ask questions about the uploaded dataset in natural language.

Examples:

* What is the average salary?
* Which category has the highest sales?
* Show top 10 products by revenue.
* Find missing values in the dataset.

Google Gemini generates Pandas code dynamically to answer user queries.

### Interactive Visualization Generator

Users can:

1. Select a chart type
2. Select columns
3. Click Generate Chart

Supported charts:

* Histogram
* Box Plot
* Distribution Plot
* Scatter Plot
* Bar Chart
* Count Plot
* Pie Chart
* Correlation Heatmap

The system automatically identifies:

* Numerical columns
* Categorical columns

and provides appropriate column options for each chart type.

## Technology Stack

### Backend

* Python
* Flask

### Data Analysis

* Pandas
* NumPy

### AI

* Google Gemini API

### Visualization

* Matplotlib
* Seaborn

### Frontend

* HTML
* CSS
* JavaScript

## Project Structure

```text
Data-analyst-agent/
│
├── new_app_py.py
├── new_analysis_tools.py
├── gemini_pandas_model.py
├── requirements.txt
│
└── templates/
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Sushil-RsP/Data-analyst-agent.git
cd Data-analyst-agent
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Key

Create a .env file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
python app_py.py
```

Open:

```text
http://127.0.0.1:5000
```

## Supported Dataset Formats

* CSV

## Future Improvements

* Excel (.xlsx) support
* PDF report generation
* Dashboard export
* Advanced business insights
* Multi-file analysis
* SQL database support
* Local LLM support

## Author

Sushil Chavan

GitHub: https://github.com/Sushil-RsP

Portfolio: https://sushil-rsp.github.io/Portfolio_/
