# Member Munger

Member Munger is a web application designed to process health plan enrollment data. Users can upload a source CSV file, interactively map its columns to a standard format defined in `rules.yaml` (assisted by fuzzy matching, content analysis, and reusable templates), apply cleaning and standardization rules, and then preview the processed data within the application. The ultimate goal is to produce a standardized dataset suitable for further use, such as exporting to an Excel file.

## Key Files

*   **Main Logic**: [`main.py`](./main.py) - A [FastAPI](https://fastapi.tiangolo.com/) web application that serves the user interface and orchestrates the data processing workflow.
*   **Web Interface Templates**: `templates/` - Contains HTML templates (e.g., `index.html`) for the web UI, using [Jinja2](https://jinja.palletsprojects.com/).
*   **Configuration**: [`rules.yaml`](./rules.yaml) - Defines the standard output format (column names, data types), column name aliases, transformation rules, and reusable mapping templates. This file is critical for mapping and standardizing input data.
*   **Input Data Examples**: Sample input CSV files are provided in the `test_files/` directory (e.g., `input_data_perfect_match.csv`, `input_data_messy_1.csv`). These are used to test the application's parsing and transformation capabilities.
*   **Dependencies**: [`requirements.txt`](./requirements.txt) - Lists all Python package dependencies required to run the project (e.g., `fastapi`, `uvicorn`, `pandas`, `pyyaml`, `thefuzz`).
*   **Project Description**: [`README.md`](./README.md) - This document.

## Setup and Installation

Follow these steps to get the Member Munger web application running on your local machine.

### 1. Prerequisites

*   **Python**: Ensure you have Python installed. Version 3.8 or higher is recommended. You can download it from [python.org](https://www.python.org/downloads/).
    *   During installation on Windows, make sure to check the box that says "Add Python to PATH".
    *   Verify your installation: `python --version` or `python3 --version`.
*   **pip**: Python's package installer, usually installed with Python. Verify with `pip --version`.

### 2. Clone the Repository

Get a copy of the project files. If you have Git:
```bash
git clone <repository_url>
cd member_munger
```
(Replace `<repository_url>` with the actual URL). Otherwise, download and extract the ZIP file.

### 3. Set Up a Virtual Environment (Recommended)

Navigate to the project directory (`member_munger`) and create/activate a virtual environment:
```bash
# Create (e.g., named 'venv')
python -m venv venv 
# Activate
# Windows (Command Prompt/PowerShell):
.\venv\Scripts\activate
# macOS/Linux (bash/zsh):
source venv/bin/activate 
```
Your terminal prompt should now show `(venv)`. To deactivate later: `deactivate`.

### 4. Install Dependencies

With the virtual environment active, install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

Once setup is complete and your virtual environment is active:

1.  **Start the Web Server**:
    Navigate to the project's root directory (where `main.py` is located) in your terminal and run:
    ```bash
    uvicorn main:app --reload
    ```
    The `--reload` flag makes the server automatically update when you change code, which is handy for development. Uvicorn will typically tell you the address where the application is running (e.g., `http://127.0.0.1:8000`).

2.  **Access in Browser**:
    Open your web browser and go to the address provided by Uvicorn (usually `http://127.0.0.1:8000`).

## Workflow Overview

1.  **Upload**: Use the web interface to upload your member enrollment CSV file.
2.  **Map Columns**: The application will attempt to automatically map columns from your file to the standard fields defined in `rules.yaml`.
    *   It may use direct matches, normalized matches, aliases, fuzzy matching, or pre-configured file templates.
    *   The system will show you the proposed mappings and any unmapped columns.
    *   Content analysis may suggest mappings for unmapped columns or transformations (e.g., splitting a full name).
3.  **Adjust & Confirm**: Review the mappings. You can adjust the fuzzy matching threshold, manually change mappings, and approve or define transformations. You can also save your mapping configuration as a new template for future use.
4.  **Process & Preview**: Once you confirm the mappings, the application processes the data according to the rules and your choices. An HTML preview of the transformed data is displayed in the web interface.
5.  **(Next Steps)**: The standardized data previewed in the application is then ready for further use, which could include manually copying it or a future enhancement to directly download it as an Excel file.

**Troubleshooting Tips:**
*   **`ModuleNotFoundError`**: Ensure your virtual environment is active and dependencies were installed via `pip install -r requirements.txt`.
*   **Uvicorn errors**: Ensure `uvicorn` and `fastapi` are in `requirements.txt` and installed. Check the terminal output from Uvicorn for specific error messages.
*   **File Not Found (rules.yaml, templates)**: Make sure you are running `uvicorn` from the project's root directory where these files/folders are located.

## Sample Input Data

The application is designed to work with various CSV input files. Several examples are provided in the `test_files/` directory:
* `input_data_perfect_match.csv`: Columns exactly match the standard format.
* `input_data_messy_original.csv`: Original simple test file with a few differing column names.
* `input_data_messy_1.csv`: Columns have slightly different names, casing, and some are missing or reordered.
* `input_data_messy_2.csv`: Columns have significantly different names, different order, extra columns, and some standard columns are missing. 