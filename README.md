# Member Munger

This application processes health plan enrollment data from a source CSV file, 
attempts to map columns to a standard format defined in `rules.yaml`, 
cleans and standardizes the data, and outputs it to an Excel file. 

## Key Files

*   **Main Logic**: [`main.py`](./main.py) - The primary application logic.
*   **Configuration**: [`rules.yaml`](./rules.yaml) - Defines the standard output format, column names, and data types. Critical for mapping and standardizing input data.
*   **Dependencies**: [`requirements.txt`](./requirements.txt) - Lists Python packages required to run the project.
*   **Input Data Examples**: See the section below for descriptions of sample CSV files like `input_data_perfect_match.csv`, `input_data_messy_original.csv`, etc. These are used to test parsing and transformation. (Note: Ensure these files are present in your data directory).
*   **This Document**: [`README.md`](./README.md) - Provides an overview of the project.

## Setup and Installation

Follow these steps to get the Member Munger application running on your local machine. These instructions assume you have some technical familiarity but may be new to Python development.

### 1. Prerequisites

*   **Python**: Ensure you have Python installed. Version 3.8 or higher is recommended. You can download it from [python.org](https://www.python.org/downloads/).
    *   During installation on Windows, make sure to check the box that says "Add Python to PATH".
*   **pip**: Python's package installer, `pip`, is usually installed automatically with Python. You can verify by opening a terminal or command prompt and typing `pip --version`.

### 2. Clone the Repository

Get a copy of the project files. If you have Git installed, you can clone the repository using the following command in your terminal:

```bash
git clone <repository_url>
cd member_munger
```
(Replace `<repository_url>` with the actual URL of this repository).

If you don't have Git, you can usually download the project as a ZIP file from the repository page and then extract it.

### 3. Set Up a Virtual Environment (Recommended)

It's a good practice to create a virtual environment for each Python project to manage dependencies separately.

*   Navigate to the project directory in your terminal (`member_munger` if you cloned it).
*   Create a virtual environment. Common commands are:

    ```bash
    python -m venv venv
    ```
    or
    ```bash
    python3 -m venv venv
    ```

*   Activate the virtual environment:
    *   **Windows (Command Prompt/PowerShell):**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux (bash/zsh):**
        ```bash
        source venv/bin/activate
        ```
    You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

### 4. Install Dependencies

With the virtual environment active, install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Running the Application

Once everything is set up, you can run the main application script. The basic command structure will be:

```bash
python main.py <input_csv_path> <output_excel_path>
```

For example:

```bash
python main.py ./test_files/input_data_perfect_match.csv ./output/perfect_match_output.xlsx
```

Make sure the output directory (e.g., `./output/`) exists, or modify the script to create it if it doesn't.

Refer to the "Sample Input Data" section below for more examples of input files.

## Sample Input Data

The application is designed to work with various CSV input files. Several examples are provided in the `test_files/` directory:
* `input_data_perfect_match.csv`: Columns exactly match the standard format.
* `input_data_messy_original.csv`: Original simple test file with a few differing column names.
* `input_data_messy_1.csv`: Columns have slightly different names, casing, and some are missing or reordered.
* `input_data_messy_2.csv`: Columns have significantly different names, different order, extra columns, and some standard columns are missing. 