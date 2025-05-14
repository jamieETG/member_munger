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

## Sample Input Data

The application is designed to work with various CSV input files. Several examples are provided in the `test_files/` directory:
* `input_data_perfect_match.csv`: Columns exactly match the standard format.
* `input_data_messy_original.csv`: Original simple test file with a few differing column names.
* `input_data_messy_1.csv`: Columns have slightly different names, casing, and some are missing or reordered.
* `input_data_messy_2.csv`: Columns have significantly different names, different order, extra columns, and some standard columns are missing. 