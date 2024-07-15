# VAR Automation for SRC

This is a React-based web application designed to automate the Value at Risk (VAR) calculation for SRC. The application allows users to input financial tickers, group them into various categories, and specify parameters for the VAR calculation.

## Features

- **Ticker Input**: Add and manage financial tickers.
- **Group Management**: Create and manage groups of tickers, specifying sizes and types.
- **Parameter Input**: Specify periods and end dates for VAR calculations.
- **Excel Upload**: Automatically populate tickers and groups from an Excel sheet.
- **Validation**: Ensure all necessary fields are filled before generating the output.
- **Backend Integration**: Processes the inputs to perform VAR calculations using a Python backend.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd var-automation
    ```
3. Install dependencies:
    ```sh
    npm install
    ```

## Usage

1. Start the development server:
    ```sh
    npm run dev
    ```
2. Open the app in your browser at `http://localhost:3000`.

## Components

### TickerInput

Allows users to add and manage financial tickers.

### GroupCard

Allows users to create and manage groups of tickers, specifying sizes and types.

### ParametersInput

Allows users to specify periods and end dates for VAR calculations.

### ExcelUpload

Allows users to upload an Excel sheet to automatically populate tickers and groups.

## Example

1. **Add Tickers**: Input financial tickers and add them to the list.
2. **Create Groups**: Create groups, specify sizes, types, and assign tickers to each group.
3. **Specify Parameters**: Enter periods and end date for the VAR calculations.
4. **Upload Excel Sheet**: (Optional) Use the Excel upload feature to automatically populate tickers and groups.
5. **Generate Output**: Click the generate button to process the inputs and perform VAR calculations.

## Backend Integration

The backend integration is handled via a Python script that processes the input and performs the VAR calculations. The results are logged and can be saved to a file.

## .gitignore

The `.gitignore` file ensures that `node_modules` and other unnecessary files are not tracked by Git.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.