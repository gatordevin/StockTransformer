# Project Running Instructions

This document outlines the steps and parameters required to run the project successfully.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setting Up Environment Variables](#setting-up-environment-variables)
3. [Installing Required Python Packages](#installing-required-python-packages)
4. [Step 1: Run `stockdata.py`](#step-1-run-stockdatapy)
5. [Step 2: Run `data.py`](#step-2-run-datapy)
6. [Step 3: Run `train.py`](#step-3-run-trainpy)
7. [Additional Notes](#additional-notes)

---

## Prerequisites

Before running the scripts, ensure you have the following:
- Access to an Alpaca API key and secret (if required).
- Python installed on your system.
- A virtual environment or a dedicated Python environment.

---

## Setting Up Environment Variables

To configure the environment for running the scripts:

### On Unix/Linux/MacOS
1. Export your Alpaca API key and secret:
   ```bash
   export ALPACA_API_KEY="your_api_key"
   export ALPACA_API_SECRET="your_api_secret"
   ```

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On MacOS or Linux
   ```
   
### On Windows
1. **Set your Alpaca API key and secret as environment variables using the `setx` command:**
   ```bash
   setx ALPACA_API_KEY "your_api_key"
   setx ALPACA_API_SECRET "your_api_secret"  # Optional, but recommended for security
   ```

2. **Create a Python virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Activates the virtual environment
   ```

## Installing Required Python Packages

Install the necessary packages using pip:
```bash
pip install -r requirements.txt
```

This will install all dependencies listed in `requirements.txt`.

---

## Step 1: Run `stockdata.py`

The `stockdata.py` script performs data retrieval and preprocessing. Run it from your project directory:

```bash
python stockdata.py
```

### What to Expect:
- **Data Retrieval**: Fetches latest data from Alpaca.
- **Preprocessing**: Cleans and prepares the data for analysis.
- **Statistics Generation**: Outputs key metrics.

---

## Step 2: Run `data.py`

The `data.py` script visualizes the dataset statistics. Run it after `stockdata.py`:

```bash
python data.py
```

### What to Expect:
- **Visualization**: Generates graphs and charts showing statistical properties.
- **Verification**: Confirms proper normalization and scaling of data.

---

## Step 3: Run `train.py`

The `train.py` script trains a machine learning model using the preprocessed data. Run it after `data.py`:

```bash
python train.py
```

### What to Expect:
- **Model Training**: Trains the model using the dataset.
- **Progress Logs**: Displays training metrics like loss and accuracy.
- **Model Saving**: Saves the trained model for future use.

---

## Additional Notes

### Configuration
Ensure all configuration files (e.g., `config.yaml`) are properly set up before running the scripts. Incorrect
configurations may cause errors.

### Environment Variables
- **ALPACA_API_KEY**: Set to your Alpaca API key.
- **ALPACA_API_SECRET**: Set if you have an API secret.

### Error Handling
Review the logs for any errors during execution and troubleshoot accordingly.