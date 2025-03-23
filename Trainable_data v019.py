# =============================================================================
# IMPORTS
# =============================================================================

# Standard Library Imports
import os
import re
import json
import logging
from datetime import datetime, timedelta

# Third-Party Library Imports
import pandas as pd
import fitz  # PyMuPDF for reading PDFs
import yfinance as yf
from tqdm import tqdm  # For progress bars
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import zipfile
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# DIRECTORIES
# =============================================================================

# Base directories
try:
    # Get program directory and calculate base directory
    program_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(program_dir, "../../../08 Models"))
except NameError:
    # Fallback if __file__ is not available (e.g., in interactive sessions)
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Main directories
opt_dir = os.path.join(base_dir, "01 Scripts", "07 OPT Model", "OPT")
files_dir = os.path.join(base_dir, "02 Data")

# Data subdirectories
articles_dir = os.path.join(files_dir, "01 Articles", "01 Reliance", "2020", "01 Common", "01 Zipped")
stocks_dir = os.path.join(files_dir, "02 Stocks")
log_dir = os.path.join(files_dir, "03 Reports", "04 OPT")

# Training data directories
train_data = os.path.join(files_dir, "06 Trainable data", "generic trainble data")
synthetic_dir = os.path.join(files_dir, "06 Trainable data", "Synthetic Instructions")
opt_tokenized_dir = os.path.join(files_dir, "06 Trainable data", "OPT tokenized data")

# =============================================================================
# CONSTANTS & FILE PATHS
# =============================================================================

STOCK_SYMBOL = "RELIANCE.NS"
STOCK_DIR = os.path.join(stocks_dir, "01 Reliance")
CSV_FILE = os.path.join(STOCK_DIR, "RELIANCE.NS_2020.csv")
CACHE_FILE = "stock_beta_cache.json"

# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

def load_cache():
    """Load the cached beta values from a JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Cache file {CACHE_FILE} is corrupted. Using an empty cache.")
    return {}

def save_cache(cache):
    """Save the beta values to a JSON cache file."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")

# =============================================================================
# STOCK BETA FUNCTION
# =============================================================================

def get_beta_value(stock_symbol):
    """
    Retrieve the beta value of a stock. Uses caching to avoid repeated API calls.
    
    Parameters:
        stock_symbol (str): Stock ticker symbol.
    
    Returns:
        float: Beta value of the stock.
    """
    print("\n\n***** Retrieving Stock Beta Value *****")
    cache = load_cache()
    if stock_symbol in cache:
        logging.info(f"Using cached beta value for {stock_symbol}.")
        return cache[stock_symbol]

    try:
        ticker = yf.Ticker(stock_symbol)
        beta = ticker.info.get("beta", 0.0)  # Default to 0.0 if beta is not available
        cache[stock_symbol] = beta
        save_cache(cache)
        return beta
    except Exception as e:
        logging.error(f"Failed to fetch beta for {stock_symbol}: {e}")
        return 0.0  # Fallback value
    print("\nStock Beta Value Retrived.\n")

# Retrieve and display beta value
beta_value = get_beta_value(STOCK_SYMBOL)
print(f"\nBeta value is: {beta_value}")
logging.info(f"Beta for {STOCK_SYMBOL}: {beta_value}")

# =============================================================================
# STOCK DATA LOADING FUNCTION
# =============================================================================

def load_stock_data(csv_path):
    """
    Load stock data from a CSV file into a Pandas DataFrame.
    
    - Skips the first three rows (assumed extra headers).
    - Assigns appropriate column names.
    
    Parameters:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame or None: Stock data DataFrame, or None if an error occurs.
    """
    print("\n\n***** Loading Stock Data from CSV *****\n")
    print(f"Processing CSV file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, skiprows=3, header=None)

        if df.shape[1] != 7:
            logging.warning(f"Expected 7 columns but found {df.shape[1]} in {csv_path}.")

        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Percentage Change"]
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        print(f"\nSuccessfully loaded data from {csv_path}.\n")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_path}: {e}")
        return None

# Load stock data
df_stock = load_stock_data(CSV_FILE)
print(f"{os.path.basename(CSV_FILE)} file loaded to df_stock variable.\n\n\n")

# =============================================================================
# STOCK PRICE LOOKUP FUNCTION (Using CSV data)
# =============================================================================

def get_stock_prices_from_csv(df, pub_date):
    """
    Given a DataFrame of stock data and a publication date (datetime), retrieve the closing price on:
      - the publication day (price_day0)
      - 3 days later (price_day3).
    Adjusts for weekends by moving to the next business day if necessary.
    """
    def adjust_for_weekend(date):
        """Adjust the date to the next business day if it's a weekend."""
        if date.weekday() == 5:  # Saturday
            return date + timedelta(days=2)
        elif date.weekday() == 6:  # Sunday
            return date + timedelta(days=1)
        return date

    # Adjust publication date to next business day if it's a weekend
    adjusted_pub_date = adjust_for_weekend(pub_date)
    pub_date_only = adjusted_pub_date.date()

    # Get price_day0 (adjusted to next business day if needed)
    df_match = df[df["Date"].dt.date == pub_date_only]
    if not df_match.empty:
        price_day0 = df_match["Close"].iloc[0]
    else:
        df_after = df[df["Date"].dt.date > pub_date_only]
        price_day0 = df_after["Close"].iloc[0] if not df_after.empty else None

    # Calculate day3_date (3 calendar days + weekend adjustment)
    day3_date = adjust_for_weekend(adjusted_pub_date + timedelta(days=3))
    day3_only = day3_date.date()

    # Get price_day3 (adjusted to next business day if needed)
    df_day3 = df[df["Date"].dt.date == day3_only]
    if not df_day3.empty:
        price_day3 = df_day3["Close"].iloc[0]
    else:
        df_after_day3 = df[df["Date"].dt.date > day3_only]
        price_day3 = df_after_day3["Close"].iloc[0] if not df_after_day3.empty else None

    return price_day0, price_day3

# =============================================================================
# HELPER FUNCTIONS (PDF Extraction & Date Extraction)
# =============================================================================

def extract_pdf_text(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
    return text

def extract_load_date(text):
    """
    Extract the publication date from the article text.
    Expects a date in the format 'Month Day, Year' (e.g., 'January 15, 2020').
    If a month abbreviation like 'Sept' is encountered, it is replaced with 'Sep'.
    """
    match = re.search(r"([A-Za-z\s]+?)\s*(\w+\s\d{1,2},\s?\d{4})", text)
    if match:
        date_str = match.group(2).strip()
        # Fix common abbreviation issue (e.g., 'Sept' -> 'Sep')
        try:
            publish_date = datetime.strptime(date_str, "%B %d, %Y")
            logging.info(f"Extracted Publish-Date: {publish_date}")
            return publish_date
        except ValueError as e:
            logging.warning(f"Error parsing date: {date_str}. Error: {e}")
            return None
    logging.warning("No valid date found in the text.")
    return None

import re

def clean_article_text(pdf_text):
    """
    Extracts the main article content from pdf_text by:
      1. Starting from the line after a "Body" marker, if available.
      2. Removing lines that match header/footer patterns (e.g., page numbers, "Load-Date:", "End of Document").
    """
    # Split text into lines
    lines = pdf_text.splitlines()
    
    # Define patterns for lines we want to exclude (headers/footers)
    header_footer_patterns = [
        r'^\s*Page\s+\d+\s+of\s+\d+\s*$',  # lines like "Page 1 of 2"
        r'^\s*Load[-\s]?Date:',           # lines starting with "Load-Date:" or "Load Date:"
        r'^\s*End of Document',           # lines indicating the end of document
        r'^\s*Copyright',                 # copyright notices
        r'^\s*Length:\s*\d+\s*words'       # the "Length:" line
    ]
    
    # If a "Body" marker exists, start processing from the line after it
    start_index = 0
    for i, line in enumerate(lines):
        if re.search(r'\bBody\b', line, re.IGNORECASE):
            start_index = i + 1
            break

    # Process the lines starting from start_index, filtering out header/footer lines.
    content_lines = [
        line for line in lines[start_index:]
        if not any(re.search(pattern, line, re.IGNORECASE) for pattern in header_footer_patterns)
    ]
    
    # Join the lines to form the cleaned text.
    return "\n".join(content_lines).strip()

# =============================================================================
# SENTIMENT ANALYSIS & SYNTHETIC INSTRUCTION (Placeholder Functions)
# =============================================================================

# Load OPT model and tokenizer for sentiment analysis
opt_model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(opt_model_name)
model = AutoModelForSequenceClassification.from_pretrained(opt_model_name)

def opt_sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    sentiment = "POSITIVE" if scores[1] > scores[0] else "NEGATIVE"
    return {"label": sentiment, "score": scores[1] if sentiment == "POSITIVE" else scores[0]}

def create_synthetic_instruction(article_text, symbol):
    return f"Predict whether {symbol} stock will rise or fall in the next 3 days after: {article_text[:100]}..."

# =============================================================================
# PROCESS ZIP FILES & BUILD JSON DATA
# =============================================================================

def process_zip_files(zip_files, df_stock):
    # Initialize datasets
    generic_entries = []
    tokenized_entries = []
    synthetic_instructions = []
    article_counter = 1

    # Load tokenizer once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    symbol = "RELIANCE.NS"
    beta_value = get_beta_value(symbol)

    def get_unique_filename(base_path, base_name):
        """Generate unique filename with incremental suffix and .jsonl extension"""
        counter = 1
        while True:
            suffix = f"_{counter:02d}" if counter > 1 else ""
            full_path = os.path.join(base_path, f"{base_name}{suffix}.jsonl")
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    def save_dataset(data, base_dir, base_name):
        """
        Save the dataset as a JSONL file where each record is written on a new line.
        """
        os.makedirs(base_dir, exist_ok=True)
        path = get_unique_filename(base_dir, base_name)
        with open(path, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
        return path

    for zip_file in tqdm(zip_files, desc="Processing ZIP files"):
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                for info in z.infolist():
                    if not info.filename.lower().endswith('.pdf'):
                        continue
                    try:
                        # Read PDF from zip into memory
                        with z.open(info) as f:
                            pdf_bytes = f.read()
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        pdf_text = "".join(page.get_text() for page in doc)
                    except Exception as e:
                        logging.error(f"Error reading {info.filename} in {zip_file}: {e}")
                        continue

                    pub_date = extract_load_date(pdf_text)
                    if pub_date is None:
                        logging.warning(f"No valid date found in {info.filename}. Skipping.")
                        continue

                    price_day0, price_day3 = get_stock_prices_from_csv(df_stock, pub_date)
                    if price_day0 is None or price_day3 is None:
                        logging.warning(f"Could not find stock prices for {info.filename} on {pub_date}. Skipping.")
                        continue

                    three_day_return = round(((price_day3 - price_day0) / price_day0) * 100, 2)
                    sentiment_result = opt_sentiment_analysis(pdf_text)
                    opt_sentiment_score = sentiment_result["score"]
                    
                    # Create binary labels based on thresholds:
                    sentiment_label = 1 if opt_sentiment_score > 0.60 else 0
                    return_label = 1 if three_day_return > 0 else 0
                    
                    processed_text = clean_article_text(pdf_text)
                    
                    # Create generic_entry
                    generic_entry = {
                        "article_id": f"IN{article_counter:04d}",
                        "date_published": pub_date.strftime("%Y-%m-%d"),
                        "stock_symbol": symbol,
                        "article_text": processed_text,
                        "price_day0": float(price_day0),
                        "price_day3": float(price_day3),
                        "3_day_return_pct": float(three_day_return),
                        "return_label": return_label
                    }
                    generic_entries.append(generic_entry)

                    # Create tokenized_entry
                    tokens = tokenizer.encode(
                        processed_text,
                        max_length=2048,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    tokenized_entry = {
                        **generic_entry,
                        "tokens": tokens[0].tolist(),
                        "article_text": None  # Remove redundant text
                    }
                    tokenized_entries.append(tokenized_entry)

                    # Create synthetic instruction entry
                    synthetic_instructions.append({
                        "instruction": f"Predict {symbol} stock movement",
                        "input": processed_text[:2048],
                        "answer": "rise" if return_label == 1 else "fall"
                    })

                    article_counter += 1

        except Exception as e:
            logging.error(f"Error processing {zip_file}: {e}")

    # --- Shuffle the datasets before saving ---
    random.shuffle(generic_entries)
    random.shuffle(tokenized_entries)
    random.shuffle(synthetic_instructions)

    # Save only the required datasets
    generic_path = save_dataset(generic_entries, train_data, "generic_data")
    tokenized_path = save_dataset(tokenized_entries, opt_tokenized_dir, "OPT_tokenized_data")
    synthetic_path = save_dataset(synthetic_instructions, synthetic_dir, "synthetic_instructions")

    logging.info(f"""
    Successfully saved:
    - Generic data: {generic_path}
    - Tokenized data: {tokenized_path}
    - Synthetic instructions: {synthetic_path}
    """)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Get list of ZIP files from the articles directory.
zip_files = [os.path.join(articles_dir, f) for f in os.listdir(articles_dir) if f.endswith('.zip')]
process_zip_files(zip_files, df_stock)
