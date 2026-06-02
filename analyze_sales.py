"""
Sales Data Analysis using Gemini API
Author: mingitdynasty@gmail.com
Analyzes sales_data_sample.csv using Google's Gemini AI
"""

#import google.genai as genai
#from google import genai
import google.generativeai as genai
import os
import pandas as pd
from typing import Optional
import json


def configure_gemini(api_key: Optional[str] = None):
    """
    Configure the Gemini API with your API key.
    
    Args:
        api_key: Your Gemini API key. If not provided, will look for GEMINI_API_KEY environment variable.
    """
    #api_key = os.environ.get('AIzaSyCtitdwRnnReRwjx5E1jJe5v70TH2uklJQ')
    if api_key is None:
        api_key = os.getenv('')
        api_key = os.environ.get('GEMINI_API_KEY')
        os.environ["GEMINI_API_KEY"] = ""
        genai.api_key = ''
        if api_key is None:
            raise ValueError(
                "API key not found. Please provide it as an argument or set GEMINI_API_KEY environment variable.\n"
                "Get your API key from: https://aistudio.google.com/app/apikey"
           )
    
    genai.configure(api_key="GEMINI_API_KEY")


def load_sales_data(filepath: str = "sales_data_sample.csv") -> pd.DataFrame:
    """
    Load the sales data from CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame containing the sales data
    """
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"✓ Successfully loaded {len(df)} records from {filepath} (encoding: {encoding})")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            if encoding == encodings[-1]:  # Last encoding attempt
                raise Exception(f"Error loading CSV: {e}")
            continue
    
    raise Exception(f"Could not decode {filepath} with any of the attempted encodings: {encodings}")


def get_data_summary(df: pd.DataFrame) -> str:
    """
    Generate a summary of the sales data for Gemini.
    
    Args:
        df: Sales DataFrame
    
    Returns:
        String representation of data summary
    """
    summary = {
        "total_records": len(df),
        "date_range": f"{df['ORDERDATE'].min()} to {df['ORDERDATE'].max()}",
        "total_sales": f"${df['SALES'].sum():,.2f}",
        "avg_order_value": f"${df['SALES'].mean():,.2f}",
        "unique_customers": df['CUSTOMERNAME'].nunique(),
        "unique_products": df['PRODUCTCODE'].nunique(),
        "product_lines": df['PRODUCTLINE'].unique().tolist(),
        "countries": df['COUNTRY'].unique().tolist(),
        "order_statuses": df['STATUS'].unique().tolist(),
        "columns": df.columns.tolist()
    }
    
    return json.dumps(summary, indent=2)


def analyze_with_gemini(df: pd.DataFrame, query: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Analyze sales data using Gemini AI.
    
    Args:
        df: Sales DataFrame
        query: Analysis query/question
        model_name: Gemini model to use
    
    Returns:
        Analysis results from Gemini
    """
    # Prepare data summary
    data_summary = get_data_summary(df)
    
    # Get sample of the data (first 10 rows)
    sample_data = df.head(10).to_string()
    
    # Construct the prompt
    prompt = f"""You are a sales data analyst. I have a sales dataset with the following characteristics:

DATA SUMMARY:
{data_summary}

SAMPLE DATA (first 10 rows):
{sample_data}

USER QUERY:
{query}

Please provide a detailed analysis based on the data summary and characteristics. Focus on insights, trends, and actionable recommendations. If the query requires specific calculations, provide those based on the summary statistics provided."""
    
    # Send to Gemini
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def detailed_analysis_with_stats(df: pd.DataFrame, model_name: str = "gemini-2.5-flash") -> str:
    """
    Perform a comprehensive analysis with detailed statistics.
    
    Args:
        df: Sales DataFrame
        model_name: Gemini model to use
    
    Returns:
        Comprehensive analysis from Gemini
    """
    # Calculate detailed statistics
    stats = {
        "Sales Overview": {
            "Total Sales": f"${df['SALES'].sum():,.2f}",
            "Average Order": f"${df['SALES'].mean():,.2f}",
            "Median Order": f"${df['SALES'].median():,.2f}",
            "Min Order": f"${df['SALES'].min():,.2f}",
            "Max Order": f"${df['SALES'].max():,.2f}"
        },
        "Top 5 Customers": df.groupby('CUSTOMERNAME')['SALES'].sum().nlargest(5).to_dict(),
        "Sales by Product Line": df.groupby('PRODUCTLINE')['SALES'].sum().to_dict(),
        "Sales by Country": df.groupby('COUNTRY')['SALES'].sum().nlargest(10).to_dict(),
        "Sales by Year": df.groupby('YEAR_ID')['SALES'].sum().to_dict(),
        "Sales by Quarter": df.groupby('QTR_ID')['SALES'].sum().to_dict(),
        "Order Status Distribution": df['STATUS'].value_counts().to_dict(),
        "Deal Size Distribution": df['DEALSIZE'].value_counts().to_dict()
    }
    
    # Format stats as string
    stats_str = json.dumps(stats, indent=2, default=str)
    
    prompt = f"""As a senior sales data analyst, please provide a comprehensive analysis of this sales dataset:

DETAILED STATISTICS:
{stats_str}

Please provide:
1. Key Performance Insights
2. Trend Analysis
3. Customer Behavior Patterns
4. Product Performance Analysis
5. Geographic Market Insights
6. Actionable Recommendations

Format the response in a clear, professional manner with specific numbers and percentages where applicable."""
    
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def interactive_analysis_mode(df: pd.DataFrame):
    """
    Run an interactive analysis session where users can ask questions about the sales data.
    
    Args:
        df: Sales DataFrame
    """
    print("\n=== Interactive Sales Data Analysis ===")
    print("Ask questions about the sales data. Type 'quit' or 'exit' to end.\n")
    print("Example questions:")
    print("  - What are the top selling product lines?")
    print("  - Which customers contribute most to revenue?")
    print("  - What are the sales trends over time?")
    print("  - Which countries have the highest sales?\n")
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Prepare context for conversation
    data_summary = get_data_summary(df)
    
    # Calculate some key stats for context
    top_products = df.groupby('PRODUCTLINE')['SALES'].sum().nlargest(5).to_dict()
    top_customers = df.groupby('CUSTOMERNAME')['SALES'].sum().nlargest(5).to_dict()
    
    context = f"""You are analyzing a sales dataset with these characteristics:

{data_summary}

Top 5 Product Lines by Sales:
{json.dumps(top_products, indent=2, default=str)}

Top 5 Customers by Sales:
{json.dumps(top_customers, indent=2, default=str)}

Please answer questions about this sales data based on the provided context."""
    
    # Start chat with context
    chat = model.start_chat(history=[
        {"role": "user", "parts": [context]},
        {"role": "model", "parts": ["I understand. I'm ready to analyze this sales dataset and answer your questions about sales trends, customer behavior, product performance, and geographic insights. What would you like to know?"]}
    ])
    
    while True:
        user_input = input("Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Analysis session ended. Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = chat.send_message(user_input)
            print(f"\n{response.text}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """
    Main function to run sales data analysis.
    """
    print("=== Sales Data Analysis with Gemini AI ===\n")
    
    # Configure Gemini API
    try:
        configure_gemini()
        print("✓ Gemini API configured successfully\n")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    
    # Load sales data
    try:
        df = load_sales_data()
        print(f"  - Columns: {', '.join(df.columns.tolist()[:5])}... ({len(df.columns)} total)")
        print(f"  - Date range: {df['ORDERDATE'].min()} to {df['ORDERDATE'].max()}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Menu options
    while True:
        print("\n=== Analysis Options ===")
        print("1. Comprehensive Analysis Report")
        print("2. Ask a Custom Question")
        print("3. Interactive Analysis Mode")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            print("\nGenerating comprehensive analysis report...\n")
            try:
                analysis = detailed_analysis_with_stats(df)
                print("="*80)
                print(analysis)
                print("="*80)
            except Exception as e:
                print(f"Error during analysis: {e}")
        
        elif choice == '2':
            query = input("\nEnter your question: ").strip()
            if query:
                print("\nAnalyzing...\n")
                try:
                    result = analyze_with_gemini(df, query)
                    print("="*80)
                    print(result)
                    print("="*80)
                except Exception as e:
                    print(f"Error during analysis: {e}")
        
        elif choice == '3':
            interactive_analysis_mode(df)
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()
