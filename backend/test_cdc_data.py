import logging
from agents.snowflake_agent import SnowflakeAgent
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 20)       # Show reasonable number of rows

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_table_info(df: pd.DataFrame, table_name: str):
    """Print detailed information about a table"""
    print(f"\n=== Table: {table_name} ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
    print("\nSample Data:")
    print(df.head(5).to_string())
    print("\n" + "="*50)

def main():
    try:
        # Initialize Snowflake agent
        logger.info("Initializing Snowflake agent...")
        agent = SnowflakeAgent()

        # Get all tables from CDC schema
        logger.info("Fetching CDC schema tables...")
        tables = agent.get_all_cdc_tables()
        
        if not tables:
            logger.warning("No tables found in CDC schema or using mock data")
            return

        print(f"\nFound {len(tables)} tables in CDC schema:")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")

        # Get data from each table
        for table in tables:
            logger.info(f"Fetching data from {table}...")
            df = agent.get_table_data(table)
            print_table_info(df, table)

        # Close connection
        agent.close_connection()
        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main() 