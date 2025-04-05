import logging
from agents.snowflake_agent import SnowflakeAgent
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_statistics(stats: dict):
    """Print statistics in a readable format"""
    print("\n=== Cancer Statistics ===")
    
    if 'historical' in stats:
        print("\nHistorical Data:")
        print(stats['historical'].to_string())
    
    if 'regional' in stats:
        print("\nRegional Data:")
        print(stats['regional'].to_string())
    
    if 'cancer_types' in stats:
        print("\nCancer Types Data:")
        print(stats['cancer_types'].to_string())

def print_treatment_costs(costs):
    """Print treatment costs in a readable format"""
    print("\n=== Treatment Costs ===")
    if costs is not None:
        print(costs.to_string())
    else:
        print("No treatment cost data available")

def main():
    try:
        # Initialize Snowflake agent
        logger.info("Initializing Snowflake agent...")
        agent = SnowflakeAgent()

        # Get cancer statistics
        logger.info("Fetching cancer statistics...")
        stats = agent.get_cancer_statistics()
        print_statistics(stats)

        # Get treatment costs
        logger.info("Fetching treatment costs...")
        costs = agent.get_treatment_costs()
        print_treatment_costs(costs)

        # Get visualizations
        logger.info("Generating visualizations...")
        visualizations = agent.get_visualizations()
        print(f"\nNumber of visualizations generated: {len(visualizations)}")

        # Check if using mock data
        print(f"\nUsing mock data: {agent.use_mock_data}")

        # Close connection
        agent.close_connection()
        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main() 