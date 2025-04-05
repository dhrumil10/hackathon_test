import snowflake.connector
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SnowflakeAgent:
    def __init__(self):
        """Initialize Snowflake connection using environment variables"""
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            # Log environment variables (without sensitive info)
            logger.info("Checking Snowflake configuration...")
            logger.info(f"SNOWFLAKE_ACCOUNT: {'Set' if os.getenv('SNOWFLAKE_ACCOUNT') else 'Not Set'}")
            logger.info(f"SNOWFLAKE_USER: {'Set' if os.getenv('SNOWFLAKE_USER') else 'Not Set'}")
            logger.info(f"SNOWFLAKE_DATABASE: {'Set' if os.getenv('SNOWFLAKE_DATABASE') else 'Not Set'}")
            
            self.conn = self._get_snowflake_connection()
            if not self.conn:
                logger.warning("Failed to connect to Snowflake, using mock data")
                self.use_mock_data = True
            else:
                self.use_mock_data = False
                logger.info("Successfully connected to Snowflake")
        except Exception as e:
            logger.error(f"Error initializing Snowflake agent: {e}")
            self.use_mock_data = True

    def _get_snowflake_connection(self):
        """Establish Snowflake connection using credentials from environment variables"""
        try:
            return snowflake.connector.connect(
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                database=os.getenv('SNOWFLAKE_DATABASE'),
                schema=os.getenv('SNOWFLAKE_SCHEMA')
            )
        except Exception as e:
            st.error(f"Error connecting to Snowflake: {str(e)}")
            return None

    def get_cancer_statistics(self) -> Dict[str, pd.DataFrame]:
        """Retrieve cancer statistics from Snowflake"""
        try:
            if self.use_mock_data:
                logger.info("Using mock data for cancer statistics")
                return self._get_mock_statistics()
            
            stats = {}
            
            # Historical Statistics
            historical_query = """
            SELECT 
                YEAR,
                SUM(NEW_CASES) as NEW_CASES,
                SUM(DEATHS) as DEATHS,
                AVG(SURVIVAL_RATE) as SURVIVAL_RATE
            FROM CANCER_STATISTICS
            WHERE YEAR BETWEEN 2015 AND 2023
            GROUP BY YEAR
            ORDER BY YEAR;
            """
            stats['historical'] = pd.read_sql(historical_query, self.conn)
            
            # Regional Statistics
            regional_query = """
            SELECT 
                REGION,
                AVG(CASES_PER_100K) as CASES_PER_100K,
                AVG(DEATH_RATE) as DEATH_RATE,
                COUNT(DISTINCT TREATMENT_CENTER_ID) as TREATMENT_CENTERS
            FROM REGIONAL_STATISTICS
            GROUP BY REGION;
            """
            stats['regional'] = pd.read_sql(regional_query, self.conn)
            
            # Cancer Type Statistics
            cancer_types_query = """
            SELECT 
                CANCER_TYPE,
                AVG(INCIDENCE_RATE) as INCIDENCE_RATE,
                AVG(MORTALITY_RATE) as MORTALITY_RATE,
                AVG(FIVE_YEAR_SURVIVAL) as FIVE_YEAR_SURVIVAL
            FROM CANCER_TYPES
            GROUP BY CANCER_TYPE;
            """
            stats['cancer_types'] = pd.read_sql(cancer_types_query, self.conn)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching cancer statistics: {str(e)}")
            logger.info("Falling back to mock data")
            return self._get_mock_statistics()

    def _get_mock_statistics(self) -> Dict[str, pd.DataFrame]:
        """Generate mock statistics when database connection fails"""
        try:
            # Historical Statistics
            historical = pd.DataFrame({
                'Year': range(2015, 2024),
                'New Cases': [1643820, 1603410, 1765730, 1731100, 1628730, 
                             1532430, 1679560, 1511370, 1603980],
                'Deaths': [593565, 589353, 515674, 580178, 578191, 
                          582543, 597793, 501095, 587296],
                'Survival Rate': [71.25, 74.09, 67.09, 71.70, 69.04, 
                                69.05, 72.55, 69.40, 69.11]
            })
            
            # Regional Statistics
            regional = pd.DataFrame({
                'Region': ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'],
                'Cases per 100k': [460.78, 338.96, 372.60, 332.15, 478.73],
                'Death Rate': [139.46, 122.61, 134.81, 165.50, 113.14],
                'Treatment Centers': [108, 83, 54, 161, 93]
            })
            
            # Cancer Type Statistics
            cancer_types = pd.DataFrame({
                'Cancer Type': ['Breast', 'Lung', 'Prostate', 'Colorectal', 'Melanoma'],
                'Incidence Rate': [70.83, 119.26, 108.06, 113.97, 102.11],
                'Mortality Rate': [6.45, 31.97, 20.16, 33.39, 29.03],
                '5-Year Survival': [68.38, 85.46, 82.26, 83.37, 89.81]
            })
            
            return {
                'historical': historical,
                'regional': regional,
                'cancer_types': cancer_types
            }
        except Exception as e:
            logger.error(f"Error generating mock statistics: {str(e)}")
            return {}

    def get_visualizations(self) -> List[go.Figure]:
        """Generate visualizations from Snowflake data"""
        try:
            visualizations = []
            
            # Time series visualization
            historical_data = self.get_cancer_statistics().get('historical')
            if historical_data is not None:
                fig1 = px.line(historical_data, 
                              x='YEAR', 
                              y=['NEW_CASES', 'DEATHS'],
                              title='Cancer Cases and Deaths Over Time')
                visualizations.append(fig1)
            
            # Regional comparison
            regional_data = self.get_cancer_statistics().get('regional')
            if regional_data is not None:
                fig2 = px.bar(regional_data,
                             x='REGION',
                             y='CASES_PER_100K',
                             title='Cancer Cases per 100,000 by Region')
                visualizations.append(fig2)
            
            return visualizations

        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
            return []

    def get_treatment_costs(self) -> Optional[pd.DataFrame]:
        """Get treatment cost analysis from Snowflake"""
        try:
            query = """
            SELECT 
                TREATMENT_TYPE,
                AVG(COST) as AVERAGE_COST,
                MIN(COST) as MIN_COST,
                MAX(COST) as MAX_COST
            FROM TREATMENT_COSTS
            GROUP BY TREATMENT_TYPE;
            """
            return pd.read_sql(query, self.conn)
        except Exception as e:
            st.error(f"Error fetching treatment costs: {str(e)}")
            return None

    def close_connection(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()

    def get_all_cdc_tables(self) -> List[str]:
        """Get all tables from CDC schema"""
        try:
            if self.use_mock_data:
                logger.info("Using mock data - cannot fetch actual CDC schema tables")
                return []
            
            query = """
            SHOW TABLES IN SCHEMA CDC;
            """
            cursor = self.conn.cursor()
            cursor.execute(query)
            tables = cursor.fetchall()
            cursor.close()
            
            # Log found tables
            table_names = [table[1] for table in tables]  # table[1] contains the table name
            logger.info(f"Found {len(table_names)} tables in CDC schema: {table_names}")
            return table_names
            
        except Exception as e:
            logger.error(f"Error fetching CDC schema tables: {str(e)}")
            return []

    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get data from a specific table in CDC schema"""
        try:
            if self.use_mock_data:
                logger.info(f"Using mock data - cannot fetch actual data from {table_name}")
                return pd.DataFrame()
            
            query = f"""
            SELECT *
            FROM CDC.{table_name}
            LIMIT 1000;  -- Add limit for safety
            """
            
            df = pd.read_sql(query, self.conn)
            logger.info(f"Retrieved {len(df)} rows from {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {str(e)}")
            return pd.DataFrame()
