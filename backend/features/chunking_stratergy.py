import re
import spacy
import spacy.cli
import tiktoken
import snowflake.connector
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import json

TOKEN_LIMIT = 8192  # OpenAI's embedding model limit
SUB_CHUNK_SIZE = 2000  # Safe sub-chunk size to avoid exceeding limits

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
# Load tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text):
    """Returns the token count of a given text."""
    return len(tokenizer.encode(text))

def split_chunk(chunk, max_tokens=SUB_CHUNK_SIZE):
    """Splits a chunk into smaller sub-chunks if it exceeds max_tokens."""
    tokens = tokenizer.encode(chunk)
    if len(tokens) <= max_tokens:
        return [chunk]  # Already within the limit

    sub_chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i + max_tokens]
        sub_chunks.append(tokenizer.decode(sub_tokens))
    
    return sub_chunks

def markdown_chunking(markdown_text, heading_level=2):
    pattern = rf'(?=^{"#" * heading_level} )'  # Regex to match specified heading level
    # chunks = re.split(pattern, markdown_text, flags=re.MULTILINE)
    raw_chunks = re.split(pattern, markdown_text, flags=re.MULTILINE)
    
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Ensure the chunk is within token limits
        chunks.extend(split_chunk(chunk, max_tokens=TOKEN_LIMIT // 2))  # Split large chunks
    
    return chunks
    
    # return [chunk.strip() for chunk in chunks if chunk.strip()]

def semantic_chunking(text, max_sentences=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
   
    chunks = []
    current_chunk = []
    for i, sent in enumerate(sentences):
        current_chunk.append(sent)
        if (i + 1) % max_sentences == 0:
            merged_chunk = " ".join(current_chunk)
            chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))  # Ensure token limits
            current_chunk = []
   
    if current_chunk:
        merged_chunk = " ".join(current_chunk)
        chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))
   
    return chunks

def sliding_window_chunking(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks using a sliding window approach.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        start = start + chunk_size - overlap

    print(f"Generated {len(chunks)} chunks using sliding window strategy")
    return chunks

class SnowflakeAgent:
    def __init__(self):
        """Initialize Snowflake connection using Streamlit secrets"""
        self.conn = self._get_snowflake_connection()

    def _get_snowflake_connection(self):
        """Establish Snowflake connection using credentials from Streamlit secrets"""
        try:
            return snowflake.connector.connect(
                user=st.secrets["snowflake"]["user"],
                password=st.secrets["snowflake"]["password"],
                account=st.secrets["snowflake"]["account"],
                warehouse=st.secrets["snowflake"]["warehouse"],
                database=st.secrets["snowflake"]["database"],
                schema=st.secrets["snowflake"]["schema"]
            )
        except Exception as e:
            st.error(f"Error connecting to Snowflake: {str(e)}")
            return None

    def get_cancer_statistics(self) -> Dict[str, pd.DataFrame]:
        """Retrieve cancer statistics from Snowflake"""
        try:
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
            st.error(f"Error fetching cancer statistics: {str(e)}")
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

class WebAgent:
    def __init__(self):
        """Initialize WebAgent with necessary APIs and endpoints"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_urls = {
            'nih': 'https://www.cancer.gov/api/v1/',
            'who': 'https://www.who.int/data/gho/api/',
            'cdc': 'https://data.cdc.gov/api/views/'
        }

    def get_cancer_research_data(self, query: str) -> Dict[str, Any]:
        """Fetch cancer research data from multiple sources"""
        try:
            data = {
                'latest_developments': self._get_latest_developments(),
                'hospital_data': self._get_hospital_data(),
                'treatment_options': self._get_treatment_options(),
                'clinical_trials': self._get_clinical_trials()
            }
            return data
        except Exception as e:
            st.error(f"Error fetching research data: {str(e)}")
            return {}

    def _get_latest_developments(self) -> Dict[str, Any]:
        """Fetch latest cancer research developments"""
        try:
            url = f"{self.base_urls['nih']}recent-research"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            st.error(f"Error fetching latest developments: {str(e)}")
            return {}

    def _get_hospital_data(self) -> pd.DataFrame:
        """Fetch hospital data and rankings"""
        try:
            # Sample data (replace with actual API call)
            data = {
                'Hospital': ['Dana-Farber', 'MD Anderson', 'Mayo Clinic'],
                'Ranking': [1, 2, 3],
                'Success_Rate': [0.85, 0.83, 0.82],
                'Annual_Patients': [45000, 42000, 40000]
            }
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching hospital data: {str(e)}")
            return pd.DataFrame()

    def _get_treatment_options(self) -> Dict[str, Any]:
        """Fetch available treatment options and their effectiveness"""
        try:
            url = f"{self.base_urls['nih']}treatment-options"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            st.error(f"Error fetching treatment options: {str(e)}")
            return {}

    def _get_clinical_trials(self) -> List[Dict[str, Any]]:
        """Fetch ongoing clinical trials"""
        try:
            url = f"{self.base_urls['nih']}clinical-trials"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error fetching clinical trials: {str(e)}")
            return []

    def get_visualizations(self) -> List[go.Figure]:
        """Generate visualizations from web data"""
        try:
            visualizations = []
            
            # Hospital rankings visualization
            hospital_data = self._get_hospital_data()
            if not hospital_data.empty:
                fig1 = px.bar(hospital_data,
                             x='Hospital',
                             y='Success_Rate',
                             title='Hospital Success Rates')
                visualizations.append(fig1)
            
            # Add more visualizations as needed
            
            return visualizations
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
            return []

    def get_insurance_data(self) -> Optional[pd.DataFrame]:
        """Fetch insurance coverage and cost data"""
        try:
            # Sample data (replace with actual API call)
            data = {
                'Insurance_Type': ['Private', 'Medicare', 'Medicaid'],
                'Coverage_Percentage': [75, 85, 95],
                'Avg_Out_of_Pocket': [5000, 3000, 1000]
            }
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching insurance data: {str(e)}")
            return None