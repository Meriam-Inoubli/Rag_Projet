import os
from dotenv import load_dotenv

# Env variables
load_dotenv()

# GCP
PROJECT_ID = "dauphine-437611"
REGION = os.environ['REGION']

# Cloud SQL
INSTANCE = os.environ['INSTANCE']
DATABASE = os.environ['DATABASE']
DB_PASSWORD = os.environ['DB_PASSWORD']
TABLE_NAME = os.environ['TABLE_NAME']
DB_USER = os.environ['DB_USER']