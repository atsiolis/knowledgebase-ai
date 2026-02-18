"""
Supabase Client Configuration

This module initializes and exports the Supabase client for database operations.

Environment Variables Required:
- SUPABASE_URL: Your project URL (e.g., https://xxx.supabase.co)
- SUPABASE_KEY: Your project's anon/public API key
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
# This client is imported and used throughout the application
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
