"""
Database initialization for Financial Diffusion Model
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from database.schema import Base, engine, DatabaseManager

def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    try:
        # Get database connection details from environment
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            print("DATABASE_URL not found in environment variables")
            return False
        
        # Parse connection details
        import urllib.parse as urlparse
        url = urlparse.urlparse(db_url)
        
        # Connect to PostgreSQL server (not specific database)
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            user=url.username,
            password=url.password,
            database='postgres'  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        db_name = url.path[1:]  # Remove leading slash
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database {db_name} created successfully")
        else:
            print(f"Database {db_name} already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def initialize_tables():
    """Initialize all database tables"""
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        print("Database tables created successfully")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False

def setup_database():
    """Complete database setup"""
    print("Setting up financial diffusion model database...")
    
    # Create database if needed
    if not create_database_if_not_exists():
        return False
    
    # Initialize tables
    if not initialize_tables():
        return False
    
    # Test database connection
    try:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("Database setup completed successfully")
        return True
    except Exception as e:
        print(f"Error testing database connection: {e}")
        return False

if __name__ == "__main__":
    setup_database()