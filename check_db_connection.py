from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

# Load credentials from .env
load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

if not URI or not USER or not PASSWORD:
    print("Error: Missing credentials in .env file.")
    exit(1)

print(f"Connecting to {URI} as {USER}...")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def test_connection():
    try:
        # Simple read query to verify connectivity
        cypher = "RETURN 'Connected to Neo4j' AS msg"
        with driver.session() as session:
            result = session.run(cypher)
            record = result.single()
            print(record["msg"])
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    test_connection()
