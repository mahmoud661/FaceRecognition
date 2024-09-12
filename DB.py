import psycopg2
from psycopg2 import sql

# Function to create the table if it doesn't exist


def create_table_if_not_exists():
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host="localhost",  # Your database host
            database="your_db_name",  # Your database name
            user="your_db_user",  # Your database user
            password="your_db_password"  # Your database password
        )

        # Create a cursor object to execute SQL queries
        cur = conn.cursor()

        # SQL query to create the table if it doesn't exist
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS Users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            face_embedding BYTEA NOT NULL
        );
        '''

        # Execute the query to create the table
        cur.execute(create_table_query)
        conn.commit()

        print("Table 'Users' has been created (if it did not exist).")

        # Close communication with the PostgreSQL database
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error: {e}")


# Call the function to create the table
create_table_if_not_exists()
