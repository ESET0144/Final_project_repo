import psycopg2
import csv
from config import config

def load_csv_to_table(csv_file, table_name="customer_data"):
    """Load data from a CSV file into a PostgreSQL table."""

    params = config()  # read DB config

    try:
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        # 1️⃣ Read CSV header
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

        # 2️⃣ Create table dynamically
        columns = ", ".join([f'"{h}" TEXT' for h in headers])
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {columns}
        );
        """
        cur.execute(create_table_query)
        conn.commit()

        # 3️⃣ Insert rows
        insert_query = f"""
        INSERT INTO {table_name} ({", ".join(headers)})
        VALUES ({", ".join(['%s'] * len(headers))});
        """

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                cur.execute(insert_query, row)

        conn.commit()
        print(f"CSV successfully loaded into table '{table_name}'")

        cur.close()
        conn.close()

    except Exception as e:
        print("Error while loading CSV:", e)

if __name__ == "__main__":
    load_csv_to_table("customer_data.csv", "customer_data")
