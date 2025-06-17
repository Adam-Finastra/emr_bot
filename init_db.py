import mysql.connector
from config import Config

def init_database():
    try:
        # Connect to MySQL server
        conn = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DB}")
        cursor.execute(f"USE {Config.MYSQL_DB}")
        
        # Drop existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS vital_signs")
        cursor.execute("DROP TABLE IF EXISTS patients")
        
        # Create vital_signs table with proper JSON columns
        cursor.execute('''
            CREATE TABLE vital_signs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                registration_id VARCHAR(50) NOT NULL,
                name VARCHAR(100) NOT NULL,
                gender ENUM('MALE', 'FEMALE') NOT NULL,
                age INT NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                height FLOAT NOT NULL,
                weight FLOAT NOT NULL,
                bmi FLOAT NOT NULL,
                temp FLOAT NOT NULL,
                systolic_bp INT NOT NULL,
                diastolic_bp INT NOT NULL,
                pulse INT NOT NULL,
                pain_scale INT NOT NULL,
                summary TEXT NOT NULL,
                alerts JSON NOT NULL,
                recommendations JSON NOT NULL,
                risk_score FLOAT,
                risk_level ENUM('LOW', 'MODERATE', 'HIGH', 'CRITICAL'),
                comorbidities JSON,
                medications JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create patients table with proper JSON columns
        cursor.execute('''
            CREATE TABLE patients (
                registration_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                gender ENUM('MALE', 'FEMALE') NOT NULL,
                age INT NOT NULL,
                comorbidities JSON,
                medications JSON,
                last_risk_score FLOAT,
                last_risk_level ENUM('LOW', 'MODERATE', 'HIGH', 'CRITICAL'),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print(f"Database '{Config.MYSQL_DB}' created or already exists")
        print("Tables created successfully")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    init_database() 