import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_ping.log"),
        logging.StreamHandler()
    ]
)

# Configuration
BACKEND_URL = "https://lbp-webapp-backend.onrender.com"
PING_ENDPOINT = "/"  # You can use a lightweight endpoint like health check or root
PING_INTERVAL_MINUTES = 14  # Render free tier goes to sleep after 15 minutes of inactivity

def ping_server():
    """Send a request to the server to keep it alive"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        response = requests.get(f"{BACKEND_URL}{PING_ENDPOINT}", timeout=10)
        if response.status_code == 200:
            logging.info(f"Server pinged successfully. Status: {response.status_code}")
            return True
        else:
            logging.warning(f"Server responded with status code: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Failed to ping server: {str(e)}")
        return False

def main():
    """Main function to periodically ping the server"""
    logging.info("Server keep-alive service started")
    
    while True:
        success = ping_server()
        
        # Calculate sleep time in seconds
        sleep_time_seconds = PING_INTERVAL_MINUTES * 60
        
        if success:
            logging.info(f"Sleeping for {PING_INTERVAL_MINUTES} minutes before next ping")
        else:
            # If ping failed, try again sooner
            sleep_time_seconds = 60  # Try again in 1 minute
            logging.info("Ping failed, will retry in 1 minute")
            
        time.sleep(sleep_time_seconds)

if __name__ == "__main__":
    main()