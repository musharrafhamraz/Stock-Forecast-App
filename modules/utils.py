import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_error(message):
    """Log an error message."""
    logging.error(message)

def log_info(message):
    """Log an info message."""
    logging.info(message)