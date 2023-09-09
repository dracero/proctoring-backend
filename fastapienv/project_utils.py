# LOGGING

import logging

class Logger:

    def __init__(self, name: str, log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add formatter to ch
        ch.setFormatter(formatter)
        
        # Add ch to logger
        self.logger.addHandler(ch)

    def log(self, message: str, level=logging.INFO):
        """Log a message with the specified level."""
        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)
        else:
            self.logger.info(message)

# ERRORS 

class ErrorHandler:

    class Error(Exception):
        """Base class for other exceptions"""
        pass

    class DatabaseConnectionError(Error):
        """Raised when there's an issue connecting to the database."""
        pass

    class InvalidCollectionError(Error):
        """Raised when an invalid collection name is provided."""
        pass

    class ModelError(Error):
        """Raised when there's an issue with the machine learning model."""
        pass

    @staticmethod
    def handle_exception(e: Exception):
        """A generic method to handle exceptions based on their type."""
        match e:
            case ErrorHandler.DatabaseConnectionError():
                ErrorHandler._handle_database_connection_error()
            case ErrorHandler.InvalidCollectionError():
                ErrorHandler._handle_invalid_collection_error()
            case ErrorHandler.ModelError():
                ErrorHandler._handle_model_error()
            case _:
                ErrorHandler._handle_generic_error()

    @staticmethod
    def _handle_database_connection_error():
        print("Handling a database connection error.")
        # Additional handling code here

    @staticmethod
    def _handle_invalid_collection_error():
        print("Handling an invalid collection error.")
        # Additional handling code here

    @staticmethod
    def _handle_model_error():
        print("Handling a model error.")
        # Additional handling code here

    @staticmethod
    def _handle_generic_error():
        print("Handling a generic exception.")
        # Additional handling code here

