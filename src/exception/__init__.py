import sys
import logging

def error_message_details(error: Exception, error_detail: sys)-> str:
    """
    Extract details error information including file name, line number, and the error message.
    
    :param error: the exception that occurred.
    :param error_detail: the sys module to access traceback details.
    :return: A formatted error message string.
    """
    # Extract traceback details (exception information)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the excepton occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatter error message string with file name, line number, and the actual error
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script: [{file_name}] at line number [{line_number}]: {str(error)}"

    # log the error for better tracing
    logging.error(error_message)

    return error_message

class MyException(Exception):
    """
    Custom exception class handling error messages with detailed information.
    """
    def __init__(self,error_message: str, error_detail:sys):
        """
        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """

        #  call the base class constructor with the error message
        super().__init__(error_message)

        # Format the detailed error message using the error_message_details function
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self)-> str:
        """
        Return the string representation of the error message
        """
        return self.error_message