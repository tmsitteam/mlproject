import sys

def error_message_detail(error, error_detail:sys):
    #extracting error details
    _, _, exc_tb = error_detail.exc_info()
    
    #form error details, extracting file name and line number where the error occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f'Error occured in Python script\nFile: {file_name}; Line: {line_number}; Error: {str(error)}'

    return error_message


# inherit the exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message