import logging
import os
from datetime import datetime

#log file name
LOG_FILE = f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log'
#logging path where log files will be saved
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
#continuous appending the log files into the given logging path/directory even if the file already exists
os.makedirs(logs_path, exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#logging setup
logging.basicConfig(filename=LOG_FILE_PATH,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)