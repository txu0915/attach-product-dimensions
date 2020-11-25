import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging


def do_logging_setup(log_level=logging.DEBUG):
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logging.getLogger().setLevel(log_level)
    setup_logging(handler)


do_logging_setup()

logging.info('Configured logging for Google Cloud')
