import structlog
import logging
import sys

def configure_logger(json_mode: bool = False) -> None:
    """
    Function to configure the event logger. 

    Arguments:
        - json_mode: bool, Determines if the logs are displayed in the consoule or in a seperate json log file

    """

    if json_mode:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
        )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO
    )

def get_logger(module_name: str) -> structlog.BoundLogger:
    """Helper function for modules to grab a named logger."""
    return structlog.get_logger(module_name)