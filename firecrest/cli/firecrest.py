import click
import logging

from firecrest.app.firecrest_application import FirecrestApp


@click.command()
@click.option(
    "--logfile",
    type=click.Path(exists=False),
    help="If specified, the log filename. "
    " If unspecified, the log will be written to stdout.",
)
@click.argument("workflow_filepath", type=click.Path(exists=True))
def run(logfile, workflow_filepath):
    logging_config = {}
    logging_config["level"] = logging.INFO

    if logfile is not None:
        logging_config["filename"] = logfile

    logging.basicConfig(**logging_config)
    log = logging.getLogger(__name__)

    try:
        application = FirecrestApp(workflow_file=workflow_filepath)

        application.run()
    except Exception as e:
        log.exception(e)
