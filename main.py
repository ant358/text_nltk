# note does not run in jupyter notebook, run in the terminal
from fastapi import FastAPI
import uvicorn
import logging
import os
import pathlib
from datetime import datetime
from src.input_data import (get_document, get_pageids_from_graph,
                            get_keyword_relationship_from_graph, text_input)
from src.control import Job_list
from src.output_data import (Keywords, load_to_graph_db)

# setup logging
# get todays date
datestamp = datetime.now().strftime('%Y%m%d')
container_name = os.getenv('CONTAINER_NAME')
# append date to logfile name
log_name = f'log-{container_name}-{datestamp}.txt'
path = os.path.abspath('./logs/')
# add path to log_name to create a pathlib object
# required for loggin on windows and linux
log_filename = pathlib.Path(path, log_name)

# create log file if it does not exist
if os.path.exists(log_filename) is not True:
    # create the logs folder if it does not exist
    if os.path.exists(path) is not True:
        os.mkdir(path)
    # create the log file
    open(log_filename, 'w').close()

# create logger
logger = logging.getLogger()
# set minimum output level
logger.setLevel(logging.DEBUG)
# Set up the file handler
file_logger = logging.FileHandler(log_filename)

# create console handler and set level to debug
ch = logging.StreamHandler()
# set minimum output level
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('[%(levelname)s] -'
                              ' %(asctime)s - '
                              '%(name)s : %(message)s')
# add formatter
file_logger.setFormatter(formatter)
ch.setFormatter(formatter)
# add a handler to logger
logger.addHandler(file_logger)
logger.addHandler(ch)
# mark the run
logger.info(f'Lets get started! - logginng in "{log_filename}" today')

# create the FastAPI app
app = FastAPI()

# create the job list
create_keyword_nodes = Job_list()

# status
status = "paused"    # paused, running, stopped


def update_jobs():
    """Get the pageids of nodes in the graph database
    that do not have a NER result"""
    # get the pageids of nodes in the graph database
    graph_pageids = get_pageids_from_graph()
    # that do not have a NER result
    nodes_with_a_keyword = get_keyword_relationship_from_graph()
    # add the pageids to the job list
    if pageids := [
            pageid for pageid in graph_pageids
            if pageid not in nodes_with_a_keyword
    ]:
        create_keyword_nodes.bulk_add(pageids)
        logger.info(f'{len(pageids)} Jobs added to the job list')


def run():
    while len(create_keyword_nodes) > 0:
        # get the first job
        job = create_keyword_nodes.get_first_job()
        try:
            # get the document
            document = get_document(job)
            # run the model
            keyword_results = Keywords(document)
            # save the results
            load_to_graph_db(document, keyword_results.top_nouns)
            load_to_graph_db(document, keyword_results.top_verbs)
            # log the results
            logger.info(f'Job {job} complete')
        except Exception as e:
            logger.error(f'Job {job} failed to get document keywords {e}')
            continue


# OUTPUT- routes
@app.get("/")
async def root():
    logging.info("Root requested")
    return {"message": "text_nltk finding keywords in the text"}


@app.get("/get_current_jobs")
async def get_current_jobs():
    """Get the current jobs"""
    logging.info("Current jobs list requested")
    return {"Current jobs": create_keyword_nodes.jobs}


@app.get("/example_keywords_result")
async def example_ner_result():
    """Get an example of the keyword result
    show the top nouns from the text database"""
    logging.info("Example keyword result requested")
    result = get_document("18942")
    keyword_results = Keywords(result)
    return {"Example keyword result": keyword_results.top_nouns}


@app.get("/test_keywords_result")
async def test_keywords_result():
    """Get an example of the keyword result
    show the top nound from sample text"""
    logging.info("Test keyword result requested")
    result = text_input()
    keyword_results = Keywords(result)
    return {"Example keyword result": keyword_results.top_nouns}


@app.get("/get_status")
async def get_status():
    """Get the status of the controller"""
    logging.info("Status requested")
    return {"Status": status}


# INPUT routes
@app.post("/add_job/{job}")
async def add_job(job: str):
    """Add a job to the list of jobs"""
    create_keyword_nodes.add(job)
    run()
    logging.info(f"Job {job} added, running job")
    return {"message": f"Job {job} added"}


@app.post("/remove_job/{job}")
async def remove_job(job: str):
    """Remove a job from the list of jobs"""
    create_keyword_nodes.remove(job)
    logging.info(f"Job {job} removed")
    return {"message": f"Job {job} removed"}


# @app.post("/add_jobs_list/{jobs}")
# async def add_jobs_list(jobs: list[str]):
#     """Add a list of jobs to the list of jobs"""
#     create_keyword_nodes.bulk_add(jobs)
#     run()
#     logging.info(f"Jobs {jobs} added")
#     return {"message": f"Jobs {jobs} added"}


@app.post("/update_graph_keyword_nodes")
async def update_keyword_jobs():
    """Check the graph for has keyword relationships and update the jobs list"""
    update_jobs()
    run()
    logging.info("Jobs list updated")
    return {"message": "Jobs list updated keyword nodes being created"}


if __name__ == "__main__":
    # goto localhost:8080/
    # or localhost:8080/docs for the interactive docs
    uvicorn.run(app, port=8020, host="0.0.0.0")
