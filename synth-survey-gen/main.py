import os
import threading
import time
from queue import Queue
from tqdm import tqdm
from typing import Dict, List
from preprocess import process_MyDailyTravelData
from synthesize import load_config, build_agents, SurveyAgent
from survey import SurveyEngine
from langroid.utils.configuration import settings

settings.quiet = True
config_folder = "configs/Chicago"
n = 100
subsample = 5
batch_size = 1
RUN_FOLDER = None


def run_survey(result_queue: Queue,
    stop_event: threading.Event,
    survey_conf: Dict,
    questions: Dict,
    agents: List[SurveyAgent],
    batch_size: int):
    for i in tqdm(range(0, len(agents), batch_size)):
        batch = agents[i: i+batch_size]
        SE = SurveyEngine(survey_conf, questions, batch)
        SE.run()
        for r in SE.results():
            result_queue.put(r)
    stop_event.set()


def postprocess_response(result_queue: Queue, stop_event: threading.Event):
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
            print("it threaded")
            print(result.agent.agent_id)
        except:
            pass


def main():
    global RUN_FOLDER

    start_time = time.time()
    date_str = time.strftime("%Y%m%d")
    RUN_FOLDER = os.path.join("run", date_str)
    os.makedirs(RUN_FOLDER, exist_ok=True)

    model_conf, survey_conf, survey_conf = load_config(config_folder)
    questions = process_MyDailyTravelData(config_folder)
    agents, population_sample = build_agents(config_folder, n, subsample)

    result_queue = Queue()
    stop_event = threading.Event()

    survey_thread = threading.Thread(
        target=run_survey,
        args=(
            result_queue,
            stop_event,
            survey_conf,
            questions,
            agents,
            batch_size)
    )

    postprocessing_thread = threading.Thread(
        target=postprocess_response,
        args=(result_queue, stop_event)
    )

    survey_thread.start()
    postprocessing_thread.start()
    survey_thread.join()
    postprocessing_thread.join()

    end_time = time.time()
    durtation_min = (end_time - start_time) / 60.0

    log_path = os.path.join(RUN_FOLDER, "log.txt")
    with open(log_path, "w") as f:
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Duration (seconds): {durtation_min:.2f}\n")
        f.write(f"Parameters:\n")
        f.write(f"  n = {n}\n")
        f.write(f"  model = {model_conf["chat_model"]}\n")
        f.write(f"  subsample = {subsample}\n")
        f.write(f"  batch_size = {batch_size}\n")

if __name__ == "__main__":
    main()