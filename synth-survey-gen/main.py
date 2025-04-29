import os
from pathlib import Path
import threading
import time
from queue import Queue
from tqdm import tqdm
from typing import Dict, List
from preprocess import process_MyDailyTravelData
from synthesize import load_config, build_agents, SurveyAgent
from survey import SurveyEngine
from postprocess import PostProcessMyDailyTravelResponse
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


def postprocess_response(
    result_queue: Queue,
    stop_event: threading.Event,
    postprocessor: PostProcessMyDailyTravelResponse):
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
            postprocessor.serialize_response(result)
        except:
            pass


def main():
    global RUN_FOLDER

    start_time = time.time()
    date_str = time.strftime("%Y%m%d_%H%M")
    name_str = Path(config_folder).name
    dir_name = "_".join((name_str, date_str))
    RUN_FOLDER = os.path.join("run", dir_name)
    os.makedirs(RUN_FOLDER, exist_ok=True)

    model_conf, survey_conf, survey_conf = load_config(config_folder)
    questions = process_MyDailyTravelData(config_folder)
    agents, population_sample = build_agents(config_folder, n, subsample)
    population_sample.to_csv(os.path.join(RUN_FOLDER, "_".join((date_str, "population_sample.csv"))), index=False)
    postprocessor = PostProcessMyDailyTravelResponse(config_folder)

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
        args=(result_queue, stop_event, postprocessor)
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
        chat_model = model_conf["chat_model"]
        f.write(f"  model = {chat_model}\n")
        f.write(f"  subsample = {subsample}\n")
        f.write(f"  batch_size = {batch_size}\n")

    postprocessor.synthetic_dataset.to_csv(os.path.join(RUN_FOLDER, "_".join(("results", date_str))))

if __name__ == "__main__":
    main()