import sys
import os
from pathlib import Path
import threading
import time
from queue import Queue, Empty
from tqdm import tqdm
from typing import Dict, List
from preprocess import process_MyDailyTravelData
from synthesize import load_config, build_agents, SurveyAgent
from survey import SurveyEngine, AgentReponsePackage
from postprocess import PostProcessMyDailyTravelResponse
from langroid.utils.configuration import settings

if len(sys.argv) < 2:
    print("Usage: python main.py path/to/config/ [path/to/runfolder]")
    sys.exit(1)

config_folder = sys.argv[1]
RUN_FOLDER = sys.argv[2] if len(sys.argv) > 2 else None

settings.quiet = True
n = 500
subsample = 400
batch_size = 10


def run_survey(result_queue: Queue,
    stop_event: threading.Event,
    survey_conf: Dict,
    questions: Dict,
    agents: List[SurveyAgent],
    batch_size: int):
    try:
        for i in tqdm(range(0, len(agents), batch_size), desc="running batches"):
            batch = agents[i: i+batch_size]
            SE = SurveyEngine(survey_conf, questions, batch)
            SE.run()
            for r in SE.results():
                result_queue.put(r)
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        stop_event.set()


def postprocess_response(
    result_queue: Queue,
    stop_event: threading.Event,
    postprocessor: PostProcessMyDailyTravelResponse,
    date_str: str):
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
            postprocessor.serialize_response(result)
        except Empty:
            continue
        except Exception as e:
            print(f"Error at postprocess thread: {e}")
            print(type(e))


def main():
    global RUN_FOLDER

    start_time = time.time()
    date_str = time.strftime("%Y%m%d_%H%M")
    name_str = Path(config_folder).name
    dir_name = "_".join((name_str, date_str))

    if RUN_FOLDER is None:
        dir_name = "_".join((name_str, date_str))
        RUN_FOLDER = os.path.join("run", dir_name)
    else:
        RUN_FOLDER = os.path.join(RUN_FOLDER, dir_name)

    os.makedirs(RUN_FOLDER, exist_ok=True)

    model_conf, _, survey_conf = load_config(config_folder)
    questions = process_MyDailyTravelData(config_folder)
    agents, population_sample = build_agents(config_folder, n, subsample)
    population_sample.to_csv(os.path.join(RUN_FOLDER, "_".join((date_str, "population_sample.csv"))), index=False)
    postprocessor = PostProcessMyDailyTravelResponse(
        config_folder,
        batch_size=batch_size,
        RUN_FOLDER=RUN_FOLDER,
        date_str=date_str)

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
            batch_size))

    postprocessing_thread = threading.Thread(
        target=postprocess_response,
        args=(result_queue, stop_event, postprocessor, date_str))

    survey_thread.start()
    postprocessing_thread.start()
    survey_thread.join()
    postprocessing_thread.join()

    end_time = time.time()
    durtation_min = (end_time - start_time) / 60.0

    write_success = postprocessor.write_results(RUN_FOLDER, date_str)

    log_path = os.path.join(RUN_FOLDER, "log.txt")
    chat_model = model_conf["chat_model"]
    with open(log_path, "w") as f:
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"Duration (minutes): {durtation_min:.2f}\n")
        f.write(f"Successful write to disk: {write_success}\n")
        f.write(f"Parameters:\n")
        f.write(f"  n = {n}\n")
        f.write(f"  model = {chat_model}\n")
        f.write(f"  subsample = {subsample}\n")
        f.write(f"  batch_size = {batch_size}\n")

    postprocessor.synthetic_dataset.to_csv(os.path.join(RUN_FOLDER, "_".join((date_str, "results.csv"))))

if __name__ == "__main__":
    main()