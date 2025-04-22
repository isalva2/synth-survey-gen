from preprocess import process_MyDailyTravelData
from synthesize import load_config, build_agents
from survey import SurveyEngine
from langroid.utils.configuration import settings

settings.quiet = True
config_folder = "configs/Chicago"
n = 100
subsample = 3

def main():
    _, _, survey_conf = load_config(config_folder)
    questions = process_MyDailyTravelData(config_folder)
    agents = build_agents(config_folder, n, subsample)

    SE = SurveyEngine(survey_conf, questions, agents)
    SE.run()
    return SE.results()

if __name__ == "__main__":
    main()