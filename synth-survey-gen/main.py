from preprocess import process_MyDailyTravelData
from synthesize import load_config, build_agents
from survey import SurveyEngine

config_folder = "configs/Chicago"
n = 100
subsample = 10

def main():

    model_conf, synth_conf, survey_conf = load_config(config_folder)
    questions = process_MyDailyTravelData(config_folder)
    agents = build_agents(config_folder, n, subsample)

    SE = SurveyEngine(survey_conf, questions, agents)
    SE.run()

if __name__ == "__main__":
    main()