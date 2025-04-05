from preprocess import process_MyDailyTravelData
from synthesize import load_config, build_agents
from survey import SurveyEngine

config_folder = "configs/Chicago"
n = 100
subsample = 5
sequential = ["AGE", "OCCUP", "OCCUP_O", "NOGOWHY_0", "NOGOWHY2"]

def main():
    
    model_conf, synth_conf, survey_conf = load_config(config_folder)
    questions = process_MyDailyTravelData(config_folder)
    agents = build_agents(n, subsample)
    
    SE = SurveyEngine(survey_conf)
    SE.init_survey(agents)
    SE.run(questions, sequential=sequential)
    
if __name__ == "__main__":
    main()