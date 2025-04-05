from preprocess import *
from synthesize import load_config, SurveyAgent
from typing import Dict, List, Union
import re
import json
import ast


def _parse_tool_response(agent: SurveyAgent):
    """
    Internal SurveyEngine utility to parse responses from SurveyAgent class.
    """

class SurveyEngine:
    def __init__(self, survey_conf: Dict):
        self.survey_conf = survey_conf
    
    def _init_parser(self):
        # from Langriod definition of tool message response
        # self.tool_pattern = \
            # r'^\s*TOOL\s*{\s*"([^"]+)"\s*:\s*"([^"]+)",\s*"([^"]+)"\s*:\s*([^}]+)\s*}\s*$'
        # self.tool_pattern = r'\s*{\s*"([^"]+)"\s*:\s*"([^"]+)",\s*"([^"]+)"\s*:\s*([^}]+)\s*}\s*'
        self.tool_pattern = r'^\s*{\s*"([^"]+)"\s*:\s*"([^"]+)",\s*"([^"]+)"\s*:\s*\[([^\]]+)\]\s*}\s*$'

        self.tool_parser_types = self.survey_conf["tool_parser_types"]
        
    def _check_format(self, message_content: str):
        return bool(re.match(self.tool_pattern, message_content))
    
    # check if tool response is fomatted correctly and extract the
    # tool used, answer response type, and answer itself.
    def extract_tool_content(self, message_content: str):
        match = re.match(self.tool_pattern, message_content)
        if match:
            first_key = match.group(1)
            first_value = match.group(2)
            second_key = match.group(3)
            second_value = match.group(4)
            return first_key, first_value, second_key, second_value
        else:
            return None

    def extract_json_from_string(self, input_string):
        # Regex to match the JSON part of the string (i.e., content inside { })
        match = re.search(r'{\s*(.*)\s*}', input_string, re.DOTALL)
        
        if match:
            json_string = match.group(1)  # Get the content inside the braces
            
            # Now, we need to parse the JSON-like string
            try:
                data = json.loads(f"{{{json_string}}}")  # Use json.loads to parse it
                return data  # Return the parsed dictionary
            except json.JSONDecodeError:
                print("Error decoding JSON.")
                return None
        else:
            print("No JSON found.")
            return None
    
    # load agents and init parser tools
    def init_survey(self, agents: List[SurveyAgent]):
        self.agents = agents
        self._init_parser()
        
        
    def run(self, survey_questions: Dict, sequential: List[str] | None):
        
        # start variable dictates flow of survey logic
        queued_var = self.survey_conf["start"]
        survey_logic = self.survey_conf["logic"]
        
        # question package contains survey question to be asked,
        # data type of question, and dictionary of possible response
        # for the survey question.
        question_package = survey_questions[queued_var]
        
        for agent in self.agents:
            for question in sequential:
                queued_var = question
                question_package = survey_questions[queued_var]
                try:
                    # queue a question
                    agent.queue_question(queued_var, question_package)

                    print(agent.queued_question)
                    # ask question
                    agent.ask_question()
                    
                    #retrieve and parse response
                    content = agent.message_history[-1].content
                    print(content)
                    # tool_content = self.extract_tool_content(content)
                    tool_content = self.extract_json_from_string

                    # check if tool content is good
                    if tool_content is not None:
                        print("its good")
                        print(tool_content)
                        # _, tool_type, answer_type, answer = tool_content
                        
                        # # check if tool type matches
                        # # appropriate tool for question dtype
                        # if tool_type in self.tool_parser_types[question_package["dtype"]]:
                        #     if (answer_type == "answer_key") or \
                        #     (answer_type == "discrete_response"):
                        #         encoded_answer = int(answer)
                        #     elif answer_type == "answer_keys":
                        #         encoded_answer = ast.literal_eval(answer)
                        # print(answer_type)
                        
                    else:
                        # add logic to cut question short
                        print("bad response")
                        None
                    
                except: # if fail anywhere in the 
                    print("somthing went wrong")
                    pass
                
                
            
        
        