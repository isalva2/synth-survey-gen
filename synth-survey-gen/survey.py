from preprocess import *
from langroid.language_models import LLMMessage
from synthesize import SurveyAgent
from typing import Dict, List
from datetime import datetime

import re
import json


def _response_from_tool_message(survey_response: LLMMessage) -> str | Dict[str, str] | None:
    """ Returns either a formatted Tool Message response as a Dict, None in
    case of error, or text only survey response from survey agent.

    Args:
        survey_response (LLMMessage): langroid LLMMessage implementation

    Returns:
        _type_: str | Dict[str, str] | None
    """
    # eventually modify to return tuple, json and then "scraps" of anything else left over
    message_content = survey_response.content
    match = re.search(r'{\s*(.*)\s*}', message_content, re.DOTALL)
    if match:
        json_string = match.group(1)

        try:
            tool_content = json.loads(f"{{{json_string}}}")
            return tool_content  # Return the parsed dictionary
        except json.JSONDecodeError:
            return None
    else:
        return message_content


class SurveyEngine:
    def __init__(self, survey_conf: Dict, survey_questions: Dict, agents: List[SurveyAgent]):
        self.survey_conf = survey_conf
        self.questions = survey_questions
        self.agents = agents
        self.respondent_summaries = []

    def run(self):
        # survey logic and tool mapping
        survey_logic = self.survey_conf["logic"]

        for agent in self.agents:
            queued_variable = self.survey_conf["start"]               # queue up first question variable
            queued_question_package = self.questions[queued_variable] # get corresponding question package
            target_dtype = queued_question_package["dtype"]           # tool response dtype shoudl match this

            # logging
            logic_flow = []
            encoded_responses = []
            tool_dtypes = []
            raw_contents = []
            n_questions = 1


            # question queueing and execution, survey logic control
            while queued_variable != None:
                queued_question_package = self.questions[queued_variable] # get corresponding question package
                target_dtype = queued_question_package["dtype"]

                # load up question package and corresponding survey variable
                agent.queue_question(queued_variable, queued_question_package)
                agent.ask_question()

                # get latest LLM response message from message history
                last_message = agent.message_history[-1]
                parsed_response = _response_from_tool_message(last_message)

                # match case to settle different tool responses
                match parsed_response: # extensible
                    case str():
                        # get tool response type and encoded survey response
                        tool_dtype = "TEXT"
                        encoded_response = parsed_response
                    case dict():
                        try:
                            # get tool response type and encoded survey response
                            tool_dtype = list(parsed_response.keys())[-1]
                            encoded_response = parsed_response[tool_dtype]
                        except:
                            tool_dtype = "BADRESPONSE"
                            encoded_response = None
                    case None:
                        tool_dtype = "BADRESPONSE"
                        encoded_response = None
                    case _:
                        print("something bad happened")
                        pass # this should never happen

                logic_flow.append(queued_variable)
                encoded_responses.append(encoded_response)
                tool_dtypes.append(tool_dtype)
                raw_contents.append(last_message.content)

                """
                SURVEY LOGIC
                    1. check if survey question dtype matches tool response dtype
                    2. set control flag based on response keys, numeric or ELSE
                    3. check if next question requires flag or direct to next question
                """
                dtype_match = target_dtype == tool_dtype # 1.
                if dtype_match:
                    flag = str(encoded_response) # 2a.
                    if flag not in survey_logic[queued_variable]:
                        flag = "ELSE"
                # if bad match use catch
                elif not dtype_match:
                    flag = "ELSE" # 2b.

                step = survey_logic.get(queued_variable)
                if isinstance(step, dict):
                    queued_variable = step.get(flag) or (step.get("ELSE") if tool_dtype == "NUMERIC" else None)
                elif isinstance(step, str):
                    queued_variable = step
                else:
                    queued_variable = None

                n_questions+=1
                
            # log responses