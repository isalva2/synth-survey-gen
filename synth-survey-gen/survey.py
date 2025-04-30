from preprocess import *
from langroid.language_models import LLMMessage
from synthesize import SurveyAgent
from typing import Tuple, Union, Dict, List
from datetime import datetime
from dataclasses import dataclass
import re
import json

@dataclass
class AgentReponsePackage:
    logic_flow: List[str]
    parsed_responses: List[str | int | List[int]]
    responses_scraps: List[str]
    encoded_responses: int | str
    tool_dtypes: List[str]
    dtype_matches: List[bool]
    n_questions: int
    bad_iteration: bool



def _response_from_tool_message(survey_response) -> Tuple[Union[Dict[str, str], None], Union[str, None]]:
    """
    Extracts JSON content from the message and returns it along with any extra text.

    Args:
        survey_response (LLMMessage): langroid LLMMessage implementation

    Returns:
        Tuple[Dict[str, str] | None, str | None]: JSON dict and any leftover text.
    """
    message_content = survey_response.content
    match = re.search(r'{.*?}', message_content, re.DOTALL)

    if match:
        json_str = match.group(0)

        try:
            tool_content = json.loads(json_str)
        except json.JSONDecodeError:
            return None, message_content.strip()

        # get scraps
        start, end = match.span()
        scraps = (message_content[:start] + message_content[end:]).strip()
        return tool_content, scraps if scraps else None
    else:
        return message_content.strip() if message_content.strip() else None, None


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
            logic_flow =        []
            parsed_responses =  []
            scraps =            []
            encoded_responses = []
            tool_dtypes =       []
            dtype_matches =     []
            n_questions =       1
            bad_iteration =     False

            # question queueing and execution, survey logic control
            while queued_variable != None:
                queued_question_package = self.questions[queued_variable] # get corresponding question package
                target_dtype = queued_question_package["dtype"]

                try:
                    # load up question package and corresponding survey variable
                    agent.queue_question(queued_variable, queued_question_package)
                    agent.ask_question()

                    # get latest LLM response message from message history
                    last_message = agent.message_history[-1]
                    parsed_response, scrap = _response_from_tool_message(last_message)
                except:
                    parsed_response = None
                    scrap = None

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
                            bad_iteration = True
                    case None:
                        tool_dtype = "BADRESPONSE"
                        encoded_response = None
                        bad_iteration = True
                    case _:
                        print("something bad happened")
                        pass # this should never happen

                # log responses
                dtype_match = target_dtype == tool_dtype

                logic_flow.append(queued_variable)
                parsed_responses.append(parsed_response)
                scraps.append(scrap)
                encoded_responses.append(encoded_response)
                tool_dtypes.append(tool_dtype)
                dtype_matches.append(dtype_match)
                n_questions+=1

                if survey_logic[queued_variable] == None:
                    break
                elif dtype_match:
                    flag = str(encoded_response) # 2a.
                    if flag not in survey_logic[queued_variable]:
                        flag = "ELSE"
                # if bad match use catch
                elif not dtype_match:
                    flag = "ELSE" # 2b.
                    bad_iteration = True

                # survey logic
                step = survey_logic.get(queued_variable)
                if isinstance(step, dict):
                    queued_variable = step.get(flag) or (step.get("ELSE") if tool_dtype == "NUMERIC" else None)
                elif isinstance(step, str):
                    queued_variable = step
                else:
                    queued_variable = None

            # add to package
            response_package = AgentReponsePackage(
                logic_flow=logic_flow,
                parsed_responses=parsed_responses,
                responses_scraps=scraps,
                encoded_responses=encoded_responses,
                tool_dtypes=tool_dtypes,
                dtype_matches=dtype_matches,
                n_questions=n_questions,
                bad_iteration=bad_iteration)

            self.respondent_summaries.append(response_package)

    def results(self):
        return self.respondent_summaries