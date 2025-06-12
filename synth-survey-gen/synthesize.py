import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import langroid as lr
import langroid.language_models as lm
from langroid.agent.chat_agent import ChatDocument
from preprocess import *


def load_config(config_folder:str):
    """
    Prepares config folder
    """
    config_path = Path(config_folder)

    # read config, question, and mappers
    with open(config_path / "config.json", "r") as file:
        config: dict = json.load(file)

    return config.values()


def synthesize_population(config_folder:str, n_sample:int, source:str="pums", min_age: int|None = None, max_age: int|None = None, random_state=0) -> pd.DataFrame | None:
    """
    Returns a spatially proportional sample of the PUMS dataset based on CMAP My Daily Travel Survey respondent sample.
    """
    data_folder = Path(config_folder) / "data"
    na_str = "MISSING"

    if source == "pums":
        """
        Data derived from 2019 Public Use Microdata Sample (PUMS) dataset.
        PUMS: https://www.census.gov/programs-surveys/acs/microdata/access.2019.html#list-tab-735824205
        PUMA: https://catalog.data.gov/dataset/tiger-line-shapefile-2019-series-information-for-the-2010-census-public-use-microdata-area-puma
        """

        crs:str = "EPSG:4326"

        # load PUMS and PUMA data
        person_df = pd.read_csv(data_folder/"person.csv", low_memory=False)
        location_df = pd.read_csv(data_folder/"location.csv")

        pums_person_df = pd.read_csv(data_folder/"psam_p17.csv", dtype=str)
        if min_age is not None:
            pums_person_df = pums_person_df[pums_person_df.AGEP.astype(int) >= min_age]
        if max_age is not None:
            pums_person_df = pums_person_df[pums_person_df.AGEP.astype(int) <= max_age]

        puma_gdf = gpd.read_file(data_folder/"tl_2019_17_puma10.shp")
        puma_gdf = puma_gdf.to_crs(crs=crs)
        puma_gdf["PUMA"] = puma_gdf["PUMACE10"].astype(int)

        # get the size of each household
        household_sizes = person_df[["sampno", "perno"]].groupby(by="sampno")["perno"].max()
        household_locations = location_df[location_df.loctype==1] \
            .drop_duplicates(subset="sampno") \
            .set_index("sampno")[["longitude", "latitude"]] # always x, y for spatial operations apparently
        household_locations = household_locations.join(household_sizes)

        # get household locations as gdf
        geometry = household_locations.apply(
            lambda row: Point(row.longitude, row.latitude), axis=1
        )
        household_gdf = GeoDataFrame(household_locations, geometry=geometry, crs=crs)

        # get count of households in each PUMA zone
        household_join = household_gdf.sjoin(puma_gdf, predicate="within")
        household_counts = household_join.groupby(by=["PUMA"])["perno"].sum()
        puma_household_counts = puma_gdf \
            .set_index("PUMA") \
            .join(household_counts, how="inner")[["GEOID10", "PUMACE10", "NAMELSAD10", "perno", "geometry"]]
        puma_household_counts.reset_index(inplace=True)

        # sample by PUMA area
        samples = []
        puma_household_counts["weight"] = puma_household_counts.perno / puma_household_counts.perno.sum()
        for _, row in puma_household_counts.iterrows():
            puma = row.PUMACE10
            weight = row.weight
            n = max(int(weight*n_sample), 1)

            sample = pums_person_df[pums_person_df.PUMA==puma].sample(n, random_state=random_state)
            samples.append(sample)

        population_sample = pd.concat(samples)
        population_sample.fillna(na_str, inplace=True)

        return population_sample

    return None


class _singleAnswerTool(lr.agent.ToolMessage):
    request: str = "singleAnswerResponse"
    purpose: str = """
        To respond with the <TEXT> of the answer that you specify.
        """
    TEXT: int

    @classmethod
    def example(cls):
        return [
            cls(TEXT=45),
            (
                "To respond to the survey question with only one answer",
                cls(TEXT=5)
            )
        ]


class _multipleAnswerTool(lr.agent.ToolMessage):
    request: str = "multipleAnswerResponse"
    purpose: str = """
    To respond with a list of <TEXT> of the answers that apply to your response.
    """
    TEXT: Tuple[int]

    @ classmethod
    def example(cls):
        return [
            cls(TEXT=(4, 7, 12)),
            (
                "I want to response with the keys of the 4 answers that apply to me",
                cls(TEXT=(4, 8, 23, 35))
            ),
            (
                "Only one answer applys to me.",
                cls(TEXT=(5,))
            )
        ]


class _discreteNumericTool(lr.agent.ToolMessage):
    request: str = "discreteNumericResponse"
    purpose: str = """
        To respond with an appropriate numeric <NUMERIC> value when none of the possible responses make sense to apply.
        """
    NUMERIC: int

    @classmethod
    def example(cls):
        return [
            cls(NUMERIC=43),
            (
                "I want to respond with my age of 28",
                cls(NUMERIC=28)
            ),
            (
                "I want to repond with my yearly salary",
                cls(NUMERIC=43_000)
            )
        ]


class SurveyAgent(lr.ChatAgent):
    """
    Subclasses Langroid's Chat Agent Class for LLM interfacing and
    logic
    """
    def __init__(self, config: lr.ChatAgentConfig, agent_id, bio:str=None):
        super().__init__(config)

        # a lot of this logging stuff has been moved to survey logic, remove this eventually
        # record survey responses
        self.agent_id = agent_id
        self.bio = bio
        self.responses = []
        self.question_variables = []
        self.question_dtypes = []
        self.dtype_matches = []

        # tool interaction on current question
        self.queued_question: str
        self.possible_responses: Dict[int, str]
        self.queued_keys: List[int]
        self.answer_response = None
        self.answer_types = []

        # survey logging
        self.survey_complete: bool
        self.survey_failed: bool

    def queue_question(self, variable: str, question_package: Dict[str, str | Dict[int, str]]):
        """Takes a question/response

        Args:
            variable (str): Question variable
            question_package (Dict): Survey question and possible response dict containing variable encoding and response.
            strict_format (bool):
        """
        self.question_beginning = question_package["question"]
        self.possible_responses = question_package["response"]
        self.queued_keys = list(self.possible_responses.keys())

        self.question_variables.append(variable)
        self.question_dtypes.append(question_package["dtype"])

        # format question
        if self.question_dtypes[-1] == "TEXT":
            self.queued_question = f"{self.question_beginning} Available options: " + "; ".join(
                f"{key}: {value}" for key, value in self.possible_responses.items()
                )
        elif self.question_dtypes[-1] == "NUMERIC":
            self.queued_question = f"{self.question_beginning} Please provide a numeric response or select an alternative: " + "; ".join(
                f"{key}: {value}" for key, value in self.possible_responses.items()
                )

    def llm_response(self, message: Optional[str | ChatDocument] = None) -> Optional[ChatDocument]:
        return super().llm_response(message)

    def ask_question(self):
        self.llm_response(self.queued_question)

    def singleAnswerResponse(self, msg: _singleAnswerTool) -> str:
        # return answer if exists in queued keys
        self.dtype_matches.append("TEXT" == self.question_dtypes[-1])
        return str(msg.TEXT if msg.TEXT in self.queued_keys else None)

    def multipleAnswerResponse(self, msg: _multipleAnswerTool):
        self.dtype_matches.append("TEXT" == self.question_dtypes[-1])
        return str(
            _multipleAnswerTool.TEXT if all(
                key in self.queued_keys for key in _multipleAnswerTool.TEXT) else None)

    def discreteNumericResponse(self, msg: _discreteNumericTool):
        self.dtype_matches.append("NUMERIC" == self.question_dtypes[-1])
        return str(msg.NUMERIC if msg.NUMERIC not in self.queued_keys else None)


def build_agents(config_folder:str, n: int, subsample: int | None = None):
    model_config, _, _ = load_config(config_folder)
    person = process_pums_data(config_folder)
    population_sample = synthesize_population(config_folder, n, min_age=18)
    ploc = puma_locations(config_folder)

    MsgGen = SystemMessageGenerator(config_folder, "SystemMessage.j2")
    year = 2015

    system_messages = []
    attribute_descriptions = get_attribute_descriptions(person)
    for i, individual in population_sample.iterrows():
        individual_attributes = attribute_decoder_dict(individual.to_dict(), person)
        system_message = MsgGen.write_system_message(
            **individual_attributes,
            **attribute_descriptions,
            ploc=ploc,
            YEAR=year)
        system_messages.append(system_message)

    llm_config = lm.OpenAIGPTConfig(**model_config)
    agents = []
    for i, system_message in enumerate(system_messages[0:subsample]):
        agent_config = lr.ChatAgentConfig(
            name=f"Agent_{i}",
            llm=llm_config,
            system_message= f"We are role playing. Please assume the identity provided below and answer the questions to the best of your ability. " \
                + system_message + \
                " The date is July 17, 2015. Please answer the following travel survey questions.",
            use_tools=True,
            use_functions_api=False)
        agent = SurveyAgent(config=agent_config, agent_id = i, bio = system_message)
        agent.enable_message(_singleAnswerTool)
        agent.enable_message(_multipleAnswerTool)
        agent.enable_message(_discreteNumericTool)
        agents.append(agent)

    return agents, population_sample


if __name__ == "__main__":
    build_agents("configs/Chicago", 50, 3)
