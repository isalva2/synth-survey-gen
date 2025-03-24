import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point
from math import ceil
import json
from typing import List, Dict, Tuple, Optional
import copy
from tqdm import tqdm
from pathlib import Path
import ollama
from ollama import ChatResponse
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
        model_config: dict = json.load(file)

    with open(config_path / "questions.json", "r") as file:
        questions: dict = json.load(file)

    return model_config, questions


def synthesize_population(config_folder:str, n_sample:int, source:str="pums", min_age: int | None = None, random_state=0) -> pd.DataFrame | None:
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


def write_bio(population_sample: pd.DataFrame):
    pass


class singleAnswerTool(lr.agent.ToolMessage):
    request: str = "singleAnswerResponse"
    purpose: str = """
        To respond with the <answer_key> of the answer that you specify. 
        """
    answer_key: int
    
    @classmethod
    def example(cls):
        return [
            cls(answer_key=45),
            (
                "To respond to the survey question with only one answer",
                cls(answer_key=5)
            )
        ]


class multipleAnswerTool(lr.agent.ToolMessage):
    request: str = "multipleAnswerResponse"
    purpose: str = """
    To respond with a list of <answer_keys> of the answers that apply to your response.
    """
    answer_keys: Tuple[int]
    
    @ classmethod
    def example(cls):
        return [
            cls(answer_keys=(4, 7, 12)),
            (
                "I want to response with the keys of the 4 answers that apply to me",
                cls(answer_keys=(4, 8, 23, 35))
            ),
            (
                "Only one answer applys to me.",
                cls(answer_keys=(5,))
            )
        ]
        
        
class discreteNumericTool(lr.agent.ToolMessage):
    request: str = "discreteNumericResponse"
    purpose: str = """
        To respond with an appropriate numeric <discrete_response> value.
        """
    discrete_response: int
    
    @classmethod
    def example(cls):
        return [
            cls(discrete_response=43),
            (
                "I want to respond with my age of 28",
                cls(discrete_response=28)
            ),
            (
                "I want to repond with my yearly salary",
                cls(discrete_respnose=43_000)
            )
        ]


class SurveyAgent(lr.ChatAgent):
    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        self.responses = []
        self.question_variables = []
        

    def singleAnswerResponse():
        pass

    def multipleAnswerResponse():
        pass

    def discreteNumericResponse():
        pass

    def llm_response(self, message: Optional[str | ChatDocument] = None) -> Optional[ChatDocument]:
        return super().llm_response(message)


def main(config_folder:str, n):

    person = process_pums_data(config_folder, True)
    household = process_pums_data(config_folder, False)
    population_sample = synthesize_population(config_folder, n)

    attribute_descriptions = get_attribute_descriptions(person)
    for i, individual in population_sample.iterrows():
        individual_attributes = attribute_decoder_dict(individual.to_dict(), person)

        write_individual_bio(individual_attributes, attribute_descriptions, config_folder)


if __name__ == "__main__":
    main("configs/Chicago", 50)
