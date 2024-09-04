import pandas as pd

def generate_system_prompt() -> list[str]:
    """
    This function reads in data from `data/demographic` and 'data/land_use' to generate
    an initial system prompt for each llm-agent survey recipient. This function should
    return a list of str prompts which we will then use to initialize every llm-agent.
    
    Demographics: https://datahub.cmap.illinois.gov/search?categories=%252Fcategories%252Fdemographics
    Land Use: https://datahub.cmap.illinois.gov/search?categories=%252Fcategories%252Fland%2520use
    
    We may use some spatial data in the future e.g. using RAG to interface with a transit stop location,
    but that is a later problem.
    """
    pass

def build_travel_survey():
    """
    This function should read from `data/travel_survey`. The contents of this directory
    should be the zipped folder `MyDailyTravelData.zip` folder from this gh repo:
    https://github.com/CMAP-REPOS/mydailytravel.
    
    In particular, the file `data_dictionary.xlsx` contains the actual survey questions.
    This may have to be modified by hand, but we shall see.
    
    This dataset corresponds to this CMAP report: https://cmap.illinois.gov/wp-content/uploads/My-Daily-Travel-pre-pandemic-travel-1.pdf
    of which we are trying to replicate (statistically) similar survey results using our
    synthetic llm-agent survey recipients.
    """
    
# When you are done downloading data, add `data/` to the gitignore. Later down the line we will be using
# CMAP's full dataset, which is under security clearance with my employer.