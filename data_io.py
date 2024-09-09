import pandas as pd
import numpy as np

def get_df():
    return pd.read_csv('data/demographic/demographic.csv')

def generate_system_prompt(n_prompts: int) -> list[str]:
    """
    This function reads in data from `data/demographic` and 'data/land_use' to generate
    an initial system prompt for each llm-agent survey recipient. This function should
    return a list of str prompts which we will then use to initialize every llm-agent.
    
    Demographics: https://datahub.cmap.illinois.gov/search?categories=%252Fcategories%252Fdemographics
    Land Use: https://datahub.cmap.illinois.gov/search?categories=%252Fcategories%252Fland%2520use
    
    We may use some spatial data in the future e.g. using RAG to interface with a transit stop location,
    but that is a later problem.
    """
    demographic_data = pd.read_csv('data/demographic/demographic.csv')
    # I am going to first check with the demographic data, then continue land use data 
    # if my implementation for demographic data is working as intended
    
    # for this synthetic population model, we'll run stats on contiguous columns, e.g. age brackets, and return col names.
    # for example ages 20 - over 85, return the col name for the selected bracket 'A20_34' and link this to a string prompt.
    demographic_col_names = list(demographic_data.columns)

    # get list of OBJECTIDs to iterate on
    prompt_ids = np.random.choice(demographic_data['OBJECTID'], size = n_prompts, replace = True)

    ages = []
    for prompt_id in prompt_ids:
        geo_row = demographic_data[demographic_data["OBJECTID"] == prompt_id]
        age_bracket_offset = 6
        age_bracket_list = list(geo_row.iloc[0, age_bracket_offset: age_bracket_offset + 6])
        age_col_idx = _choice_from_columns(age_bracket_list, age_bracket_offset)
        ages.append(demographic_col_names[age_col_idx])
        
        print(demographic_data[demographic_data["OBJECTID"] == prompt_id]['GEOG'].values, "\t\t", demographic_col_names[age_col_idx])

    prompt_template = ("You live in {city}. The total population is {population} and the median age is {median_age}. "
                       "The employment rate is {employment_rate:.2f}%. The median income is ${income}, and most households "
                       "own around {car_ownership} vehicle(s).")
    # # this is the result that returns a list of strings for demographic data
    # res = []

    # # I am iterating over each row demogrphic dataframe
    # for index, row in demographic_data.iterrows():
    #     # if we want to modify this data by performing operations
    #     city = row['GEOG']
    #     population = row['TOT_POP']
    #     median_age = row['MED_AGE']
    #     income = row['MEDINC']
    #     employment_rate = row['EMP'] / row['POP_16OV'] * 100 if row['POP_16OV'] != 0 else 0
    #     car_ownership = row['ONE_VEH'] + row['TWO_VEH'] + row['THREEOM_VEH']
    #     # I formated the template in a way it can be expressed with variables
    #     prompt = prompt_template.format(city=city, population=population, median_age=median_age, 
    #                                     employment_rate=employment_rate, income=income, car_ownership=car_ownership)
        
    #     res.append(prompt)
    
    return ages

def _choice_from_columns(int_list: list[int], offset: int, seed: int = None):
    list_sum = sum(int_list)
    probs_list = [val / list_sum for val in int_list]
    choice_idx = np.random.choice(len(int_list), 1, p = probs_list) + offset
    return choice_idx[0]
    

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


def main():
    system_prompts = generate_system_prompt()
    # For this case I am testing with 5 prompts
    for prompt in system_prompts[:5]:
        print(prompt)

if __name__ == "__main__":
    main()