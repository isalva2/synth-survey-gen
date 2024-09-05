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
    demographic_data = pd.read_csv('data/demographic/demographic.csv')
    # I am going to first check with the demographic data, then continue land use data 
    # if my implementation for demographic data is working as intended

    #land_use_data = pd.read_csv('data/land_use/land_use.csv

    prompt_template = ("You live in {city}. The total population is {population} and the median age is {median_age}. "
                       "The employment rate is {employment_rate:.2f}%. The median income is ${income}, and most households "
                       "own around {car_ownership} vehicle(s).")
    # this is the result that returns a list of strings for demographic data
    res = []

    # I am iterating over each row demogrphic dataframe
    for index, row in demographic_data.iterrows():
        # if we want to modify this data by performing operations
        city = row['GEOG']
        population = row['TOT_POP']
        median_age = row['MED_AGE']
        income = row['MEDINC']
        employment_rate = row['EMP'] / row['POP_16OV'] * 100 if row['POP_16OV'] != 0 else 0
        car_ownership = row['ONE_VEH'] + row['TWO_VEH'] + row['THREEOM_VEH']
        # I formated the template in a way it can be expressed with variables
        prompt = prompt_template.format(city=city, population=population, median_age=median_age, 
                                        employment_rate=employment_rate, income=income, car_ownership=car_ownership)
        
        res.append(prompt)
    
    return res

    pass
system_prompts = generate_system_prompt()
# For this case I am testing with 5 prompts
for prompt in system_prompts[:5]:
    print(prompt)

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