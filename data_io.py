import pandas as pd
import numpy as np
import functools

def get_df():
    return pd.read_csv('data/demographic/demographic.csv')

def set_random_seed(seed=None):
    # Setting the random seed for reproducibility, Decorator function
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if seed is not None:
                np.random.seed(seed)
            result = func(*args, **kwargs)
            np.random.seed(None)  
            return result
        return wrapper
    return decorator

def random_select_from_group(row, group_columns):    
    # Convert the values to float
    values = row[group_columns].values.astype(float)
    total = values.sum()

    if total == 0:
        # Assign equal probability if total is zero
        probs = np.ones(len(values)) / len(values)
    else:
        # Assign probability based on the value
        probs = values / total

    # Select a column based on the probability
    selected_idx = np.random.choice(len(group_columns), p=probs)

    # Return the column name
    return group_columns[selected_idx]

# Decorator active
@set_random_seed(seed=None)  
def generate_system_prompt(n_prompts: int, seed: int = None) -> list[str]:
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

    demographic_col_names = list(demographic_data.columns)

    # for index, current_column in enumerate(demographic_col_names):
    #     print(f"{index}: {current_column}")

    Category_Groups = {
        'Age Brackets': ['UND19', 'A20_34', 'A35_49', 'A50_64', 'A65_74', 'A75_84', 'OV85'],
        'Race/Ethnicity': ['WHITE', 'BLACK', 'ASIAN', 'HISP'],
        'Vehicle Ownership': ['ONE_VEH', 'TWO_VEH', 'NO_VEH', 'THREEOM_VEH'],
        'Employment': ['EMP', 'UNEMP'],
        'Mode of Transportation': ['WORK_AT_HOME', 'TOT_COMM', 'DROVE_AL', 'TRANSIT', 'WALK_BIKE'],
        'Income': ['INC_LT_25K', 'INC_25_50K','INC_50_75K', 'INC_75_100K', 'INC_100_150K', 'INC_GT_150'],
        # I can add more categories if needed, but so far I found the ones that are most important
    }
    category_dict = {
        # created a dictionary to convert the column names to phrases
        #Ages
        'UND19': 'between 0 and 19','A20_34': 'between 20 and 34','A35_49': 'between 35 and 49',
        'A50_64': 'between 50 and 64','A65_74': 'between 65 and 74','A75_84': 'between 75 and 84','OV85': '85 or older',
        #Race
        'WHITE': 'White','BLACK': 'Black','ASIAN': 'Asian','HISP': 'Hispanic',
        #Mode of Transportation
        'WORK_AT_HOME': 'work from home','TOT_COMM': 'commute','DROVE_AL': 'drive alone',
        'TRANSIT': 'take public transit','WALK_BIKE': 'walk or bike',
        #Number of Vehicles
        'NO_VEH': 'do not own a vehicle','ONE_VEH': 'own one vehicle','TWO_VEH': 'own two vehicles','THREEOM_VEH': 'own three or more vehicles',
        #Employment
        'EMP': 'are employed','UNEMP': 'are unemployed',
        #Income
        'INC_LT_25K': 'less than $25,000','INC_25_50K': 'between $25,000 and $50,000','INC_50_75K': 'between $50,000 and $75,000',
        'INC_75_100K': 'between $75,000 and $100,000','INC_100_150K': 'between $100,000 and $150,000','INC_GT_150': 'more than $150,000',
    }
    prompts = []
    for _ in range(n_prompts):
        # Randomly select a row from the data, I used a new function to do this
        row = demographic_data.sample(n=1).iloc[0]

        prompt_info = {}

        for group_name, group_columns in Category_Groups.items():
            selected_column = random_select_from_group(row, group_columns   )
            prompt_info[group_name] = selected_column

        # Handle Income and Median Age directly
        income_value = row['MEDINC']
        median_age_value = row['MED_AGE']
        city = row['GEOG']
        population = row['TOT_POP']

        # Convert column names to phrases
        age_phrase = f"age {category_dict.get(prompt_info['Age Brackets'])}"
        race_phrase = category_dict.get(prompt_info['Race/Ethnicity'])
        vehicle_phrase = f"You {category_dict.get(prompt_info['Vehicle Ownership'])}."
        employment_phrase = f"You {category_dict.get(prompt_info['Employment'])}"
        transportation_phrase = f"you {category_dict.get(prompt_info['Mode of Transportation'])}."
        current_income = category_dict.get(prompt_info['Income'])

        prompt_template = (
            # created a template for the prompt
            "You live in {city}. You are {age_phrase} and identify as {race_phrase}. "
            " {employment_phrase} and {transportation_phrase}"
            " {vehicle_phrase} and your current income is {current_income}."
            " The total population is {population:,}."
            " The median age is {median_age_value} and the median income is ${income_value:,}. \n\n"
        )

        prompt = prompt_template.format(
            # unpacked the prompt_info dictionary
            population=population,
            city=city,
            age_phrase=age_phrase,
            race_phrase=race_phrase,
            employment_phrase=employment_phrase,
            transportation_phrase=transportation_phrase,
            vehicle_phrase=vehicle_phrase,
            current_income=current_income,
            income_value=income_value,
            median_age_value=median_age_value
        )

        prompts.append(prompt)

    return prompts

# def _choice_from_columns(int_list: list[int], offset: int, seed: int = None):
#     list_sum = sum(int_list)
#     probs_list = [val / list_sum for val in int_list]
#     choice_idx = np.random.choice(len(int_list), 1, p = probs_list) + offset
#     return choice_idx[0]
    

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
    system_prompts = generate_system_prompt(n_prompts=5, seed=42)
    # For this case I am testing with 5 prompts
    for prompt in system_prompts[:5]:
        print(prompt)
    # there is an output of total 222 Categories

if __name__ == "__main__":
    main()