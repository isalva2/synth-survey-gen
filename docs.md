# LLM-ABM Research

## 9/2/2024 First Steps

- Obtain Demographic Data from [CMAP](https://cmap.illinois.gov/data/demographic-economic/) to generate an initial "Person Profile" system prompt. This will be found in `generate_population()` and return a list of strings that look like this:

```sh
"Your name is [NAME] you are [AGE] years old and you live in [CITY]. Your occupation is [JOB] and you have been employed for [YEARS] years and make $[AMOUNT] annually."
```

- we could initialize agents in a manner similar to this:

```python

def initialize_agents():
    for agent_id, initial_prompt in enumerate(prompt_list):
        new_agent = agent(id = agent_id).initialize(initial_prompt)
```

```python
class agent():
    string: initial system prompt
    agent: id

    function: write_out()
```
```
made ssh
```
