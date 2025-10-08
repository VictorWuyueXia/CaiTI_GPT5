# src/response_analyzer.py

import os
import logging
from openai import OpenAI
from src.utils.config_loader import OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS

# Retrieve OpenAI API key from environment variable; fail fast if not set
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
client = OpenAI(api_key=_api_key, base_url=OPENAI_BASE_URL)

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("ResponseAnalyzer")

# === Prompt templates for OpenAI API ===

# Prompt for classifying user input into dimension and score
INIT_ASKER_SYSTEM_PROMPT_V2 = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You will be provided with:
1. All dimension names.
2. The example user inputs with their dimensions and scores. The example will be provided in the following format: {"in": "[USER_INPUT]", "res": "DIMENSION, SCORE"}

Your goal is:
To assign the user input with DIMENSION and SCORE.

All dimension names are:
1_weight, 1_mood, 1_medication, 1_care, 2_house, 3_talk, 3_emo, 4_safe, 4_risk, 5_sleep, 5_eat, 5_work, 5_work_dayoff,
5_showup, 6_finance, 7_nutrition, 8_problem, 9_support, 9_family, 10_drug, 10_ciga, 10_alcohol, 11_hobbies, 11_creativity, 12_community, 
13_support, 13_social, 14_sex, 14_comfortable, 14_protection, 15_productivity, 15_work_motivation, 16_coping, 17_sib, 17_arrest,17_legal, DLA_18_hygiene, DLA_21_sports
Yes, No, Maybe, Question, Stop

The definition of each dimension are:
1_weight: Maintaining stable weight
1_mood: Managing mood 
1_medication: Taking medication as prescribed
1_care: Participating primary and mental health care
2_house: Organizing personal possessions and doing housework
3_talk: Talking to other people
3_emo: Expressing feelings to other people
4_safe: Managing personal safety
4_risk: Managing risk
5_sleep: Following regular schedule for bedtime and sleeping enough
5_eat: Maintaining regular schedule for eating
5_work: Managing work/school
5_work_dayoff: Having work-life balance
5_showup: Showing up for appointments and obligations
6_finance: Managing finance and items of value 
7_nutrition: Getting adequate nutrition
8_problem: Problem solving and decision making capability
9_support: Family support
9_family: Family relationship
10_drug: Other substances abuse
10_ciga: Tobacco abuse
10_alcohol: Alcohol abuse
11_hobbies: Enjoying personal choices for leisure activities
11_creativity: Creativity
12_community: Participation in community
13_support: Support from social network
13_social: Relationship with friends and colleagues
14_sex: Active in Sex
14_comfortable: Managing boundaries in close relationship
14_protection: Managing sexual safety
15_productivity: Productivity at work or school
15_work_motivation: Motivation at work or school
16_coping: Coping skills to de-stress
17_sib: Exhibiting control over self-harming behavior
17_arrest: Law-abiding
17_legal: Managing legal issue
18_hygiene: Maintaining personal hygiene
21_sports: Doing exercises and sports


There are some dimension maybe confusing, to distinguish them:
1. 5_eat cares does the user eat regularly and 5_nutirtion cares more about whether the user eat enough good food for nutrition.
2. 1_mood cares about the feeling of the user, while 3_emo cares about whether the user is able to express their feelings to others.
3. 4_safe concerns the safety of users' lives, while 4_risk cares if the user is taking any risks. 

If the user input is general response, such as “Yes”, “No”, “I don’t know”, “Stop”, and “I don’t understand your question”. The DIMENSION will be within [Yes, No, Maybe, Question, Stop], and the SCORE will be 0.

The score ranges from 0 to 2, where:
0 indicates that the user performs well in this dimension;
1 indicates that the user has some problems in this dimension, but no immediate action is needed;
2 indicates a need for heightened attention from health-care providers;

If the user input does not belong to any of these dimension, the "DIMENSION, SCORE" will be: "Other, 0" 

The example user inputs with their dimensions and scores: 
{"in":"Yes, I do.", "res": "Yes, 0"}
{"in":"My weight doesn't change.", "res": "1_weight, 0"}
{"in":"I didn't measure my weight recently.", "res": "1_weight, 2"}
{"in":"My weight has increased a lot these days.", "res": "1_weight, 2"}
{"in":"I get some weight these days.", "res": "1_weight, 1"}
{"in":"My emotions are out of my control.", "res": "1_mood, 2"}
{"in":"I don't have a therapist.", "res": "1_care, 0"}
{"in":"I don't have a psychiatrist.", "res": "1_care, 0"}
{"in":"I haven't visited my prescriber for a while.", "res": "1_care, 1"}
{"in":"I haven't gone to my case manager for a while.", "res": "1_care, 1"}
'''

# Prompt for summarizing user response in a reflective way
REFLECTIVE_SUMMERIZER_PROMPT = ''' You are an intelligent agent to summarize what the user said.

You will be provide with:
The original question asked and the user response in the format of '{"Original Question": XXXX, "User Response": XXXX}'
If the user’s response is essentially “Yes,” use the information from the original question; otherwise, base it on the user input and restate it in third-person voice.
Response format:
REFLECTIVE_SUMMERIZER: XXXXX

Example 1:
{"Original Question": "Do you have coping skills to help you calm down?", "User Response": "Yes, I do"}
REFLECTIVE_SUMMERIZER: You mentioned that you have coping skills to help you calm down. 

Example 2:
{"Original Question": "Are you involved in any legal issues recently?", "User Response": "Yes, I do"}
REFLECTIVE_SUMMERIZER: You shared that you are involved in some legal issues recently.

Example 3:
{"Original Question": "How's your mood recently?", "User Response": "I feel so depressed daily."}
REFLECTIVE_SUMMERIZER: You shared that you feel so depressed daily.

Example 3:
{"Original Question": "Have your weight changed significantly recently?", "User Response": "My weight increased a lot recently."}
REFLECTIVE_SUMMERIZER: You mentioned that your weight increased a lot recently.
'''

# Prompt for rephrasing a question as a therapist
REPHRASER_PROMPT = ''' You are an intelligent agent who have strong reasoning capability and psychology and mental health commonsense knowledge. 

You will be provide with:
The original question in the format of {"Original Question": XXXXX}. 
And you need to act as a therapist to rephrase the question to client.


Response format:
REPHRASER: XXXXX

Example 1:
{"Original Question": "Do you have coping skills to help you calm down?"}
REPHRASER: Do you have strategies that help you calm yourself when you are upset?

Example 2:
{"Original Question": "Are you involved in any legal issues recently?"}
REPHRASER: Are you dealing with any legal issues right now?

Example 3:
{"Original Question": "How's your mood recently?"}
REPHRASER: How would you describe your mood recently?.

Example 3:
{"Original Question": "Have your weight changed significantly recently?"}
REPHRASER: Have you noticed any significant changes in your weight lately?
'''

def _chat_complete(system_content: str, user_content: str):
    """
    Send a chat completion request to OpenAI API with the given system and user content.
    Returns the content of the first message in the response.
    """
    logger.info("Sending request to OpenAI API for chat completion.")
    logger.debug(f"Model: {OPENAI_MODEL}, User content preview: {user_content[:100]}")
    if "gpt-5" not in OPENAI_MODEL:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE,
        )
        logger.info("Received response from OpenAI API.")
        return resp.choices[0].message.content
    else:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "low"},
            instructions=system_content,
            input=user_content,
        )
        logger.info("Received response from OpenAI API.")
        return resp.output_text

def classify_dimension_and_score(user_input: str) -> str:
    """
    Classify user input into a dimension and score using the OpenAI API.
    Input: user_input (str) - any user response string.
    Output: Raw model text, e.g., '1_weight, 2' or 'Yes, 0'.
    """
    logger.info("Classifying user input for dimension and score.")
    logger.debug(f"User input: {user_input}")
    return _chat_complete(INIT_ASKER_SYSTEM_PROMPT_V2, user_input)

def reflective_summarizer(original_question: str, user_response: str) -> str:
    """
    Summarize the user's response in a reflective, third-person style.
    Input: original_question (str), user_response (str)
    Output: Reflective summary string.
    """
    logger.info("Generating reflective summary for user response.")
    logger.debug(f"Original question: {original_question}, User response: {user_response}")
    payload = f'{{"Original Question": "{original_question}", "User Response": "{user_response}"}}'
    return _chat_complete(REFLECTIVE_SUMMERIZER_PROMPT, payload)

def rephrase_question(original_question: str) -> str:
    """
    Rephrase the original question as a therapist would.
    Input: original_question (str)
    Output: Rephrased question string.
    """
    logger.info("Rephrasing question for therapist style.")
    logger.debug(f"Original question: {original_question}")
    payload = f'{{"Original Question": "{original_question}"}}'
    return _chat_complete(REPHRASER_PROMPT, payload)