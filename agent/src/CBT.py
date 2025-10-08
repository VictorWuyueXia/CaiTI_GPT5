import os
import logging
from openai import OpenAI
from src.utils.config_loader import OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
client = OpenAI(api_key=_api_key, base_url=OPENAI_BASE_URL)

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("CBT")


PROMPTER_CBT_STAGE0_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are reviewing the therapy session history and trying to ask the patient to choose a dimension that he/she would like to work on through this CBT process.
Only choose those dimensions that received a score of 2 in the conversation history.
Response format:
QUESTION: xxxx
'''

REASONER_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.



You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The user's answer towards the CBT question "Can you try to identify any unhelpful thoughts you have that contribute to this situation?". This is the step that the patient tries to recognize negative thoughts. These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.

You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx;


Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter. And you might want to pay attention to.
1. Filtering: focusing on the negative but ignore the positive
2. Polarized thinking/extreme thinking: seeing everything in all-or-nothing terms.
3. Control fallacies: assumes only self or other takes all the responsibility and is to be blamed. Includes personalization (assuming self is responsible) and blaming (assuming others at fault). 
4. Fallacy of fairness: assumes life should be fair
5. Overgeneralization: assumes a rule from one experience, using one experience for all future experiences. 
6. Emotional reasoning: “if I feel it, it must be true.” Using emotional “terms” for all the situations. 
7. Fallacy of change: expects others to change
8. “shoulds”: using personal rules to judge self and others if the rules broken
9. Catastrophizing: expecting the worst case scenario.
10. Heaven’s reward fallacy: expecting to be rewarded in some way.
11. Always being right: being wrong is unacceptable, needs to be right all the time. 
12. Personalization (like control fallacies): assuming self is responsible.
13. Jumping to conclusions: make assumptions based on little evidence


If any of these cognitive distortion is included in the UNHELPFUL_THOUGHTS, the user may still properly identifies the unhelpful thoughts. But outline the cognitive distortions in analysis.

Your goal is:
Justify if the user is identify the unhelpful thoughts properly in the statement(0: identified properly, 1: not properly identified).
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.





Example 1:
"STATEMENT: I have concern with your recent spending habits. I spent a lot of money. I spent a lot of money on clothing.; UNHELPFUL_THOUGHTS: I have issue on spending habits because I buy too much clothes.;"
DECISION: 0


Example 2:
"STATEMENT: I haven't done any creative work recently. I just don't know what are the creative things I can do.; UNHELPFUL THOUGHTS: I'm just not a creative person. I don't have any good ideas, and even if I did, they wouldn't be worth pursuing. "
DECISION:0


Example 3:
"STATEMENT: I have concern with my recent spending habits. I spent a lot of money. I spent a lot of money on clothing. RESPONSE: I like to go shopping "
DECISION:1

'''

REASONER_CBT_STAGE2_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's answer to the CBT question "Can you challenge your thought?". This is the step when the patient begin to challenge the UNHELPFUL_THOUGHTS in the STATEMENT after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?

You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx; CHALLENGE: xxxx;

Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter. And you might want to pay attention to.
1. Filtering: focusing on the negative but ignore the positive
2. Polarized thinking/extreme thinking: seeing everything in all-or-nothing terms.
3. Control fallacies: assumes only self or other takes all the responsibility and is to be blamed. Includes personalization (assuming self is responsible) and blaming (assuming others at fault). 
4. Fallacy of fairness: assumes life should be fair
5. Overgeneralization: assumes a rule from one experience, using one experience for all future experiences. 
6. Emotional reasoning: “if I feel it, it must be true.” Using emotional “terms” for all the situations. 
7. Fallacy of change: expects others to change
8. “shoulds”: using personal rules to judge self and others if the rules broken
9. Catastrophizing: expecting the worst case scenario.
10. Heaven’s reward fallacy: expecting to be rewarded in some way.
11. Always being right: being wrong is unacceptable, needs to be right all the time. 
12. Personalization (like control fallacies): assuming self is responsible.
13. Jumping to conclusions: make assumptions based on little evidence


Your goal is:
Justify if the patient challenges the unhelpful thoughts (UNHELPFUL_THOUGHTS) properly. (0: properly challenge the unhelpful thoughts, 1: not challenge the unhelpful thoughts)
Note that:
1. The patient might identify the unhelpful thoughts in a wrong way (with cognitive distortions). In this case, the patient might challenge the STATEMENT or some unhelpful thoughts that related to this STATEMENT that is not explicitly identified, which is acceptable.
2. It would be acceptable if the patient not fully challenge the validity and usability of the unhelpful thoughts/situation. As long as the CHALLENGE is related to the STATEMENT and UNHELPFUL THOUGHTS, it is acceptable.
Make notes about the distortions in the analysis.
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.



Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?;"
DECISION: 0


Example 2:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I don't know how to challenge my thoughts;"
DECISION: 1

Example 3:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I am not cool to engage in the social events;"
DECISION: 1

Example 4:
"STATEMENT: I don't smoke cigarettes, but I vape every day. I vape when I am working hard or debugging.; UNHELPFUL_THOUGHTS: I can't work or solve problems effectively without vaping.; CHALLENGE: There are several potential health impacts of vaping. It's bad for my lung.;"
DECISION: 0

'''

REASONER_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's response to challenge the UNHELPFUL_THOUGHTS after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?
4. The patient's answer to the CBT question "What is another way of thinking about this situation?". This is the step that the patient tires to reframe your unhelpful thoughts into more balanced, realistic, and constructive ones. This process is about changing the way the patient thinks about the situation, which can lead to changes in emotions and behaviors.


You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx; CHALLENGE: xxxxx; REFRAME: xxxxx;

Your goal is:
Justify if the patient reframes the unhelpful thoughts properly. (0: properly reframe the unhelpful thoughts, 1: fail to reframe the unhelpful thoughts).
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.




Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'; REFRAME: People may have their own concerns and may not be focused on me all the time and I've had good conversations in the past without embarrassing myself.;"
DECISION: 0

Example 2:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'; REFRAME: I'm finding it hard to reframe them. I still believe that: 'People are definitely judging me; I just know it.;"
DECISION: 1
'''

GUIDE_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to answer the cognitive behavioural therapy (CBT) questions based-on patient's statement provided.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.


Your goal is:
Try to recognize negative thoughts. These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic. You are trying to understand the patient's unhelpful thoughts, , so please answer the questions using the second person.


Response format:
UNHELPFUL_THOUGHTS: xxxx

You will be provideed with several examples with the statement and example unhelpful thoughts in the format of "STATEMENT: xxxxx, UNHELPFUL_THOUGHTS: xxxxxx". 



Example 1:
STATEMENT: I have not taken days off recently. Paper deadline is coming up! I don't even have time to sleep. 
UNHELPFUL_THOUGHTS: Your unhelpful thoughts might be taking days off will hinder your progress on meeting the paper deadline.

Example 2:
STATEMENT: I don't chat a lot with my colleagues. I can talk to them about work, but I can't talk to them about life. I can't seem to find common ground for life conversations with them. My personal life is quite dull and lacks the variety of personal and family activities that they have.
UNHELPFUL_THOUGHTS: Everyone will think you are boring so you don't chat with your colleagues might be your unhelpful thought.
'''

GUIDE_CBT_STAGE2_PROMPT  = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.

Your goal is:
Try to help the patient challenge the unhelpful thoughts (UNHELPFUL_THOUGHTS) properly. After recognizing and analyzing these UNHELPFUL_THOUGHTS, challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?

Response format:
CHALLENGE: xxxx

You will be provideed with several examples with the statement and example unhelpful thoughts in the format of "STATEMENT: xxxxx. UNHELPFUL_THOUGHTS: xxxxxx. CHALLENGE: xxxxxx". 



Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say. UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. 
CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'"


Example 2:
"STATEMENT: My issue is procrastination, especially when it comes to completing assignments for work or school. UNHELPFUL_THOUGHTS: When faced with a task I need to complete, I often have negative thoughts like: 'I'll never finish this on time; I'm so lazy.' 
CHALLENGE: asking myself: I have successfully complete a similar fairly challenging school project before. I might not be fair to label myself as lazy just because I'm struggling with this task."

'''

GUIDE_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's response to challenge the UNHELPFUL_THOUGHTS after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?
4. The patient's response to reframe the UNHELPFUL_THOUGHTS into more balanced, realistic, and constructive ones. This process is about changing the way the patient thinks about the situation, which can lead to changes in emotions and behaviors.


Your goal is:
Try to reframe the unhelpful thoughts (UNHELPFUL_THOUGHTS) for the patient. This is the step to reframe the patient's unhelpful thoughts into more balanced, realistic, and constructive ones. 


Response format:
REFRAME: xxxx




Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say. UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'."
REFRAME: People may have their own concerns and may not be focused on you all the time. You may had good conversations in the past without embarrassing your self.

Example 2:
"STATEMENT: I often avoid speaking up in meetings at work or in front of others. I’m afraid my ideas aren’t good enough. UNHELPFUL_THOUGHTS: If I speak up, people will think my ideas are silly. Others are much smarter than me, so my opinion doesn’t matter. CHALLENGE: I can challenge these thoughts by asking: ‘Have my colleagues ever reacted negatively when I spoke before?’ and ‘Do people usually respect different opinions, even if they’re not perfect?’."
REFRAME: My ideas have value, and sharing them can contribute to the discussion. Others are likely focused on the topic, not on judging me, and speaking up can help me grow more confident.
'''

def _chat_complete(system_content: str, user_content: str):
    logger.debug({"model": OPENAI_MODEL, "user": user_content[:200]})
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
        return resp.choices[0].message.content
    else:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "low"},
            instructions=system_content,
            input=user_content,
        )
        return resp.output_text

def stage0_prompter(history: str) -> str:
    payload = f"HISTORY: {history}"
    return _chat_complete(PROMPTER_CBT_STAGE0_PROMPT, payload)

def stage1_reasoner(statement: str, unhelpful_thoughts: str) -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
    return _chat_complete(REASONER_CBT_STAGE1_PROMPT, payload)

def stage2_reasoner(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
    return _chat_complete(REASONER_CBT_STAGE2_PROMPT, payload)

def stage3_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, reframe: str) -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
    return _chat_complete(REASONER_CBT_STAGE3_PROMPT, payload)

def stage1_guide(statement: str) -> str:
    payload = f"STATEMENT: {statement}"
    return _chat_complete(GUIDE_CBT_STAGE1_PROMPT, payload)

def stage2_guide(statement: str, unhelpful_thoughts: str) -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}"
    return _chat_complete(GUIDE_CBT_STAGE2_PROMPT, payload)

def stage3_guide(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}. CHALLENGE: {challenge}"
    return _chat_complete(GUIDE_CBT_STAGE3_PROMPT, payload)

__all__ = [
    "stage0_prompter",
    "stage1_reasoner",
    "stage2_reasoner",
    "stage3_reasoner",
    "stage1_guide",
    "stage2_guide",
    "stage3_guide",
]


