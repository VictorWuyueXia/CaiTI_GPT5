from typing import List, Tuple, Dict, Any

import logging

from src.utils.response_bridge import get_openai_resp
from src.utils.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("Questioner")

def classify_segments(user_segments: List[str]) -> List[Tuple[str, int]]:
    """
    Classifies each user segment using the OpenAI response bridge.
    Returns a list of (label, score) tuples for each non-empty segment.
    """
    logger.info("Classifying user segments. Total segments: %d", len(user_segments))
    result = []
    for seg in user_segments:
        if not seg:
            # Skip empty segments
            continue
        label, score = get_openai_resp(seg)
        logger.debug("Segment classified: '%s' -> (label: %s, score: %s)", seg, label, str(score))
        result.append((label, score))
    logger.info("Classification complete. Results: %s", str(result))
    return result

def evaluate_result_core(
    dla_result: List[Tuple[str, Any]],
    item_index: int,
    question_index: str,
    user_segments: List[str],
    original_question: str,
    question_lib: Dict[str, Any],
    ) -> Tuple[int, int, str, Dict[str, Any]]:
    """
    Evaluates the DLA result and updates the question library accordingly.
    Returns a tuple: (valid, terminate_flag, last_question_text, updated_question_lib)
    """
    last_question = ""
    if not dla_result:
        # No DLA result, nothing to evaluate
        logger.info("No DLA result provided. Returning default values.")
        return 0, 0, last_question, question_lib

    # Handle direct label tokens (e.g., "Yes", "No", "Stop")
    if isinstance(dla_result[0][1], str):
        logger.info("Processing direct label token: %s", dla_result[0][1])
        if dla_result[0][1] == "Stop":
            logger.info("Received 'Stop' label. Terminating evaluation.")
            return 1, 1, last_question, question_lib
        # Retrieve score from question library for the label
        score = question_lib[str(item_index)][str(question_index)][dla_result[0][1]]
        question_lib[str(item_index)][str(question_index)]["score"].append(score)
        logger.info("Appended score %s for label %s to question_lib[%s][%s].", str(score), dla_result[0][1], str(item_index), str(question_index))
        if score > 1:
            # If score is high, generate a positive or negative follow-up question
            text = question_lib[str(item_index)][str(question_index)]["question"][0]
            if dla_result[0][1] == "Yes":
                text = generate_change_positive(text)
            else:
                text = generate_change_negative(text)
            text_temp = generate_synonymous_sentences(" Can you tell me more about it?")
            last_question = "It seems that " + text + " " + text_temp
            logger.info("Generated follow-up question: %s", last_question)
            # Prepare note for follow-up, to be appended by caller after collecting follow-up
            original_resp = "original_resp: " + user_segments[0]
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
        else:
            # Score is not high, just record the original response
            original_resp = "original_resp: " + user_segments[0]
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
        return 1, 0, last_question, question_lib

    # Handle numeric scores (multi-label scenario)
    # Mark valid if any scored label matches the question label
    question_label = question_lib[str(item_index)][str(question_index)]["label"]
    valid = 0
    therapist_resp = ""
    logger.info("Processing numeric/multi-label DLA result. Question label: %s", question_label)
    for i, (label, score_val) in enumerate(dla_result):
        if isinstance(score_val, str) and score_val == "Stop":
            logger.info("Received 'Stop' label in multi-label. Terminating evaluation.")
            return 1, 1, last_question, question_lib
        if isinstance(score_val, int) and score_val != 99:
            # Check if label matches the expected question label
            if label.lower() == question_label.lower():
                valid = 1
                logger.info("Label match found: %s (valid set to 1)", label)
            # Append score to the question library
            question_lib[str(item_index)][str(question_index)]["score"].append(score_val)
            logger.debug("Appended score %s for label %s to question_lib[%s][%s].", str(score_val), label, str(item_index), str(question_index))
            if score_val > 1:
                # If score is high, generate a follow-up question
                text = generate_change(user_segments[i]).lower()
                last_question = "You mentioned that " + text + " Can you tell me more?"
                logger.info("Generated follow-up question: %s", last_question)
                # Follow-up is to be acquired by caller and saved as notes
    logger.info(f"Evaluation complete. valid: {valid}, terminate_flag: 0, last_question: {therapist_resp or last_question}")
    return valid, 0, therapist_resp or last_question, question_lib
