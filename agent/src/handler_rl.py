import time
import logging
from typing import Tuple, Dict, Any

import numpy as np
import os
import pandas as pd
import re

from src.utils.config_loader import (
    ITEM_N_STATES,
    NUMBER_QUESTIONS,
    GAMMA,
    ALPHA,
    QUESTION_LIB_FILENAME,
    SUBJECT_ID,
    DATA_DIR,
)
from src.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from src.utils.io_record import log_question, get_answer, get_resp_log, init_record
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
    initialize_question_mask,
    initialize_question_table,
)
from src.utils.response_bridge import get_openai_resp
from src.utils.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)
from src.questioner import classify_segments, evaluate_result_core
from src.reflection_validation import rv_reasoner, rv_guide, rv_validation
from src.CBT import (
    stage0_prompter,
    stage1_reasoner, stage2_reasoner, stage3_reasoner,
    stage1_guide, stage2_guide, stage3_guide,
)

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("HandlerRL")

class HandlerRL:
    """
    Top-level RL workflow coordinator.
    Handles the main reinforcement learning loop for question selection and evaluation.
    All file I/O is performed via utility modules.
    """

    def __init__(self):
        # Stores the last question asked to the user
        self.last_question: str = " "
        # Stores all user responses for later result generation
        self.new_response: list = []
        # The main question library loaded from file
        self.question_lib: Dict[str, Any] = {}
        # Masks for available questions per item (to avoid repeats)
        self.all_question_mask: Dict[int, Any] = {}
        # Q-tables for each question set (per item)
        self.all_question_q_table: Dict[int, Any] = {}
        # Q-table for item selection (top-level RL)
        self.item_q_table = None

    def setup(self):
        """
        Initialize records, load question library, and set up Q-tables and masks.
        """
        logger.info("Initializing RL handler setup: loading records and question library.")
        init_record()
        self.question_lib = load_question_lib(QUESTION_LIB_FILENAME)
        # Define possible actions for item selection (as string indices)
        item_actions = ['{0}'.format(e) for e in np.arange(0, ITEM_N_STATES)]
        # Initialize masks and Q-tables for all questions and items
        self.all_question_mask = initialize_question_mask(NUMBER_QUESTIONS)
        self.all_question_q_table = initialize_question_table(NUMBER_QUESTIONS)
        self.item_q_table = initialize_q_table(ITEM_N_STATES, item_actions)
        self.item_actions = item_actions
  
        # Load persistent Q tables (if exist)
        qdir = os.path.join(DATA_DIR, "q_tables")
        if os.path.exists(os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv")):
            self.item_q_table = pd.read_csv(os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv"), index_col=0)
        logger.info(f"Loaded item Q table for subject {SUBJECT_ID}.")
        
        logger.info("RL handler setup complete.")

    def evaluate_result(self, DLA_result, S, question_A, user_input, original_question_asked):
        """
        Evaluate the result of a user's response to a question.
        Updates the question library and last question as needed.
        ReflectionValidation three steps（topic = the dimension label of the current question）
        """
        logger.info(f"Evaluating result for item {S}, question {question_A}.")
        # Call the core evaluation logic
        valid, terminate, last_q, updated = evaluate_result_core(
            [(lbl, sc) for lbl, sc in DLA_result], S, question_A, user_input, original_question_asked, self.question_lib
        )
        self.question_lib = updated
        # Update last_question if a new one is provided
        self.last_question = last_q or self.last_question
        if last_q:
            logger.info(f"Logging last question and collecting follow-up for item {S}, question {question_A}.")
            # Log the last question and get a follow-up response
            log_question(last_q)
            follow_up = get_resp_log()

            # ReflectionValidation three steps（topic = the dimension label of the current question）
            topic = self.question_lib[str(S)][str(question_A)]["label"]
            original_resp = user_input[0] if user_input else ""

            logger.info(f"Running ReflectionValidation reasoner for topic '{topic}'.")
            rv_decision_raw = rv_reasoner(topic, original_question_asked, original_resp, follow_up)
            # Simple parsing: extract '0/1'（no try/except, error directly thrown）
            rv_decision_token = "0" if "0" in rv_decision_raw else "1"
            logger.info(f"ReflectionValidation decision: {rv_decision_token}")

            # If not related (1), give guidance, recollect follow-up
            rv_guide_text = ""
            follow_up_0 = ""
            if rv_decision_token == "1":
                logger.info("Follow-up not related, generating guidance and recollecting follow-up.")
                follow_up_0 = follow_up
                rv_guide_text = rv_guide(topic, original_question_asked, original_resp, follow_up)
                log_question(rv_guide_text)
                follow_up = get_resp_log()

            # Empathic validation
            logger.info("Running ReflectionValidation empathic validation.")
            rv_validation_text = rv_validation(topic, original_question_asked, original_resp, follow_up)
            
            # Generate therapist response (store as context, do not prompt user)
            logger.info("Generating therapist response.")
            therapist_resp = generate_therapist_chat((last_q + " " + follow_up).strip())
            self.last_question = ""
            
            # Record notes (expand RV fields)
            logger.info("Recording notes for this question/response.")
            note_resp = [
                "original_question: " + original_question_asked,
                "original_resp: " + (user_input[0] if user_input else ""),
                "followup_resp: " + follow_up_0 if follow_up_0 else "followup_resp: " + follow_up,
                "rv_decision: " + rv_decision_token,
                ("rv_guide: " + rv_guide_text) if rv_guide_text else "rv_guide: ",
                "followup_resp_1: " + follow_up if follow_up_0 else "followup_resp_1: "
                "rv_validation: " + rv_validation_text,
                "therapist_resp: " + therapist_resp
            ]
            self.question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
            
        return valid, terminate, self.last_question, self.question_lib

    def ask_question(self, S: int) -> Tuple[float, int, str]:
        """
        Handles the RL loop for asking questions within a given item (S).
        Returns the total reward, termination flag, and the last question asked.
        """
        logger.info(f"Starting question RL loop for item S={S}.")
        question_reward = []
        DLA_terminate = 0
        
        # If there is only one question for this item, ask it directly
        question_A = "1"
        # Check if the score list for this question is empty (i.e., not answered yet)
        if len(self.question_lib[str(S)][str(question_A)]["score"]) == 0:
            # Get the number of available question variants for this item
            number_of_questions = len(self.question_lib[str(S)][str(question_A)]["question"])
            # Randomly select one question variant to ask
            choice_of_question = np.random.randint(number_of_questions)
            question_text = self.question_lib[str(S)][str(question_A)]["question"][choice_of_question]
            # With probability, generate a synonymous version of the question
            if np.random.uniform() < 0.95:
                question_text = generate_synonymous_sentences(question_text)
            # Concatenate the last question (context) with the current question
            question_text_ask = question_text
            # Log the question being asked
            log_question(question_text_ask)
            # Get user input for the question
            _ , user_input = get_answer()
            # Classify the user response into DLA result segments
            DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
            # Evaluate the result and update state
            valid, DLA_terminate, self.last_question, self.question_lib = self.evaluate_result(
                DLA_result, S, question_A, user_input, question_text
            )
            # If the answer is invalid and not terminated, retry once only when classification failed (NA/empty)
            if valid == 0 and DLA_terminate == 0:
                na_only = (not DLA_result) or all(
                    (isinstance(sc, int) and sc == 99) or (lbl == "NA")
                    for (lbl, sc) in DLA_result
                )
                if na_only:
                    log_question(question_text_ask)
                    _ , user_input = get_answer()
                    DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
                    valid, DLA_terminate, self.last_question, self.question_lib = self.evaluate_result(
                        DLA_result, S, question_A, user_input, question_text
                    )
        # Retrieve all scores for this question after answering
        all_score = self.question_lib[str(S)][str(question_A)]["score"]
        # Calculate the mean score if available, otherwise set to 0.0
        question_openai_res = np.mean(all_score) if all_score else 0.0
        # Append the result to the question reward list
        question_reward.append(question_openai_res)

        # Return the total reward, termination flag, and last question
        logger.info(f"Finished question RL loop for item S={S}. Total reward: {float(sum(question_reward))}, DLA_terminate: {int(DLA_terminate)}")
        return float(sum(question_reward)), int(DLA_terminate), self.last_question

    def run_cbt(self):
        """
        Run CBT stages 0-3 after screening is finished or user said stop.
        Stage 0: ask user to choose a dimension with score=2 to work on.
        Stages 1-3: unhelpful thoughts -> challenge -> reframe, with reasoning and guidance.
        """
        logger.info("Starting CBT flow (stages 0-3).")
        # 0) Collect dimensions with score=2
        candidates = []  # list of (idx, i, j, label)
        idx = 1
        for i in range(1, len(self.question_lib) + 1):
            for j in range(1, len(self.question_lib[str(i)]) + 1):
                entry = self.question_lib[str(i)][str(j)]
                if any((s == 2) for s in entry.get("score", [])):
                    candidates.append((idx, i, j, entry["label"]))
                    idx += 1

        if not candidates:
            logger.info("No dimensions with score=2. Skipping CBT.")
            log_question("We do not have a dimension at score 2 to work on today. We will conclude here.")
            return

        # Build a brief 'history' listing choices for stage0_prompter
        history_lines = ["DIMENSIONS WITH SCORE=2:"]
        for k, _, _, lbl in candidates:
            history_lines.append(f"{k}) {lbl}")
        history = "\n".join(history_lines)

        # Stage 0: ask the user to choose a dimension
        q0 = stage0_prompter(history)
        q0_clean = q0.strip()
        if q0_clean.lower().startswith("question:"):
            q0_clean = q0_clean.split(":", 1)[1].strip()
        log_question(q0_clean)
        resp = get_resp_log()
        if isinstance(resp, str) and resp.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT stage 0.")
            return

        def _pick_candidate(answer: str):
            ans = str(answer).strip().lower()
            # by number in label, e.g., DLA_3_talk -> 3
            m = re.findall(r"\d+", ans)
            if m:
                n = int(m[0])
                for (_, i0, j0, lbl0) in candidates:
                    m2 = re.search(r"DLA_(\d+)_", lbl0, flags=re.IGNORECASE)
                    if m2 and int(m2.group(1)) == n:
                        return (i0, j0, lbl0)
            # by label keyword (tail), e.g., 'talk'
            for (_, i0, j0, lbl0) in candidates:
                parts = lbl0.split("_", 2)
                tail = parts[2] if len(parts) >= 3 else lbl0
                if tail.lower() in ans or lbl0.lower() in ans:
                    return (i0, j0, lbl0)
            return None

        chosen = _pick_candidate(resp)
        if chosen is None:
            # one retry to clarify
            opts = "; ".join([f"{k}) {lbl}" for (k, _, _, lbl) in candidates])
            log_question(f"Please choose ONE by number or label among: {opts}")
            resp = get_resp_log()
            if isinstance(resp, str) and resp.strip().lower().find("stop") != -1:
                logger.info("User requested stop at CBT stage 0 retry.")
                return
            chosen = _pick_candidate(resp)
            if chosen is None:
                logger.info("Failed to parse user choice for CBT stage 0. Exit CBT.")
                log_question("I could not determine your choice. We will stop CBT for now.")
                return

        i_sel, j_sel, label_sel = chosen
        logger.info(f"CBT dimension chosen: [{label_sel}] at ({i_sel},{j_sel}).")

        # Stage 1: collect statement and unhelpful thoughts
        log_question(f"For the dimension '{label_sel}', please briefly describe the situation you want to work on (1-3 sentences).")
        statement = get_resp_log()
        if isinstance(statement, str) and statement.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT before stage 1 question.")
            return

        log_question("Can you try to identify any unhelpful thoughts you have that contribute to this situation?")
        unhelpful = get_resp_log()
        if isinstance(unhelpful, str) and unhelpful.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT stage 1.")
            return

        # Reason and guide up to two retries
        dec1_raw = stage1_reasoner(statement, unhelpful)
        dec1 = "0" if "0" in dec1_raw else "1"
        retry = 0
        while dec1 == "1" and retry < 2:
            guide1 = stage1_guide(statement)
            log_question(guide1)
            log_question("Please provide your UNHELPFUL_THOUGHTS again, in one sentence.")
            unhelpful = get_resp_log()
            if isinstance(unhelpful, str) and unhelpful.strip().lower().find("stop") != -1:
                logger.info("User requested stop during CBT stage 1 retry.")
                return
            dec1_raw = stage1_reasoner(statement, unhelpful)
            dec1 = "0" if "0" in dec1_raw else "1"
            retry += 1
        if dec1 == "1":
            log_question("It seems difficult to identify the unhelpful thoughts right now. Let's pause CBT and revisit later.")
            # record brief CBT notes
            self.question_lib[str(i_sel)][str(j_sel)]["notes"].append([
                f"CBT_dimension: {label_sel}",
                f"CBT_statement: {statement}",
                f"CBT_unhelpful_thoughts: {unhelpful}",
                "CBT_stage: 1_failed"
            ])
            return

        # Stage 2: challenge the unhelpful thoughts
        log_question("Now, how could you challenge those unhelpful thoughts? Please write a brief challenge.")
        challenge = get_resp_log()
        if isinstance(challenge, str) and challenge.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT stage 2.")
            return

        dec2_raw = stage2_reasoner(statement, unhelpful, challenge)
        dec2 = "0" if "0" in dec2_raw else "1"
        retry = 0
        while dec2 == "1" and retry < 2:
            guide2 = stage2_guide(statement, unhelpful)
            log_question(guide2)
            log_question("Please try to CHALLENGE the unhelpful thoughts again, in one sentence.")
            challenge = get_resp_log()
            if isinstance(challenge, str) and challenge.strip().lower().find("stop") != -1:
                logger.info("User requested stop during CBT stage 2 retry.")
                return
            dec2_raw = stage2_reasoner(statement, unhelpful, challenge)
            dec2 = "0" if "0" in dec2_raw else "1"
            retry += 1
        if dec2 == "1":
            log_question("Challenging the thought seems difficult now. Let's pause CBT and revisit later.")
            self.question_lib[str(i_sel)][str(j_sel)]["notes"].append([
                f"CBT_dimension: {label_sel}",
                f"CBT_statement: {statement}",
                f"CBT_unhelpful_thoughts: {unhelpful}",
                f"CBT_challenge: {challenge}",
                "CBT_stage: 2_failed"
            ])
            return

        # Stage 3: reframe the thought
        log_question("Finally, can you reframe the unhelpful thought into a more balanced, constructive one?")
        reframe = get_resp_log()
        if isinstance(reframe, str) and reframe.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT stage 3.")
            return

        dec3_raw = stage3_reasoner(statement, unhelpful, challenge, reframe)
        dec3 = "0" if "0" in dec3_raw else "1"
        retry = 0
        while dec3 == "1" and retry < 2:
            guide3 = stage3_guide(statement, unhelpful, challenge)
            log_question(guide3)
            log_question("Please REFRAME again in one or two sentences.")
            reframe = get_resp_log()
            if isinstance(reframe, str) and reframe.strip().lower().find("stop") != -1:
                logger.info("User requested stop during CBT stage 3 retry.")
                return
            dec3_raw = stage3_reasoner(statement, unhelpful, challenge, reframe)
            dec3 = "0" if "0" in dec3_raw else "1"
            retry += 1
        if dec3 == "1":
            log_question("Reframing seems hard right now. Let's pause CBT and revisit later.")
            self.question_lib[str(i_sel)][str(j_sel)]["notes"].append([
                f"CBT_dimension: {label_sel}",
                f"CBT_statement: {statement}",
                f"CBT_unhelpful_thoughts: {unhelpful}",
                f"CBT_challenge: {challenge}",
                f"CBT_reframe: {reframe}",
                "CBT_stage: 3_failed"
            ])
            return

        # Success
        self.question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            f"CBT_challenge: {challenge}",
            f"CBT_reframe: {reframe}",
            "CBT_stage: success"
        ])
        log_question("Great work today. We completed the CBT steps for this topic. Thank you for your effort.")

    def run(self):
        """
        Main RL loop for the entire screening process.
        Iteratively selects items and asks questions using RL, updating Q-tables and saving results.
        """
        logger.info("Starting main RL screening process.")
        self.setup()
        new_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        # Mask for available items (first item is always available)
        item_mask = [0] + [1] * (ITEM_N_STATES - 1)
        while not is_terminated:
            # Select an item to ask about using RL policy
            A = choose_action(S, self.item_q_table, item_mask, ITEM_N_STATES, self.item_actions)
            # Mark this item as used
            item_mask[int(A)] = 0
            # Ask questions for the selected item
            openai_res, DLA_terminate, last_question_updated = self.ask_question(int(A))
            self.last_question = last_question_updated
            # Get next state and reward for item RL
            S_, R = get_env_feedback(S, A, openai_res, DLA_terminate, item_mask)
            # Q-learning update for item Q-table
            q_predict = self.item_q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * self.item_q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            new_q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            # If the DLA process signals termination, end the loop and save results
            if DLA_terminate == 1:
                logger.info("DLA process signaled termination. Saving question library and ending session.")
                is_terminated = True
                save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
                save_question_lib(save_filename, self.question_lib)
                # log_question("Goodbye. We will do the screening in another time. 886")
                logger.info("Goodbye. We will do the screening in another time. 886")        # Save results if terminated
        if is_terminated:
            logger.info("Session terminated. Saving question library and generating results.")
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
            
            # Save Q tables (in parallel with existing results)
            logger.info("Saving Q tables.")
            qdir = os.path.join(DATA_DIR, "q_tables")
            self.item_q_table = new_q_table
            if not os.path.exists(qdir):
                os.makedirs(qdir, exist_ok=True)
            self.item_q_table.to_csv(os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv"))

        # Run CBT after the screening loop concludes
        self.run_cbt()
        # Persist question_lib again to capture CBT notes
        save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
        save_question_lib(save_filename, self.question_lib)

        # Generate final results for this session
        generate_results(self.question_lib, self.new_response)