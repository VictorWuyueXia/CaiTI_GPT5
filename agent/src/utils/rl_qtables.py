import numpy as np
import pandas as pd
from src.utils.config_loader import ITEM_N_STATES, ITEM_IMPORTANCE, EPSILON, GAMMA, ALPHA

def build_q_table(n_states, actions):
    """
    Build a Q-table as a pandas DataFrame with zeros.
    Each row is a state, each column is an action.
    """
    return pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)

def initialize_question_q_table(n_states, actions):
    """
    Initialize a Q-table for questions.
    For each state (except the first), set all Q-values to 1.
    This can help encourage exploration at the beginning.
    """
    t = build_q_table(n_states, actions)
    for i in range(1, n_states):
        t[str(i)] = t[str(i)].apply(lambda x: x + 1)
    return t

def initialize_q_table(n_states, actions):
    """
    Initialize a Q-table for items.
    For each state, set all Q-values to the corresponding item importance.
    This biases the Q-table according to item importance.
    """
    t = build_q_table(n_states, actions)
    for i in range(0, n_states):
        t[str(i)] = t[str(i)].apply(lambda x: x + ITEM_IMPORTANCE[i])
    return t

def choose_action(state, q_table, mask, number_states, actions):
    """
    Choose an action based on the current state and Q-table.
    Mask out unavailable actions by multiplying their Q-values by 0.
    With probability EPSILON, choose the best action; otherwise, explore.
    """
    state_action = q_table.iloc[state, :]
    # Apply mask to Q-table to disable unavailable actions
    for i in range(1, number_states):
        q_table[str(i)] = q_table[str(i)].apply(lambda x: x * mask[i])
    # Exploration: with probability 1-EPSILON or if all Q-values are zero, pick randomly
    if (np.random.uniform() > EPSILON) or ((state_action == 0).all()):
        return np.random.choice(actions[1:number_states])
    else:
        # Exploitation: pick the action(s) with the highest Q-value
        return np.random.choice(state_action[state_action == np.max(state_action)].index)

def get_env_feedback(S, A, reward, terminate_flag, item_mask):
    """
    Get the next state and reward from the environment.
    If all items are masked (no available actions), return terminal state and reward 10.
    If terminate_flag is set, return terminal state and reward 0.
    Otherwise, return the next state (action taken) and the given reward.
    """
    if sum(item_mask) == 0:
        return 'terminal', 10
    elif terminate_flag == 1:
        return 'terminal', 0
    else:
        return int(A), reward

def initialize_question_mask(number_questions):
    """
    Initialize a mask for each question.
    For each item with more than one question, create a mask list:
    The first element is 0 (possibly a placeholder), followed by 1s for each question.
    """
    all_mask = {}
    for i in range(0, ITEM_N_STATES):
        if number_questions[i] > 1:
            all_mask[i] = [0] + [1] * number_questions[i]
    return all_mask

def initialize_question_table(number_questions):
    """
    Initialize a Q-table for questions.
    For each item with more than one question, create a Q-table.
    """
    all_question_q_table = {}
    for i in range(0, ITEM_N_STATES):
        if number_questions[i] > 1:
            question_actions = ['{0}'.format(e) for e in np.arange(number_questions[i] + 1)]
            qtab = initialize_question_q_table(number_questions[i] + 1, question_actions)
            all_question_q_table[i] = qtab
    return all_question_q_table


