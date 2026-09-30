import pandas as pd
import numpy as np
import itertools
import random
import pickle
import os

# LOAD DATA
df = pd.read_csv("jopacc_system_month_november_2022_to_june_2026.csv")
#########################################
# Total number of Foreign ACH transactions
df["ach_foreign_transaction_count"] = (
    df["ach_usd_transaction_count"] + df["ach_eur_transaction_count"] + df["ach_gbp_transaction_count"]
)
#########################################
# SYSTEMS
systems = ["CliQ", "JoMoPay", "eFAWATEERcom", "ACH", "ECCU"]
#########################################
# ACTIVITY LEVEL
activity_encoding = {"Low Activity": 0, "Normal Activity": 1, "High Activity": 2}
activity_decoding = {0: "Low Activity", 1: "Normal Activity", 2: "High Activity"}
# SYSTEM FEATURES
system_features = {
    "CliQ": [
        "customer_legal_entity_count",
        "tx_money_transfers_count",
    ],
    "JoMoPay": [
        "wallet_commercial_count",
        "tx_cash_out_count",
    ],
    "eFAWATEERcom": [
        "ef_digital_payment_count",
        "ef_cash_payment_count",
    ],
    "ACH": ["ach_jod_transaction_count", "ach_foreign_transaction_count"],
    "ECCU": [
        "returned_cheque_count_rate_pct",
        "returned_cheque_value_rate_pct",
    ],
}
# 1 higher: better
# -1 lower: better
feature_directions = {
    "CliQ": [1, 1],
    "JoMoPay": [1, -1],
    "eFAWATEERcom": [1, -1],
    "ACH": [1, 1],
    "ECCU": [-1, -1],
}
#########################################
# THRESHOLDS
system_thresholds = {}
for system, features in system_features.items():
    system_df = df[df["system"] == system]
    thresholds = {}
    for feature in features:
        _, bins = pd.qcut(system_df[feature], q=3, retbins=True, duplicates="drop")  ## LOW, HIGH, MED
        if len(bins) == 4:
            low_threshold = float(bins[1])
            high_threshold = float(bins[2])
        else:
            low_threshold = float(system_df[feature].quantile(0.33))
            high_threshold = float(system_df[feature].quantile(0.66))
        thresholds[feature] = [low_threshold, high_threshold]
    system_thresholds[system] = thresholds


def convert_to_level(value, thresholds):
    if value <= thresholds[0]:
        return 0  # LOW
    elif value <= thresholds[1]:
        return 1  # MED
    else:
        return 2  # HIGH


level_names = {0: "Low", 1: "Medium", 2: "High"}


#########################################
# BUDGET LEVEL
def get_budget_level(budget):
    if budget < 5:
        return 0  # Insufficient
    elif budget < 15:
        return 1  # Low
    elif budget < 30:
        return 2  # Mid
    else:
        return 3  # High


budget_values = {0: 2, 1: 10, 2: 22, 3: 40}
budget_names = {0: "Insufficient", 1: "Low", 2: "Medium", 3: "High"}
#########################################
# ACTIONS
actions = {
    "CliQ": {
        0: "Continue Monitoring",
        1: "Perform Low-Cost Operational Review",
        2: "Increase Legal-Entity Participation",
        3: "Promote CliQ Money Transfers",
    },
    "JoMoPay": {
        0: "Continue Monitoring",
        1: "Perform Low-Cost Operational Review",
        2: "Increase Commercial Wallet Usage",
        3: "Reduce Cash-Out Dependency",
    },
    "eFAWATEERcom": {
        0: "Continue Monitoring",
        1: "Perform Low-Cost Operational Review",
        2: "Increase Digital Bill Payments",
        3: "Reduce Cash Bill Payments",
    },
    "ACH": {
        0: "Continue Monitoring",
        1: "Perform Low-Cost Operational Review",
        2: "Improve JOD Settlement Activity",
        3: "Support Foreign-Currency Transfers",
    },
    "ECCU": {
        0: "Continue Monitoring",
        1: "Perform Low-Cost Operational Review",
        2: "Reduce Returned-Cheque Count Risk",
        3: "Reduce Returned-Cheque Value Risk",
    },
}
# ACTION COSTS
action_costs = {
    "CliQ": {0: 0, 1: 5, 2: 12, 3: 10},
    "JoMoPay": {0: 0, 1: 5, 2: 12, 3: 10},
    "eFAWATEERcom": {0: 0, 1: 5, 2: 12, 3: 10},
    "ACH": {0: 0, 1: 5, 2: 12, 3: 15},
    "ECCU": {0: 0, 1: 5, 2: 10, 3: 12},
}
n_actions = 4
#########################################
all_states = list(
    itertools.product(systems, range(3), range(3), range(3), range(4))
)  # 540 states × 4 actions
state_to_index = {state: index for index, state in enumerate(all_states)}


#########################################
def feature_quality(level, direction):
    if direction == 1:
        return level
    else:
        return 2 - level


#########################################
# STATE QUALITY
def state_quality(state):
    system, activity, indicator_1, indicator_2, budget_level = state
    directions = feature_directions[system]
    return (
        activity * 2
        + feature_quality(indicator_1, directions[0])
        + feature_quality(indicator_2, directions[1])
    )

#########################################
def improve_indicator(level, direction):
    if direction == 1:
        return min(level + 1, 2) #Low → Medium → High
    else:
        return max(level - 1, 0) #High → Medium → Low

#########################################
def is_target_state(state):
    system, activity, indicator_1, indicator_2, budget_level = state
    directions = feature_directions[system]
    indicator_1_good = feature_quality(indicator_1, directions[0]) >= 1
    indicator_2_good = feature_quality(indicator_2, directions[1]) >= 1
    return activity == 2 and indicator_1_good and indicator_2_good

#########################################
# STATE TRANSITION
def simulate_transition(state, action, remaining_budget):
    system, activity, indicator_1, indicator_2, budget_level = state
    directions = feature_directions[system]
    cost = action_costs[system][action]
    # Action is above the budget
    if cost > remaining_budget:
        return {
            "next_state": state,
            "cost": 0,
            "remaining_budget": remaining_budget,
            "improvement": 0,
            "budget_violation": True,
        }
    old_quality = state_quality(state)
    # Continue Monitoring
    if action == 0:
        pass
    # Improve the weakest indicator
    elif action == 1:
        quality_1 = feature_quality(indicator_1, directions[0])
        quality_2 = feature_quality(indicator_2, directions[1])
        if quality_1 <= quality_2:
            indicator_1 = improve_indicator(indicator_1, directions[0])
        else:
            indicator_2 = improve_indicator(indicator_2, directions[1])
    # Improve the first system indicator
    elif action == 2:
        indicator_1 = improve_indicator(indicator_1, directions[0])
    # Improve the second system indicator
    elif action == 3:
        indicator_2 = improve_indicator(indicator_2, directions[1])
    # Improve activity when both indicators
    # reach acceptable levels
    quality_1 = feature_quality(indicator_1, directions[0])
    quality_2 = feature_quality(indicator_2, directions[1])
    if action != 0 and quality_1 >= 1 and quality_2 >= 1:
        activity = min(activity + 1, 2)
    remaining_budget = remaining_budget - cost

    next_state = (system, activity, indicator_1, indicator_2, get_budget_level(remaining_budget),)
    
    improvement = state_quality(next_state) - old_quality
    return {
        "next_state": next_state,
        "cost": cost,
        "remaining_budget": remaining_budget,
        "improvement": improvement,
        "budget_violation": False,
    }

#########################################
# REWARD DESIGN 1
# IMPROVEMENT-FOCUSED
def reward_design_1(transition):
    if transition["budget_violation"]:
        return -25
    improvement = transition["improvement"]
    cost = transition["cost"]
    reward = improvement * 10 - cost * 0.30 - 1
    if improvement == 0:
        reward -= 8
    return float(reward)

#########################################
# REWARD DESIGN 2
# COST-AWARE
def reward_design_2(transition):
    if transition["budget_violation"]:
        return -30
    improvement = transition["improvement"]
    cost = transition["cost"]
    reward = improvement * 10 - cost * 0.70 - 2
    if improvement == 0:
        reward -= 10
    if improvement > 0 and cost <= 10:
        reward += 4
    return float(reward)

#########################################
# Q-TABLE
Q_1 = np.zeros((len(all_states), n_actions))
Q_2 = np.zeros((len(all_states), n_actions))

#########################################
# TRAINING
def train_policy(Q, reward_function, episodes=10000):
    alpha = 0.10
    gamma = 0.90
    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05
    max_steps = 5
    for episode in range(episodes):
        state = random.choice(all_states)
        remaining_budget = budget_values[state[4]]
        for step in range(max_steps):
            if is_target_state(state):
                break
            state_index = state_to_index[state]
            # Epsilon-greedy
            # Select only affordable actions
            affordable_actions = [
                action for action in range(n_actions) if action_costs[state[0]][action] <= remaining_budget
            ]
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(affordable_actions)
            else:
                action = max(affordable_actions, key=lambda action_id: Q[state_index, action_id])
            transition = simulate_transition(state, action, remaining_budget)
            reward = reward_function(transition)
            next_state = transition["next_state"]
            next_state_index = state_to_index[next_state]
            done = (
                is_target_state(next_state)
                or transition["budget_violation"]
                or transition["remaining_budget"] < 5
                or step == max_steps - 1
            )
            if done:
                target = reward
            else:
                affordable_next_actions = [
                    next_action
                    for next_action in range(n_actions)
                    if action_costs[next_state[0]][next_action] <= transition["remaining_budget"]
                ]
                best_next_q = max(Q[next_state_index, next_action] for next_action in affordable_next_actions)
                target = reward + gamma * best_next_q
            Q[state_index, action] += alpha * (target - Q[state_index, action])
            state = next_state
            remaining_budget = transition["remaining_budget"]
            if transition["budget_violation"]:
                break
            if remaining_budget < 5:
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
    return Q

#########################################
# TRAIN BOTH POLICIES
print("\nTraining Improvement-Focused Policy...")
Q_1 = train_policy(Q_1, reward_design_1)
print("Training Cost-Aware Policy...")
Q_2 = train_policy(Q_2, reward_design_2)

#########################################
# EVALUATION
def evaluate_policy(Q, reward_function, episodes=2000):
    rewards = []
    costs = []
    steps_list = []
    successes = 0
    budget_violations = 0
    for episode in range(episodes):
        state = random.choice(all_states)
        remaining_budget = budget_values[state[4]]
        total_reward = 0
        total_cost = 0
        steps = 0
        for step in range(5):
            if is_target_state(state):
                break
            state_index = state_to_index[state]
            affordable_actions = [
                action for action in range(n_actions) if action_costs[state[0]][action] <= remaining_budget
            ]
            action = max(affordable_actions, key=lambda action_id: Q[state_index, action_id])
            transition = simulate_transition(state, action, remaining_budget)
            reward = reward_function(transition)
            total_reward += reward
            total_cost += transition["cost"]
            steps += 1
            if transition["budget_violation"]:
                budget_violations += 1
                break
            state = transition["next_state"]
            remaining_budget = transition["remaining_budget"]
            if remaining_budget < 5:
                break
        if is_target_state(state):
            successes += 1
        rewards.append(total_reward)
        costs.append(total_cost)
        steps_list.append(steps)
    return {
        "Average Reward": np.mean(rewards),
        "Success Rate": successes / episodes,
        "Average Cost": np.mean(costs),
        "Average Steps": np.mean(steps_list),
        "Budget Violation Rate": budget_violations / episodes,
    }

#########################################
# COMPARE POLICIES
results_1 = evaluate_policy(Q_1, reward_design_1)
results_2 = evaluate_policy(Q_2, reward_design_2)
comparison = pd.DataFrame(
    [
        {"Policy": "Improvement-Focused", **results_1},
        {"Policy": "Cost-Aware", **results_2},
    ]
)
comparison = comparison.sort_values(
    by=["Success Rate", "Average Cost", "Budget Violation Rate"],
    ascending=[False, True, True],
).reset_index(drop=True)
print("\n--- Policy Comparison ---")
print(comparison.to_string(index=False))
# SELECT BEST POLICY
best_row = comparison.iloc[0]
best_policy_name = best_row["Policy"]
if best_policy_name == "Cost-Aware":
    best_Q = Q_2
    best_reward_name = "cost_aware"
else:
    best_Q = Q_1
    best_reward_name = "improvement_focused"
print("\nBest Policy:", best_policy_name)
# SAVE BEST POLICY
os.makedirs("models", exist_ok=True)
policy_data = {
    "Q": best_Q,
    "policy_name": best_policy_name,
    "reward_name": best_reward_name,
    "systems": systems,
    "system_features": system_features,
    "system_thresholds": system_thresholds,
    "feature_directions": feature_directions,
    "actions": actions,
    "action_costs": action_costs,
    "activity_encoding": activity_encoding,
    "activity_decoding": activity_decoding,
    "state_to_index": state_to_index,
    "comparison": comparison.to_dict("records"),
}
with open("models/best_rl_policy.pkl", "wb") as file:
    pickle.dump(policy_data, file)
print("Best policy saved to: " "models/best_rl_policy.pkl")

#########################################
# CREATE STATE
def create_state(system, activity, feature_values, budget):
    features = system_features[system]
    indicator_1 = convert_to_level(feature_values[features[0]], system_thresholds[system][features[0]])
    indicator_2 = convert_to_level(feature_values[features[1]], system_thresholds[system][features[1]])
    return (
        system,
        activity_encoding[activity],
        indicator_1,
        indicator_2,
        get_budget_level(budget),
    )

#########################################
# READABLE STATE
def readable_state(state):
    system, activity, indicator_1, indicator_2, budget_level = state
    features = system_features[system]
    return {
        "system": system,
        "activity": activity_decoding[activity],
        features[0]: level_names[indicator_1],
        features[1]: level_names[indicator_2],
        "budget_level": budget_names[budget_level],
    }

#########################################
# RECOMMEND SEQUENCE
def recommend_sequence(system, activity, feature_values, budget):
    state = create_state(system, activity, feature_values, budget)
    remaining_budget = float(budget)
    sequence = []
    if best_reward_name == "cost_aware":
        reward_function = reward_design_2
    else:
        reward_function = reward_design_1
    for step in range(5):
        if is_target_state(state):
            break
        state_index = state_to_index[state]
        affordable_actions = [
            action for action in range(n_actions) if action_costs[system][action] <= remaining_budget
        ]
        if len(affordable_actions) == 0:
            break
        action = max(affordable_actions, key=lambda action_id: best_Q[state_index, action_id])
        transition = simulate_transition(state, action, remaining_budget)
        reward = reward_function(transition)
        next_state = transition["next_state"]
        sequence.append(
            {
                "step": step + 1,
                "current_state": readable_state(state),
                "recommended_action": actions[system][action],
                "next_state": readable_state(next_state),
                "reward": round(reward, 2),
                "intervention_cost": transition["cost"],
                "remaining_budget": transition["remaining_budget"],
            }
        )
        state = next_state
        remaining_budget = transition["remaining_budget"]
        if transition["improvement"] == 0:
            break
        if remaining_budget < 5:
            break
    return sequence


# SIMULATION
simulation_note = (
    "The intervention actions, costs, rewards, "
    "state transitions, and outcomes are simulated "
    "because the JoPACC dataset does not contain "
    "actual intervention records."
)
