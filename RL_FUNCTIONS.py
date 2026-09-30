import pickle

# LOAD SAVED RL POLICY
with open("models/best_rl_policy.pkl", "rb") as file:
    policy_data = pickle.load(file)

best_Q = policy_data["Q"]
best_policy_name = policy_data["policy_name"]
best_reward_name = policy_data["reward_name"]
system_features = policy_data["system_features"]
system_thresholds = policy_data["system_thresholds"]
feature_directions = policy_data["feature_directions"]
actions = policy_data["actions"]
action_costs = policy_data["action_costs"]
activity_encoding = policy_data["activity_encoding"]
activity_decoding = policy_data["activity_decoding"]
state_to_index = policy_data["state_to_index"]

# LEVEL NAMES
level_names = {0: "Low", 1: "Medium", 2: "High"}

budget_names = {0: "Insufficient", 1: "Low", 2: "Medium", 3: "High"}

n_actions = 4

# CONVERT VALUE TO LEVEL
def convert_to_level(value, thresholds):
    if value <= thresholds[0]:
        return 0
    elif value <= thresholds[1]:
        return 1
    else:
        return 2


# BUDGET LEVEL
def get_budget_level(budget):
    if budget < 5:
        return 0
    elif budget < 15:
        return 1
    elif budget < 30:
        return 2
    else:
        return 3


# FEATURE QUALITY
def feature_quality(level, direction):
    if direction == 1:
        return level
    else:
        return 2 - level


# STATE QUALITY
def state_quality(state):
    system, activity, indicator_1, indicator_2, budget_level = state
    directions = feature_directions[system]
    return (
        activity * 2
        + feature_quality(indicator_1, directions[0])
        + feature_quality(indicator_2, directions[1])
    )


# IMPROVE INDICATOR
def improve_indicator(level, direction):
    if direction == 1:
        return min(level + 1, 2)
    else:
        return max(level - 1, 0)


# TARGET CONDITION
def is_target_state(state):
    system, activity, indicator_1, indicator_2, budget_level = state
    directions = feature_directions[system]
    indicator_1_good = feature_quality(indicator_1, directions[0]) >= 1
    indicator_2_good = feature_quality(indicator_2, directions[1]) >= 1
    return activity == 2 and indicator_1_good and indicator_2_good


# CREATE STATE
def create_state(system, activity, feature_values, budget):
    features = system_features[system]
    indicator_1 = convert_to_level(feature_values[features[0]], system_thresholds[system][features[0]])
    indicator_2 = convert_to_level(feature_values[features[1]], system_thresholds[system][features[1]])
    return (system, activity_encoding[activity], indicator_1, indicator_2, get_budget_level(budget))


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


# STATE TRANSITION
def simulate_transition(state, action, remaining_budget):
    system, activity, indicator_1, indicator_2, budget_level = state
    directions = feature_directions[system]
    cost = action_costs[system][action]
    if cost > remaining_budget:
        return {
            "next_state": state,
            "cost": 0,
            "remaining_budget": remaining_budget,
            "improvement": 0,
            "budget_violation": True,
        }
    old_quality = state_quality(state)
    if action == 0:
        pass
    elif action == 1:
        quality_1 = feature_quality(indicator_1, directions[0])
        quality_2 = feature_quality(indicator_2, directions[1])
        if quality_1 <= quality_2:
            indicator_1 = improve_indicator(indicator_1, directions[0])
        else:
            indicator_2 = improve_indicator(indicator_2, directions[1])
    elif action == 2:
        indicator_1 = improve_indicator(indicator_1, directions[0])
    elif action == 3:
        indicator_2 = improve_indicator(indicator_2, directions[1])
    quality_1 = feature_quality(indicator_1, directions[0])
    quality_2 = feature_quality(indicator_2, directions[1])
    
    # Activity can improve only after an actual intervention.
    # Continue Monitoring (action 0) must not change the state.
    if action != 0 and quality_1 >= 1 and quality_2 >= 1:
        activity = min(activity + 1, 2)
    remaining_budget = remaining_budget - cost
    next_state = (system, activity, indicator_1, indicator_2, get_budget_level(remaining_budget))
    improvement = state_quality(next_state) - old_quality
    return {
        "next_state": next_state,
        "cost": cost,
        "remaining_budget": remaining_budget,
        "improvement": improvement,
        "budget_violation": False,
    }


# REWARD FUNCTIONS
def reward_design_1(transition):
    if transition["budget_violation"]:
        return -25
    improvement = transition["improvement"]
    cost = transition["cost"]
    reward = improvement * 10 - cost * 0.30 - 1
    if improvement == 0:
        reward -= 8
    return float(reward)


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


# SIMULATION INFORMATION
simulation_note = (
    "The intervention actions, costs, rewards, "
    "state transitions, and outcomes are simulated "
    "because the JoPACC dataset does not contain "
    "actual intervention records."
)

print("Loaded RL policy:", best_policy_name)
