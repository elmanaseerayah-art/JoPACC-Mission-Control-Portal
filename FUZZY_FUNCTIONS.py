import numpy as np
import pandas as pd
import skfuzzy as fuzz

"""
# DATA-BASED RANGE EVIDENCE
def print_range_evidence(csv_path="jopacc_system_month_november_2022_to_june_2026.csv",):
    df = pd.read_csv(csv_path)
    jomopay_df = df[df["system"] == "JoMoPay"].copy()
    jomopay_df["cash_out_dependency_pct"] = (jomopay_df["tx_cash_out_count"] / jomopay_df["transactions_total"] * 100)

    fuzzy_inputs = ["transactions_total", "wallet_commercial_count", "cash_out_dependency_pct",]
    evidence = jomopay_df[fuzzy_inputs].quantile([0.00, 0.33, 0.50, 0.66, 1.00])
    evidence.index = [
        "Minimum",
        "33% Quantile",
        "Median",
        "66% Quantile",
        "Maximum",
    ]
    print("\n--- JoMoPay Fuzzy Input Range Evidence ---")
    print(evidence.round(2))
"""
# UNIVERSES
activity_universe = np.arange(0, 15_000_001, 10_000)
wallet_universe = np.arange(5_000, 8_001, 10)
cash_out_universe = np.arange(0, 10.1, 0.1)
growth_universe = np.arange(0, 101, 1)

# INPUT MEMBERSHIP FUNCTIONS
# Evidence: activity min=1.26M, median=5.09M, max=14.98M
activity_low = fuzz.trimf(activity_universe, [0, 0, 3_500_000])
activity_medium = fuzz.trimf(activity_universe, [3_000_000, 5_100_000, 7_500_000])
activity_high = fuzz.trimf(activity_universe, [6_500_000, 15_000_000, 15_000_000])

# Evidence: commercial wallets min=5,334, median=6,445.5, max=7,600
wallet_low = fuzz.trimf(wallet_universe, [5_000, 5_000, 6_300])
wallet_medium = fuzz.trimf(wallet_universe, [6_100, 6_450, 6_800])
wallet_high = fuzz.trimf(wallet_universe, [6_600, 8_000, 8_000])

# Evidence: cash-out dependency min=0.40%, median=3.43%, max=8.74%
cash_out_low = fuzz.trimf(cash_out_universe, [0, 0, 2.5])
cash_out_medium = fuzz.trimf(cash_out_universe, [2, 3.5, 5])
cash_out_high = fuzz.trimf(cash_out_universe, [4.5, 10, 10])

# OUTPUT MEMBERSHIP FUNCTIONS: GROWTH QUALITY SCORE 0-100
growth_fragile = fuzz.trimf(growth_universe, [0, 0, 30])
growth_unbalanced = fuzz.trimf(growth_universe, [20, 40, 60])
growth_healthy = fuzz.trimf(growth_universe, [45, 65, 85])
growth_sustainable = fuzz.trimf(growth_universe, [70, 100, 100])

def fuzzify(value, universe, membership_function):
    return fuzz.interp_membership(universe, membership_function, value)

def apply_rules(transaction_activity, commercial_wallets, cash_out_dependency):
    # Fuzzify the three inputs.
    a_low = fuzzify(transaction_activity, activity_universe, activity_low)
    a_medium = fuzzify(transaction_activity, activity_universe, activity_medium)
    a_high = fuzzify(transaction_activity, activity_universe, activity_high)

    w_low = fuzzify(commercial_wallets, wallet_universe, wallet_low)
    w_medium = fuzzify(commercial_wallets, wallet_universe, wallet_medium)
    w_high = fuzzify(commercial_wallets, wallet_universe, wallet_high)

    c_low = fuzzify(cash_out_dependency, cash_out_universe, cash_out_low)
    c_medium = fuzzify(cash_out_dependency, cash_out_universe, cash_out_medium)
    c_high = fuzzify(cash_out_dependency, cash_out_universe, cash_out_high)

    low_or_medium_cash = np.fmax(c_low, c_medium)
    medium_or_high_wallets = np.fmax(w_medium, w_high)

    # Fragile Growth rules
    rule1 = np.fmin(a_low, w_low)
    rule2 = np.fmin(a_low, c_high)
    rule3 = np.fmin(w_low, c_high)
    rule4 = np.fmin(a_medium, w_low)

    # Unbalanced Growth rules
    rule5 = np.fmin(a_high, c_high)
    rule6 = np.fmin(a_medium, c_high)
    rule7 = np.fmin(a_high, w_low)
    rule8 = np.fmin(a_low, medium_or_high_wallets)

    # Healthy Growth rules
    rule9 = np.fmin(a_medium, np.fmin(w_medium, low_or_medium_cash))
    rule10 = np.fmin(a_medium, np.fmin(w_high, low_or_medium_cash))
    rule11 = np.fmin(a_high, np.fmin(w_medium, low_or_medium_cash))
    rule12 = np.fmin(a_high, np.fmin(w_high, c_medium))

    # Sustainable Growth rule
    rule13 = np.fmin(a_high, np.fmin(w_high, c_low))

    fragile = max(rule1, rule2, rule3, rule4)
    unbalanced = max(rule5, rule6, rule7, rule8)
    healthy = max(rule9, rule10, rule11, rule12)
    sustainable = rule13

    rule_strengths = {
        "Rule 1: Low activity and low wallets produce Fragile Growth": rule1,
        "Rule 2: Low activity and high cash-out dependency produce Fragile Growth": rule2,
        "Rule 3: Low wallets and high cash-out dependency produce Fragile Growth": rule3,
        "Rule 4: Medium activity with low wallets produces Fragile Growth": rule4,
        "Rule 5: High activity with high cash-out dependency produces Unbalanced Growth": rule5,
        "Rule 6: Medium activity with high cash-out dependency produces Unbalanced Growth": rule6,
        "Rule 7: High activity with low wallets produces Unbalanced Growth": rule7,
        "Rule 8: Low activity despite medium or high wallets produces Unbalanced Growth": rule8,
        "Rule 9: Medium activity, medium wallets, and low or medium cash-out dependency produce Healthy Growth": rule9,
        "Rule 10: Medium activity, high wallets, and low or medium cash-out dependency produce Healthy Growth": rule10,
        "Rule 11: High activity, medium wallets, and low or medium cash-out dependency produce Healthy Growth": rule11,
        "Rule 12: High activity, high wallets, and medium cash-out dependency produce Healthy Growth": rule12,
        "Rule 13: High activity, high wallets, and low cash-out dependency produce Sustainable Growth": rule13,
    }

    activated_rules = [{"rule": name, "strength": round(float(strength), 3)} for name, strength in rule_strengths.items() if strength > 0]
    activated_rules.sort(key=lambda item: item["strength"], reverse=True)

    return fragile, unbalanced, healthy, sustainable, activated_rules

def aggregate(fragile, unbalanced, healthy, sustainable):
    fragile_cut = np.fmin(fragile, growth_fragile)
    unbalanced_cut = np.fmin(unbalanced, growth_unbalanced)
    healthy_cut = np.fmin(healthy, growth_healthy)
    sustainable_cut = np.fmin(sustainable, growth_sustainable)
    return np.fmax.reduce([fragile_cut, unbalanced_cut, healthy_cut, sustainable_cut])

def centroid_defuzz(aggregated):
    if np.max(aggregated) == 0:
        return 0.0
    return fuzz.defuzz(growth_universe, aggregated, "centroid")

def get_growth_level(score):
    memberships = {
        "Fragile Growth": fuzz.interp_membership(growth_universe, growth_fragile, score),
        "Unbalanced Growth": fuzz.interp_membership(growth_universe, growth_unbalanced, score),
        "Healthy Growth": fuzz.interp_membership(growth_universe, growth_healthy, score),
        "Sustainable Growth": fuzz.interp_membership(growth_universe, growth_sustainable, score),
    }
    return max(memberships, key=memberships.get)

def create_explanation(growth_level, activated_rules, cash_out_dependency):
    if not activated_rules:
        return "No fuzzy rule was activated for the entered values."

    strongest_rule = activated_rules[0]
    return (f"The result is {growth_level}. The strongest influencing rule was: "
            f"{strongest_rule['rule']}. The calculated cash-out dependency was "
            f"{cash_out_dependency:.2f}%."
    )

def calculate_growth_quality(transaction_activity, commercial_wallets, cash_out_count):
    transaction_activity = float(transaction_activity)
    commercial_wallets = float(commercial_wallets)
    cash_out_count = float(cash_out_count)

    if transaction_activity <= 0:
        raise ValueError("Transaction activity must be greater than zero.")
    if commercial_wallets < 0:
        raise ValueError("Commercial wallets cannot be negative.")
    if cash_out_count < 0:
        raise ValueError("Cash-out count cannot be negative.")

    cash_out_dependency = cash_out_count / transaction_activity * 100

    transaction_activity = float(np.clip(transaction_activity, activity_universe.min(), activity_universe.max(),))
    commercial_wallets = float(np.clip(commercial_wallets, wallet_universe.min(), wallet_universe.max(),))
    cash_out_dependency = float(np.clip(cash_out_dependency, cash_out_universe.min(), cash_out_universe.max(),))

    fragile, unbalanced, healthy, sustainable, activated_rules = apply_rules(transaction_activity, commercial_wallets, cash_out_dependency)
    aggregated = aggregate(fragile, unbalanced, healthy, sustainable)
    score = round(float(centroid_defuzz(aggregated)), 2)
    growth_level = get_growth_level(score)
    explanation = create_explanation(growth_level, activated_rules, cash_out_dependency,)

    return {
        "growth_quality_score": score,
        "growth_quality_level": growth_level,
        "cash_out_dependency": round(cash_out_dependency, 2),
        "activated_rules": activated_rules,
        "explanation": explanation,
    }
