import pandas as pd

# Create the DataFrame
df = pd.DataFrame(
    {
        "income": [
            "very high",
            "high",
            "medium",
            "high",
            "very high",
            "medium",
            "high",
            "medium",
            "high",
            "low",
        ],
        "credit": [
            "excellent",
            "good",
            "excellent",
            "good",
            "good",
            "excellent",
            "bad",
            "bad",
            "bad",
            "bad",
        ],
        "decision": [
            "authorize",
            "authorize",
            "authorize",
            "authorize",
            "authorize",
            "authorize",
            "request id",
            "request id",
            "reject",
            "call police",
        ],
    }
)

# Calculate probabilities for decisions
p_of_authorize = df["decision"].value_counts()["authorize"] / len(df)
print(f"Probability of 'authorize': {p_of_authorize}")

p_of_request_id = df["decision"].value_counts()["request id"] / len(df)
print(f"Probability of 'request id': {p_of_request_id}")

p_of_reject = df["decision"].value_counts()["reject"] / len(df)
print(f"Probability of 'reject': {p_of_reject}")

p_of_call_police = df["decision"].value_counts()["call police"] / len(df)
print(f"Probability of 'call police': {p_of_call_police}")

# Tuple to add ('medium', 'good')
tuple_to_add = ("medium", "good")

# Calculate conditional probabilities for each class
p_of_x_on_c1 = (
    df["income"].value_counts()["medium"] / df[df["decision"] == "authorize"].shape[0]
) * (df["credit"].value_counts()["good"] / df[df["decision"] == "authorize"].shape[0])

print(f"P(X|C1) = {p_of_x_on_c1}")

p_of_x_on_c2 = (
    df["income"].value_counts()["medium"] / df[df["decision"] == "request id"].shape[0]
) * (df["credit"].value_counts()["good"] / df[df["decision"] == "request id"].shape[0])

p_of_x_on_c3 = (
    df["income"].value_counts()["medium"] / df[df["decision"] == "reject"].shape[0]
) * (df["credit"].value_counts()["good"] / df[df["decision"] == "reject"].shape[0])

p_of_x_on_c4 = (
    df["income"].value_counts()["medium"] / df[df["decision"] == "call police"].shape[0]
) * (df["credit"].value_counts()["good"] / df[df["decision"] == "call police"].shape[0])

# Calculate posterior probabilities
p_of_c1_on_x = p_of_authorize * p_of_x_on_c1
p_of_c2_on_x = p_of_request_id * p_of_x_on_c2
p_of_c3_on_x = p_of_reject * p_of_x_on_c3
p_of_c4_on_x = p_of_call_police * p_of_x_on_c4

posterior_probabilities = [p_of_c1_on_x, p_of_c2_on_x, p_of_c3_on_x, p_of_c4_on_x]

# Find the maximum posterior probability
max_prob = max(posterior_probabilities)
print(f"Maximum posterior probability is {max_prob}")

# Initialize i outside the loop
i = 1
for probability in posterior_probabilities:
    if max_prob == probability:
        print(f"Tuple classified into c{i}")
        break
    i += 1
