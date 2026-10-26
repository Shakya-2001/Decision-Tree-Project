from decision_tree import entropy, information_gain

# Test 1: Play Tennis
y1 = ["Yes"] * 9 + ["No"] * 5

print("Play Tennis Entropy:", round(entropy(y1), 4))

# Test 2: Pure dataset
y2 = ["Yes", "Yes", "Yes", "Yes"]

print("Pure Dataset Entropy:", round(entropy(y2), 4))

# Test 3: Balanced dataset
y3 = ["Yes", "Yes", "No", "No"]

print("Balanced Dataset Entropy:", round(entropy(y3), 4))

# Test 4: Multiclass dataset
y4 = ["Cat", "Dog", "Bird", "Cat", "Dog", "Bird"]

print("Multiclass Entropy:", round(entropy(y4), 4))

# Parent: 9 Yes, 5 No
parent = ["Yes"] * 9 + ["No"] * 5

# Example split: Humidity = High vs Normal
high = ["Yes"] * 3 + ["No"] * 4
normal = ["Yes"] * 6 + ["No"] * 1

gain = information_gain(parent, high, normal)

print("Humidity Information Gain:", round(gain, 4))