# Auto-generated calculators module
# Contains all calculator functions for data mining concepts

import math


def bayes_calculator():
    """Interactive Bayes theorem calculator."""
    print("Bayes Calculator")
    try:
        prior = float(input("Enter prior probability P(H): "))
        likelihood = float(input("Enter likelihood P(E|H): "))
        evidence = float(input("Enter evidence P(E): "))
        posterior = (prior * likelihood) / evidence
        print(f"Posterior probability P(H|E): {posterior}")
    except ValueError:
        print("Invalid input. Please enter numbers.")
    except ZeroDivisionError:
        print("Error: Evidence P(E) cannot be zero.")


def euclidean_distance_calculator():
    """Interactive Euclidean distance calculator."""
    print("Euclidean Distance Calculator")
    try:
        n = int(input("Enter number of dimensions: "))
        point1 = []
        point2 = []
        for i in range(n):
            p1 = float(input(f"Enter coordinate {i+1} for point 1: "))
            p2 = float(input(f"Enter coordinate {i+1} for point 2: "))
            point1.append(p1)
            point2.append(p2)
        distance = sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
        print(f"Euclidean distance: {distance}")
    except ValueError:
        print("Invalid input. Please enter numbers.")


def manhattan_distance_calculator():
    """Interactive Manhattan distance calculator."""
    print("Manhattan Distance Calculator")
    try:
        n = int(input("Enter number of dimensions: "))
        point1 = []
        point2 = []
        for i in range(n):
            p1 = float(input(f"Enter coordinate {i+1} for point 1: "))
            p2 = float(input(f"Enter coordinate {i+1} for point 2: "))
            point1.append(p1)
            point2.append(p2)
        distance = sum(abs(a - b) for a, b in zip(point1, point2))
        print(f"Manhattan distance: {distance}")
    except ValueError:
        print("Invalid input. Please enter numbers.")


def entropy_calculator():
    """Interactive entropy calculator."""
    print("Entropy Calculator")
    try:
        probs = []
        while True:
            prob = input("Enter probability (or 'done'): ")
            if prob.lower() == "done":
                break
            probs.append(float(prob))
        if not probs:
            print("No probabilities entered.")
            return
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        print(f"Entropy: {entropy}")
    except ValueError:
        print("Invalid input. Please enter numbers.")


def gini_calculator():
    """Interactive Gini impurity calculator."""
    print("Gini Impurity Calculator")
    try:
        probs = []
        while True:
            prob = input("Enter class probability (or 'done'): ")
            if prob.lower() == "done":
                break
            probs.append(float(prob))
        if not probs:
            print("No probabilities entered.")
            return
        gini = 1 - sum(p ** 2 for p in probs)
        print(f"Gini impurity: {gini}")
    except ValueError:
        print("Invalid input. Please enter numbers.")


def information_gain_calculator():
    """Interactive information gain calculator."""
    print("Information Gain Calculator")
    try:
        parent_entropy = float(input("Enter parent entropy: "))
        n_children = int(input("Enter number of child nodes: "))
        total_samples = int(input("Enter total samples in parent: "))
        
        weighted_child_entropy = 0
        for i in range(n_children):
            samples = int(input(f"Enter samples in child {i+1}: "))
            entropy = float(input(f"Enter entropy of child {i+1}: "))
            weighted_child_entropy += (samples / total_samples) * entropy
        
        info_gain = parent_entropy - weighted_child_entropy
        print(f"Information Gain: {info_gain}")
    except ValueError:
        print("Invalid input. Please enter numbers.")
    except ZeroDivisionError:
        print("Error: Total samples cannot be zero.")


def execute_math():
    """Simple math expression evaluator."""
    print("Math Executor")
    print("Available functions: sqrt, sin, cos, tan, log, log2, log10, exp, pow, abs")
    expr = input("Enter math expression: ")
    try:
        # Create a safe namespace with math functions
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "exp": math.exp,
            "pow": pow,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Invalid expression: {e}")
