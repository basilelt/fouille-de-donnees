import math


def display_text(text, max_lines=20):
    """Display text in chunks for limited screen."""
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        print(chunk)
        if i + max_lines < len(lines):
            input("Press Enter to continue...")


# Calculators functions


def bayes_calculator():
    """Interactive Bayes theorem calculator."""
    print("Bayes Calculator")
    try:
        prior = float(input("Enter prior probability P(H): "))
        likelihood = float(input("Enter likelihood P(E|H): "))
        evidence = float(input("Enter evidence P(E): "))
        posterior = (prior * likelihood) / evidence
        print("Posterior probability P(H|E): {}".format(posterior))
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
            p1 = float(input("Enter coordinate {} for point 1: ".format(i + 1)))
            p2 = float(input("Enter coordinate {} for point 2: ".format(i + 1)))
            point1.append(p1)
            point2.append(p2)
        distance = sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
        print("Euclidean distance: {}".format(distance))
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
            p1 = float(input("Enter coordinate {} for point 1: ".format(i + 1)))
            p2 = float(input("Enter coordinate {} for point 2: ".format(i + 1)))
            point1.append(p1)
            point2.append(p2)
        distance = sum(abs(a - b) for a, b in zip(point1, point2))
        print("Manhattan distance: {}".format(distance))
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
        print("Entropy: {}".format(entropy))
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
        gini = 1 - sum(p**2 for p in probs)
        print("Gini impurity: {}".format(gini))
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
            samples = int(input("Enter samples in child {}: ".format(i + 1)))
            entropy = float(input("Enter entropy of child {}: ".format(i + 1)))
            weighted_child_entropy += (samples / total_samples) * entropy
        info_gain = parent_entropy - weighted_child_entropy
        print("Information Gain: {}".format(info_gain))
    except ValueError:
        print("Invalid input. Please enter numbers.")
    except ZeroDivisionError:
        print("Error: Total samples cannot be zero.")


def execute_math():
    """Simple math expression evaluator."""
    print("Math Executor")
    print("Available functions: sqrt, sin, cos, tan, log, log2,")
    print("log10, exp, pow, abs")
    expr = input("Enter math expression: ")
    try:
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
        print("Result: {}".format(result))
    except Exception as e:
        print("Invalid expression: {}".format(e))


def calculators_menu():
    """Display calculators menu."""
    while True:
        print("\nCalculators:")
        print("1. Bayes Calculator")
        print("2. Euclidean Distance Calculator")
        print("3. Manhattan Distance Calculator")
        print("4. Entropy Calculator")
        print("5. Gini Impurity Calculator")
        print("6. Information Gain Calculator")
        print("7. Math Expression Evaluator")
        print("0. Back to main menu")
        choice = input("\nSelect a calculator: ")
        if choice == "0":
            return
        elif choice == "1":
            bayes_calculator()
        elif choice == "2":
            euclidean_distance_calculator()
        elif choice == "3":
            manhattan_distance_calculator()
        elif choice == "4":
            entropy_calculator()
        elif choice == "5":
            gini_calculator()
        elif choice == "6":
            information_gain_calculator()
        elif choice == "7":
            execute_math()
        else:
            print("Invalid choice.")


def main():
    """Main menu."""
    while True:
        print("\n" + "=" * 50)
        print("Data Mining Calculators")
        print("=" * 50)
        print("1. Calculators")
        print("2. Quit")
        choice = input("\nChoose an option: ")
        if choice == "1":
            calculators_menu()
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


main()
