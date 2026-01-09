import math


def display_text(text, max_lines=20):
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        print(chunk)
        if i + max_lines < len(lines):
            input("Enter...")


def bayes_calculator():
    print("Bayes")
    print("P(A|B)=(P(A)*P(B|A))/P(B)")
    try:
        prior = float(input("P(A): "))
        likelihood = float(input("P(B|A): "))
        evidence = float(input("P(B): "))
        posterior = (prior * likelihood) / evidence
        print("P(A|B):{}".format(posterior))
    except ValueError:
        print("Invalid num.")
    except ZeroDivisionError:
        print("P(B)!=0")


def euclidean_distance_calculator():
    print("Euclid Dist")
    print("sqrt(sum((x_i-y_i)^2))")
    try:
        n = int(input("Dims: "))
        point1 = []
        point2 = []
        for i in range(n):
            p1 = float(input("P1[{}]: ".format(i + 1)))
            p2 = float(input("P2[{}]: ".format(i + 1)))
            point1.append(p1)
            point2.append(p2)
        distance = sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
        print("Dist:{}".format(distance))
    except ValueError:
        print("Invalid num.")


def manhattan_distance_calculator():
    print("Manhattan Dist")
    print("sum(|x_i-y_i|)")
    try:
        n = int(input("Dims: "))
        point1 = []
        point2 = []
        for i in range(n):
            p1 = float(input("P1[{}]: ".format(i + 1)))
            p2 = float(input("P2[{}]: ".format(i + 1)))
            point1.append(p1)
            point2.append(p2)
        distance = sum(abs(a - b) for a, b in zip(point1, point2))
        print("Dist:{}".format(distance))
    except ValueError:
        print("Invalid num.")


def entropy_calculator():
    print("Entropy")
    print("-sum(p_i*log2(p_i))")
    print("Enter probs. Enter to finish.")
    try:
        probs = []
        while True:
            prob = input("Prob: ")
            if not prob:
                break
            probs.append(float(prob))
        if not probs:
            print("No probs.")
            return
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        print("Entropy:{}".format(entropy))
    except ValueError:
        print("Invalid num.")


def gini_calculator():
    print("Gini")
    print("1-sum(p_i^2)")
    print("Enter probs. Enter to finish.")
    try:
        probs = []
        while True:
            prob = input("Prob: ")
            if not prob:
                break
            probs.append(float(prob))
        if not probs:
            print("No probs.")
            return
        gini = 1 - sum(p**2 for p in probs)
        print("Gini:{}".format(gini))
    except ValueError:
        print("Invalid num.")


def information_gain_calculator():
    print("Info Gain")
    print("ParentEnt - weighted child ent")
    try:
        parent_entropy = float(input("Parent ent: "))
        n_children = int(input("Children: "))
        total_samples = int(input("Total smp: "))
        weighted_child_entropy = 0
        for i in range(n_children):
            samples = int(input("Smp{}: ".format(i + 1)))
            entropy = float(input("Ent{}: ".format(i + 1)))
            weighted_child_entropy += (samples / total_samples) * entropy
        info_gain = parent_entropy - weighted_child_entropy
        print("Gain:{}".format(info_gain))
    except ValueError:
        print("Invalid num.")
    except ZeroDivisionError:
        print("Total!=0")


def execute_math():
    print("Math")
    print("Funcs: sqrt,sin,cos,tan")
    print("log,log2,log10,exp,pow,abs")
    print("Consts: pi,e")
    expr = input("Expr: ")
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
        print("Res:{}".format(result))
    except Exception as e:
        print("Error:{}".format(e))


def main():
    print("Calc. 0=exit")
    while True:
        print("\nCalc:")
        print("1.Bayes")
        print("2.Euclid")
        print("3.Manhattan")
        print("4.Entropy")
        print("5.Gini")
        print("6.Info Gain")
        print("7.Math")
        print("0.Exit")
        choice = input("Sel: ")
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
            print("Invalid.")


main()
