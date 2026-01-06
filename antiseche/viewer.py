import os
import ure as re


def list_md_files(path):
    """Recursively list all .md files in the given path."""
    files = []
    try:
        items = os.listdir(path)
        for item in items:
            full_path = path + "/" + item if path != "/" else "/" + item
            try:
                stat = os.stat(full_path)
                if stat[0] & 0x4000:  # directory
                    files.extend(list_md_files(full_path))
                elif item.endswith(".md"):
                    files.append(full_path)
            except:
                pass
    except:
        pass
    return files


def strip_markdown(text):
    """Strip basic markdown syntax from text."""
    # Remove headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove bold
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    # Remove italic
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove lists
    text = re.sub(r"^[\s]*[-\*\+]\s*", "", text, flags=re.MULTILINE)
    return text


def display_text(text, max_lines=20):
    """Display text in chunks for limited screen."""
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        print(chunk)
        if i + max_lines < len(lines):
            input("Press Enter to continue...")


def view_courses(directory):
    """View all courses by listing and displaying .md files."""
    files = list_md_files(directory)
    if not files:
        print("No .md files found.")
        return
    print("Available courses:")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    choice = input("Enter number to view (or 'q' to quit): ")
    if choice.lower() == "q":
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            with open(files[idx], "r") as f:
                content = f.read()
            stripped = strip_markdown(content)
            display_text(stripped)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")


def search_courses(directory, query):
    """Search for query in all .md files."""
    files = list_md_files(directory)
    results = []
    for file in files:
        try:
            with open(file, "r") as f:
                content = f.read()
            if query.lower() in content.lower():
                results.append(file)
        except:
            pass
    if not results:
        print("No matches found.")
        return
    print("Matching files:")
    for i, file in enumerate(results):
        print(f"{i+1}. {file}")
    choice = input("Enter number to view (or 'q' to quit): ")
    if choice.lower() == "q":
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            with open(results[idx], "r") as f:
                content = f.read()
            stripped = strip_markdown(content)
            display_text(stripped)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")


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
        entropy = -sum(
            p * (p ** (1 / p) if p > 0 else 0) for p in probs
        )  # Wait, wrong formula
        # Correct entropy: -sum(p * log2(p) for p in probs if p > 0)
        import math

        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        print(f"Entropy: {entropy}")
    except ValueError:
        print("Invalid input. Please enter numbers.")


def execute_math():
    """Simple math expression evaluator."""
    print("Math Executor")
    expr = input("Enter math expression: ")
    try:
        result = eval(expr)
        print(f"Result: {result}")
    except:
        print("Invalid expression.")


def main():
    """Main menu."""
    directory = "antiseche"  # Assuming script is run from parent directory
    while True:
        print("\nData Mining Course Viewer and Calculator")
        print("1. View Courses")
        print("2. Search Courses")
        print("3. Bayes Calculator")
        print("4. Euclidean Distance Calculator")
        print("5. Entropy Calculator")
        print("6. Execute Math")
        print("7. Quit")
        choice = input("Choose an option: ")
        if choice == "1":
            view_courses(directory)
        elif choice == "2":
            query = input("Enter search query: ")
            search_courses(directory, query)
        elif choice == "3":
            bayes_calculator()
        elif choice == "4":
            euclidean_distance_calculator()
        elif choice == "5":
            entropy_calculator()
        elif choice == "6":
            execute_math()
        elif choice == "7":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
