#!/usr/bin/env python3
"""
Generator script that reads all markdown files from the antiseche directory
and generates a single Python file with embedded content for Numworks calculator.

Generated file:
- main.py (all functionality in one file for Numworks compatibility)

Note: Numworks calculators don't support loading from other scripts, so everything
must fit in one file.
"""

import os
from pathlib import Path
from collections import defaultdict

MAX_FILE_SIZE = 43000  # Maximum characters per file
MAX_LINE_LENGTH = 80  # Maximum characters per line for Numworks


def find_md_files(base_dir):
    """Find all .md files recursively in base_dir."""
    md_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                # Get relative path from base_dir
                rel_path = os.path.relpath(full_path, base_dir)
                md_files.append((rel_path, full_path))
    return sorted(md_files)


def read_file_content(filepath):
    """Read file content with UTF-8 encoding."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def break_long_lines(text, max_length=MAX_LINE_LENGTH):
    """Break lines longer than max_length."""
    lines = text.split("\n")
    broken_lines = []
    for line in lines:
        if len(line) <= max_length:
            broken_lines.append(line)
        else:
            # Break at spaces if possible
            words = line.split(" ")
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_length:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    if current_line:
                        broken_lines.append(current_line)
                    current_line = word
            if current_line:
                broken_lines.append(current_line)
    return "\n".join(broken_lines)


def escape_line(line):
    """Escape a line for embedding in Python string list."""
    # Escape backslashes first, then quotes
    line = line.replace("\\", "\\\\")
    line = line.replace('"', '\\"')
    return line


def get_topic_from_path(rel_path):
    """Extract topic name from relative path."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) > 1:
        # It's in a subdirectory, use the directory name
        return parts[0]
    else:
        # It's a root file like introduction.md
        return "introduction"


def sanitize_module_name(name):
    """Convert topic name to valid Python module name."""
    return name.replace("-", "_").replace(" ", "_").lower()


def group_files_by_topic(md_files):
    """Group markdown files by their topic directory."""
    topics = defaultdict(list)
    for rel_path, full_path in md_files:
        topic = get_topic_from_path(rel_path)
        topics[topic].append((rel_path, full_path))
    return topics


def generate_single_file(topics):
    """Generate a single file containing all functionality for Numworks."""
    # Build content dict
    content_entries = []
    topic_display_names = {}
    for topic in sorted(topics.keys()):
        display_name = topic.replace("-", " ").replace("_", " ").title()
        topic_display_names[topic] = display_name
        files_content = []
        for rel_path, full_path in topics[topic]:
            content = read_file_content(full_path)
            lines = content.split("\n")
            escaped_lines = [f'            "{escape_line(line)}"' for line in lines]
            key = rel_path.replace("\\", "/")
            files_content.append(
                f'        "{key}": [\n' + ",\n".join(escaped_lines) + "\n        ]"
            )
        content_entries.append(
            f'    "{topic}": {{\n' + ",\n".join(files_content) + "\n    }"
        )

    content_dict_str = "{\n" + ",\n".join(content_entries) + "\n}"

    display_names_str = (
        "{\n"
        + ",\n".join(
            f'    "{topic}": "{name}"' for topic, name in topic_display_names.items()
        )
        + "\n}"
    )

    template = '''#!/usr/bin/env python3
# Auto-generated single file for Numworks calculator
# All functionality in one file for Numworks compatibility

import math

# Utilities functions

def strip_markdown(text):
    """Strip basic markdown syntax from text."""
    lines = text.split('\\n')
    stripped_lines = []
    in_code_block = False
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        # Remove headers
        if line.strip().startswith('#'):
            i = 0
            while i < len(line) and line[i] in '# \\t':
                i += 1
            line = line[i:]
        # Remove inline code
        new_line = ''
        i = 0
        while i < len(line):
            if line[i] == '`':
                j = i + 1
                while j < len(line) and line[j] != '`':
                    j += 1
                if j < len(line):
                    i = j + 1
                else:
                    new_line += line[i]
                    i += 1
            else:
                new_line += line[i]
                i += 1
        line = new_line
        # Remove bold and italic markers
        line = line.replace('**', '').replace('*', '')
        # Remove links [text](url) -> text
        while '[' in line and ']' in line:
            start = line.find('[')
            end = line.find(']', start)
            if end != -1 and end + 1 < len(line) and line[end + 1] == '(':
                close = line.find(')', end + 1)
                if close != -1:
                    text = line[start + 1:end]
                    line = line[:start] + text + line[close + 1:]
                else:
                    break
            else:
                break
        # Remove list markers
        if line.strip() and len(line.lstrip()) > 1 and line.lstrip()[0] in '-*+' and line.lstrip()[1] in ' \\t':
            line = line.lstrip()[1:].lstrip()
        stripped_lines.append(line)
    return '\\n'.join(stripped_lines)

def display_text(text, max_lines=20):
    """Display text in chunks for limited screen."""
    lines = text.split("\\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\\n".join(lines[i : i + max_lines])
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
            p1 = float(input("Enter coordinate {} for point 1: ".format(i+1)))
            p2 = float(input("Enter coordinate {} for point 2: ".format(i+1)))
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
            p1 = float(input("Enter coordinate {} for point 1: ".format(i+1)))
            p2 = float(input("Enter coordinate {} for point 2: ".format(i+1)))
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
        gini = 1 - sum(p ** 2 for p in probs)
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
            samples = int(input("Enter samples in child {}: ".format(i+1)))
            entropy = float(input("Enter entropy of child {}: ".format(i+1)))
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

# Content data

TOPIC_CONTENTS = content_dict_str

TOPIC_DISPLAY_NAMES = display_names_str

# Content functions

def get_all_topics():
    """Return list of all available topics."""
    return sorted(TOPIC_CONTENTS.keys())

def get_topic_content(topic):
    """Get content dict for a topic."""
    return TOPIC_CONTENTS.get(topic, {{}})

def get_files(topic):
    """Return list of files in a topic."""
    return sorted(get_topic_content(topic).keys())

def get_content(topic, file_key):
    """Get content for a specific file in a topic."""
    lines = get_topic_content(topic).get(file_key, [])
    return "\\n".join(lines)

def search_topic(topic, query):
    """Search for query in a topic."""
    results = []
    for file_key, content in get_topic_content(topic).items():
        if query.lower() in "\\n".join(content).lower():
            results.append(file_key)
    return results

def search_all_topics(query):
    """Search across all topics."""
    results = []
    for topic in get_all_topics():
        for file_key in search_topic(topic, query):
            results.append((topic, file_key))
    return results

# Menu functions

def view_topic_menu():
    """Display topic selection menu."""
    print("\\nAvailable Topics:")
    topics = get_all_topics()
    for i, topic in enumerate(topics, 1):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        print("{}. {}".format(i, display_name))
    print("0. Back to main menu")
    choice = input("\\nSelect a topic: ")
    if choice == "0":
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(topics):
            topic_key = topics[idx]
            display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
            view_topic_files(topic_key, display_name)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")

def view_topic_files(topic_key, display_name):
    """View files within a specific topic."""
    files = get_files(topic_key)
    if not files:
        print("No files found in {}.".format(display_name))
        return
    while True:
        print("\\n{} - Available Files:".format(display_name))
        for i, file in enumerate(files, 1):
            print("{}. {}".format(i, file))
        print("0. Back to topics")
        choice = input("\\nSelect a file to view: ")
        if choice == "0":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                content = get_content(topic_key, files[idx])
                stripped = strip_markdown(content)
                print("\\n--- {} ---\\n".format(files[idx]))
                display_text(stripped)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")

def search_all_topics_menu():
    """Search across all topics."""
    query = input("Enter search query: ")
    if not query:
        return
    results = search_all_topics(query)
    if not results:
        print("No matches found.")
        return
    print("\\nFound {} matches:".format(len(results)))
    for i, (topic, file_key) in enumerate(results, 1):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        print("{}. [{}] {}".format(i, display_name, file_key))
    choice = input("\\nEnter number to view (or 'q' to quit): ")
    if choice.lower() == "q":
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            topic, file_key = results[idx]
            content = get_content(topic, file_key)
            stripped = strip_markdown(content)
            print("\\n--- {} ---\\n".format(file_key))
            display_text(stripped)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")

def calculators_menu():
    """Display calculators menu."""
    while True:
        print("\\nCalculators:")
        print("1. Bayes Calculator")
        print("2. Euclidean Distance Calculator")
        print("3. Manhattan Distance Calculator")
        print("4. Entropy Calculator")
        print("5. Gini Impurity Calculator")
        print("6. Information Gain Calculator")
        print("7. Math Expression Evaluator")
        print("0. Back to main menu")
        choice = input("\\nSelect a calculator: ")
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
        print("\\n" + "=" * 50)
        print("Data Mining Course Viewer and Calculator")
        print("=" * 50)
        print("1. Browse Topics")
        print("2. Search All Content")
        print("3. Calculators")
        print("4. Quit")
        choice = input("\\nChoose an option: ")
        if choice == "1":
            view_topic_menu()
        elif choice == "2":
            search_all_topics_menu()
        elif choice == "3":
            calculators_menu()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

main()
'''
    return template.replace("content_dict_str", content_dict_str).replace(
        "display_names_str", display_names_str
    )


def check_file_constraints(
    content, filename, max_size=MAX_FILE_SIZE, max_line_length=MAX_LINE_LENGTH
):
    """Check file size and line lengths."""
    size = len(content)
    if size > max_size:
        print(f"WARNING: {filename} is {size} characters (exceeds {max_size})")

    lines = content.split("\n")
    max_len = max(len(line) for line in lines) if lines else 0
    if max_len > max_line_length:
        print(
            f"WARNING: {filename} has line of {max_len} characters (exceeds {max_line_length})"
        )

    return size <= max_size


def write_file(path, content, filename):
    """Write file and check constraints."""
    check_file_constraints(content, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: {filename} ({len(content)} chars)")


def generate_all(base_dir, output_dir):
    """Generate single file for Numworks."""
    # Find and group markdown files
    md_files = find_md_files(base_dir)
    topics = group_files_by_topic(md_files)

    print(f"Found {len(md_files)} markdown files in {len(topics)} topics:")
    for topic, files in sorted(topics.items()):
        print(f"  - {topic}: {len(files)} files")

    print(f"\nGenerating single file in {output_dir} for Numworks:")

    # Generate single file
    single_content = generate_single_file(topics)
    write_file(output_dir / "main.py", single_content, "main.py")

    print(f"\nGeneration complete!")
    print(f"Single file: {output_dir}/main.py")
    print(f"Upload main.py to your Numworks calculator and run it")


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Output directory is same as script directory (flat structure)
    output_dir = script_dir

    # Generate all modules
    generate_all(script_dir, output_dir)


if __name__ == "__main__":
    main()
