#!/usr/bin/env python3
"""
Generator script that reads all markdown files from the antiseche directory
and generates multiple Python modules with embedded content.

Generated files (all in the same directory for Numworks compatibility):
- utilities.py (markdown stripping and display functions)
- calculators.py (all calculator functions)
- content_<topic>.py (one per topic directory)
- main.py (user interaction menu)

Note: Numworks calculators don't support subdirectories, so all files
are generated in the same flat directory.
"""

import os
from pathlib import Path
from collections import defaultdict

MAX_FILE_SIZE = 43000  # Maximum characters per file


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


def escape_content(content):
    """Escape content for embedding in Python string."""
    # Escape backslashes first, then quotes
    content = content.replace("\\", "\\\\")
    content = content.replace('"""', '\\"\\"\\"')
    return content


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


def generate_utilities_module():
    """Generate the utilities.py module content."""
    return '''# Auto-generated utilities module
# Contains markdown processing and display functions

import re


def strip_markdown(text):
    """Strip basic markdown syntax from text."""
    # Remove headers
    text = re.sub(r"^#+\\s*", "", text, flags=re.MULTILINE)
    # Remove links
    text = re.sub(r"\\[([^\\]]+)\\]\\([^\\)]+\\)", r"\\1", text)
    # Remove bold
    text = re.sub(r"\\*\\*([^\\*]+)\\*\\*", r"\\1", text)
    # Remove italic
    text = re.sub(r"\\*([^\\*]+)\\*", r"\\1", text)
    # Remove code blocks
    text = re.sub(r"```[\\s\\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\\1", text)
    # Remove lists
    text = re.sub(r"^[\\s]*[-\\*\\+]\\s*", "", text, flags=re.MULTILINE)
    return text


def display_text(text, max_lines=20):
    """Display text in chunks for limited screen."""
    lines = text.split("\\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\\n".join(lines[i : i + max_lines])
        print(chunk)
        if i + max_lines < len(lines):
            input("Press Enter to continue...")
'''


def generate_calculators_module():
    """Generate the calculators.py module content."""
    return '''# Auto-generated calculators module
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
'''


def generate_content_module(topic, files):
    """Generate a content module for a specific topic."""
    module_name = sanitize_module_name(topic)

    content_entries = []
    for rel_path, full_path in files:
        content = read_file_content(full_path)
        escaped_content = escape_content(content)
        key = rel_path.replace("\\", "/")
        content_entries.append(f'    "{key}": """{escaped_content}"""')

    embedded_dict = "{\n" + ",\n".join(content_entries) + "\n}"

    # Get display name for topic
    display_name = topic.replace("-", " ").replace("_", " ").title()

    return f'''# Auto-generated content module for {display_name}
# Contains embedded markdown content for this topic

TOPIC_NAME = "{display_name}"
TOPIC_KEY = "{topic}"

CONTENT = {embedded_dict}


def get_files():
    """Return list of files in this topic."""
    return sorted(CONTENT.keys())


def get_content(file_key):
    """Get content for a specific file."""
    return CONTENT.get(file_key, "")


def search(query):
    """Search for query in topic content."""
    results = []
    for file_key, content in CONTENT.items():
        if query.lower() in content.lower():
            results.append(file_key)
    return results
'''


def generate_main_module(topics):
    """Generate main.py with the interactive menu (flat imports for Numworks)."""
    # Build topic imports - flat structure, no package
    topic_imports = []
    topic_module_mappings = []
    for topic in sorted(topics.keys()):
        module_name = sanitize_module_name(topic)
        topic_imports.append(f"import content_{module_name}")
        topic_module_mappings.append(f'    "{topic}": content_{module_name},')

    imports_str = "\n".join(topic_imports)
    mappings_str = "\n".join(topic_module_mappings)

    # Build topic menu entries
    topic_entries = []
    for i, topic in enumerate(sorted(topics.keys()), 1):
        display_name = topic.replace("-", " ").replace("_", " ").title()
        topic_entries.append(f'    "{i}": ("{topic}", "{display_name}"),')
    topics_str = "\n".join(topic_entries)

    return f'''#!/usr/bin/env python3
# Auto-generated main module for Numworks calculator
# Provides interactive menu for course viewer and calculators
# All imports are flat (no subdirectories) for Numworks compatibility

from utilities import strip_markdown, display_text
from calculators import (
    bayes_calculator,
    euclidean_distance_calculator,
    manhattan_distance_calculator,
    entropy_calculator,
    gini_calculator,
    information_gain_calculator,
    execute_math,
)
{imports_str}

# Topic modules registry
TOPIC_MODULES = {{
{mappings_str}
}}

TOPICS = {{
{topics_str}
}}


def get_all_topics():
    """Return list of all available topics."""
    return sorted(TOPIC_MODULES.keys())


def get_topic_module(topic):
    """Get the module for a specific topic."""
    return TOPIC_MODULES.get(topic)


def view_topic_menu():
    """Display topic selection menu."""
    print("\\nAvailable Topics:")
    for key, (topic_key, display_name) in sorted(TOPICS.items(), key=lambda x: int(x[0])):
        print(f"{{key}}. {{display_name}}")
    print("0. Back to main menu")
    
    choice = input("\\nSelect a topic: ")
    if choice == "0":
        return
    
    if choice in TOPICS:
        topic_key, display_name = TOPICS[choice]
        view_topic_files(topic_key, display_name)
    else:
        print("Invalid choice.")


def view_topic_files(topic_key, display_name):
    """View files within a specific topic."""
    module = get_topic_module(topic_key)
    if not module:
        print(f"Topic '{{topic_key}}' not found.")
        return
    
    files = module.get_files()
    if not files:
        print(f"No files found in {{display_name}}.")
        return
    
    while True:
        print(f"\\n{{display_name}} - Available Files:")
        for i, file in enumerate(files, 1):
            print(f"{{i}}. {{file}}")
        print("0. Back to topics")
        
        choice = input("\\nSelect a file to view: ")
        if choice == "0":
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                content = module.get_content(files[idx])
                stripped = strip_markdown(content)
                print(f"\\n--- {{files[idx]}} ---\\n")
                display_text(stripped)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")


def search_all_topics():
    """Search across all topics."""
    query = input("Enter search query: ")
    if not query:
        return
    
    results = []
    for topic_key in get_all_topics():
        module = get_topic_module(topic_key)
        if module:
            for file_key in module.search(query):
                results.append((topic_key, file_key, module))
    
    if not results:
        print("No matches found.")
        return
    
    print(f"\\nFound {{len(results)}} matches:")
    for i, (topic, file_key, _) in enumerate(results, 1):
        print(f"{{i}}. [{{topic}}] {{file_key}}")
    
    choice = input("\\nEnter number to view (or 'q' to quit): ")
    if choice.lower() == "q":
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            topic, file_key, module = results[idx]
            content = module.get_content(file_key)
            stripped = strip_markdown(content)
            print(f"\\n--- {{file_key}} ---\\n")
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
            search_all_topics()
        elif choice == "3":
            calculators_menu()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
'''


def check_file_size(content, filename, max_size=MAX_FILE_SIZE):
    """Check if file content exceeds maximum size and warn if so."""
    size = len(content)
    if size > max_size:
        print(f"WARNING: {filename} is {size} characters (exceeds {max_size})")
        return False
    return True


def write_file(path, content, filename):
    """Write file and check size."""
    check_file_size(content, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: {filename} ({len(content)} chars)")


def generate_all(base_dir, output_dir):
    """Generate all modules in flat structure (no subdirectories)."""
    # Find and group markdown files
    md_files = find_md_files(base_dir)
    topics = group_files_by_topic(md_files)

    print(f"Found {len(md_files)} markdown files in {len(topics)} topics:")
    for topic, files in sorted(topics.items()):
        print(f"  - {topic}: {len(files)} files")

    print(f"\nGenerating modules in {output_dir} (flat structure for Numworks):")

    # Generate utilities module
    utilities_content = generate_utilities_module()
    write_file(output_dir / "utilities.py", utilities_content, "utilities.py")

    # Generate calculators module
    calculators_content = generate_calculators_module()
    write_file(output_dir / "calculators.py", calculators_content, "calculators.py")

    # Generate content modules for each topic
    for topic, files in sorted(topics.items()):
        module_name = f"content_{sanitize_module_name(topic)}.py"
        content = generate_content_module(topic, files)
        write_file(output_dir / module_name, content, module_name)

    # Generate main.py in same directory
    main_content = generate_main_module(topics)
    write_file(output_dir / "main.py", main_content, "main.py")

    print(f"\nGeneration complete!")
    print(f"All files are in: {output_dir}")
    print(f"Upload all .py files to your Numworks calculator, then run main.py")


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Output directory is same as script directory (flat structure)
    output_dir = script_dir

    # Generate all modules
    generate_all(script_dir, output_dir)


if __name__ == "__main__":
    main()
