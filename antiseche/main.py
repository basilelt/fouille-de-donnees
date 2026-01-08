#!/usr/bin/env python3
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
import content_cours

# Topic modules registry
TOPIC_MODULES = {
    "cours": content_cours,
}

TOPICS = {
    "1": ("cours", "Cours"),
}


def get_all_topics():
    """Return list of all available topics."""
    return sorted(TOPIC_MODULES.keys())


def get_topic_module(topic):
    """Get the module for a specific topic."""
    return TOPIC_MODULES.get(topic)


def view_topic_menu():
    """Display topic selection menu."""
    print("\nAvailable Topics:")
    for key, (topic_key, display_name) in sorted(TOPICS.items(), key=lambda x: int(x[0])):
        print(f"{key}. {display_name}")
    print("0. Back to main menu")
    
    choice = input("\nSelect a topic: ")
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
        print(f"Topic '{topic_key}' not found.")
        return
    
    files = module.get_files()
    if not files:
        print(f"No files found in {display_name}.")
        return
    
    while True:
        print(f"\n{display_name} - Available Files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        print("0. Back to topics")
        
        choice = input("\nSelect a file to view: ")
        if choice == "0":
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                content = module.get_content(files[idx])
                stripped = strip_markdown(content)
                print(f"\n--- {files[idx]} ---\n")
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
    
    print(f"\nFound {len(results)} matches:")
    for i, (topic, file_key, _) in enumerate(results, 1):
        print(f"{i}. [{topic}] {file_key}")
    
    choice = input("\nEnter number to view (or 'q' to quit): ")
    if choice.lower() == "q":
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            topic, file_key, module = results[idx]
            content = module.get_content(file_key)
            stripped = strip_markdown(content)
            print(f"\n--- {file_key} ---\n")
            display_text(stripped)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")


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
        print("Data Mining Course Viewer and Calculator")
        print("=" * 50)
        print("1. Browse Topics")
        print("2. Search All Content")
        print("3. Calculators")
        print("4. Quit")
        
        choice = input("\nChoose an option: ")
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
