import pstats
import sys

# Increase recursion depth limit if needed for deep call stacks
# sys.setrecursionlimit(2000) 

profile_file = "train_profile.prof"
output_limit = 20 # Number of lines to print

try:
    # Create a Stats object
    stats = pstats.Stats(profile_file)

    # Sort the statistics by cumulative time and print the top entries
    print(f"--- Top {output_limit} functions by cumulative time (cumtime) ---")
    stats.sort_stats('cumulative').print_stats(output_limit)

    # Optionally, sort by total time spent *in* the function (tottime)
    # print(f"\n--- Top {output_limit} functions by total time (tottime) ---")
    # stats.sort_stats('tottime').print_stats(output_limit)

except FileNotFoundError:
    print(f"Error: Profile file '{profile_file}' not found.")
except Exception as e:
    print(f"An error occurred while analyzing the profile: {e}")