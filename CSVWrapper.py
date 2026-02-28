import csv
import os

def log_metrics_to_csv(filepath, metrics_dict):
    """
    Appends a dictionary of metrics to a CSV file.
    Creates the file and writes headers if it doesn't exist.
    """
    # Check if the file already exists so we know whether to write headers
    file_exists = os.path.isfile(filepath)
    
    # Extract headers from the dictionary keys
    headers = list(metrics_dict.keys())
    
    # Open the file in 'append' mode ('a')
    with open(filepath, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write the header row only if the file is new
        if not file_exists:
            writer.writeheader()
            
        # Write the data row
        writer.writerow(metrics_dict)

# --- Example Usage ---
if __name__ == "__main__":
    metrics_file = "agent_metrics.csv"
    
    # Example metrics from a router turn
    turn_metrics = {
        "turn": 1,
        "model_used": "small_mistral",
        "tokens_used": 150,
        "input_complexity": "low",
        "cumulative_cost": 0.00015
    }
    
    turn_metrics = {
        "turn2": 2,
        "model_used2": "small_mistral",
        "tokens_used2": 150,
        "input_complexity2": "low",
        "cumulative_cost2": 0.00015
    }

    log_metrics_to_csv(metrics_file, turn_metrics)
    print(f"Metrics saved to {metrics_file}")