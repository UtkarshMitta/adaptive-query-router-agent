from CSVWrapper import log_metrics_to_csv

class AgentMetricsTracker:
    def __init__(self, metrics_file="agent_metrics.csv"):
        self.metrics_file = metrics_file
        self.turn_count = 0
        self.cumulative_cost = 0.0

    def record_turn(self, model_used, tokens_used, input_complexity, cost):
        """
        Record a single turn's metrics and save it to the CSV.
        """
        self.turn_count += 1
        self.cumulative_cost += cost
        
        turn_metrics = {
            "turn": self.turn_count,
            "model_used": model_used,
            "tokens_used": tokens_used,
            "input_complexity": input_complexity,
            "cumulative_cost": self.cumulative_cost
        }
        
        log_metrics_to_csv(self.metrics_file, turn_metrics)
        print(f"Recorded turn {self.turn_count}: {model_used} ({tokens_used} tokens)")

def main():
    print("Initializing Agent...")
    tracker = AgentMetricsTracker()
    
    # Simulate a few turns in the agent
    print("\n--- Simulating Conversation Turns ---")
    tracker.record_turn("small_mistral", 120, "low", 0.00012)
    tracker.record_turn("medium_mistral", 350, "medium", 0.0005)
    tracker.record_turn("large_mistral", 850, "high", 0.0015)
    
    print(f"\nAll metrics pushed to {tracker.metrics_file} successfully.")

if __name__ == "__main__":
    main()
