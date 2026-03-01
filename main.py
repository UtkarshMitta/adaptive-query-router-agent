from CSVWrapper import log_metrics_to_csv
import time
from inference.router import MistralRouter

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
        print(f"Recorded turn {self.turn_count}: {model_used} ({tokens_used} tokens, cost: ${cost:.6f})")

def main():
    print("Initializing Adaptive Agent and Mistral Router...")
    tracker = AgentMetricsTracker()
    
    # Initialize our fine-tuned 7B router
    router = MistralRouter("mistral-hackaton-2026/mistral-query-router")
    
    # Rough simulated query cost per token 
    cost_map = {
        1: 0.0001 / 1000, # Ministral 3B
        2: 0.0002 / 1000, # Mistral Small
        3: 0.002 / 1000,  # Mistral Medium
        4: 0.003 / 1000   # Mistral Large
    }

    print("\n--- Interactive Query Router Started ---")
    print("Type your queries below. Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            q = input("\nQuery: ").strip()
            if q.lower() in ('quit', 'exit', 'q'):
                break
            if not q:
                continue
            
            # 1. Route the query using our trained model
            result = router.route(q)
            
            # 2. Extract metrics
            tier = result["model_tier"]
            model_name = result["model_name"]
            confidence = result["confidence"]
            complexity = result["tier_label"]
            
            # Simulate token counts and cost for logging
            in_tokens = len(q.split()) * 2 
            cost = cost_map.get(tier, cost_map[2]) * in_tokens
            
            print(f"Routed to: Tier {tier} ({model_name}) | Confidence: {confidence:.2f}")
            
            # 3. Log the turn metrics to the CSV dataset
            tracker.record_turn(model_name, in_tokens, complexity, cost)
            
        except KeyboardInterrupt:
            break
            
    print(f"\nSession ended. All metrics pushed to {tracker.metrics_file} successfully.")

if __name__ == "__main__":
    main()
