"""
A/B testing evaluator for comparative creativity assessment.

Provides structured interfaces for direct comparison between
AI outputs to determine relative creativity.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ...core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    OutputType,
)

logger = logging.getLogger(__name__)


class ABTestEvaluator(EvaluatorInterface):
    """
    A/B testing evaluator for comparative creativity assessment.
    
    This evaluator facilitates direct comparison between AI outputs
    to determine relative creativity through pairwise comparisons.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the A/B test evaluator.
        
        Args:
            config: Configuration options including comparison mode
        """
        self.config = config or {}
        self.comparison_mode = self.config.get("mode", "pairwise")  # pairwise, ranking, tournament
        self.randomize_order = self.config.get("randomize_order", True)
    
    @property
    def name(self) -> str:
        return f"ab_test_evaluator_{self.comparison_mode}"
    
    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.HUMAN
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.CODE, OutputType.STRUCTURED]
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        valid_keys = {"mode", "randomize_order", "evaluator_info", "criteria"}
        valid_modes = {"pairwise", "ranking", "tournament"}
        
        if "mode" in config and config["mode"] not in valid_modes:
            return False
        
        return all(key in valid_keys for key in config.keys())
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Evaluate creativity using A/B testing."""
        logger.info(f"Starting A/B test evaluation in {self.comparison_mode} mode")
        
        # Filter supported outputs
        supported_outputs = [
            output for output in request.outputs 
            if output.output_type in self.supported_output_types
        ]
        
        if len(supported_outputs) < 2:
            return self._create_insufficient_outputs_results(request)
        
        if self.comparison_mode == "pairwise":
            return await self._pairwise_comparison(supported_outputs, request)
        elif self.comparison_mode == "ranking":
            return await self._ranking_comparison(supported_outputs, request)
        elif self.comparison_mode == "tournament":
            return await self._tournament_comparison(supported_outputs, request)
        else:
            return self._create_unsupported_mode_results(request)
    
    async def _pairwise_comparison(
        self, 
        outputs: List[Any], 
        request: EvaluationRequest
    ) -> List[EvaluationResult]:
        """Conduct pairwise comparisons between outputs."""
        print("\n" + "="*60)
        print("🆚 A/B TESTING - PAIRWISE COMPARISON")
        print("="*60)
        print("You will compare pairs of AI outputs to determine which is more creative.")
        
        comparison_results = {}
        total_comparisons = len(outputs) * (len(outputs) - 1) // 2
        comparison_count = 0
        
        # Perform all pairwise comparisons
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                comparison_count += 1
                
                print(f"\n🔍 Comparison {comparison_count}/{total_comparisons}")
                print("-" * 40)
                
                winner_index = self._compare_pair(outputs[i], outputs[j], i, j)
                
                # Record results
                if i not in comparison_results:
                    comparison_results[i] = {"wins": 0, "losses": 0}
                if j not in comparison_results:
                    comparison_results[j] = {"wins": 0, "losses": 0}
                
                if winner_index == i:
                    comparison_results[i]["wins"] += 1
                    comparison_results[j]["losses"] += 1
                elif winner_index == j:
                    comparison_results[j]["wins"] += 1
                    comparison_results[i]["losses"] += 1
                # Tie case - no changes
        
        # Convert to evaluation results
        results = []
        for i, output in enumerate(outputs):
            if i in comparison_results:
                wins = comparison_results[i]["wins"]
                losses = comparison_results[i]["losses"]
                total_matches = wins + losses
                win_rate = wins / total_matches if total_matches > 0 else 0.5
                
                scores = {
                    "win_rate": win_rate,
                    "wins": wins,
                    "total_comparisons": total_matches,
                    "relative_creativity": win_rate,  # Use win rate as creativity score
                }
                
                explanations = {
                    "comparison_summary": f"Won {wins}/{total_matches} pairwise comparisons",
                    "performance": "High" if win_rate > 0.7 else "Medium" if win_rate > 0.3 else "Low"
                }
                
                results.append(EvaluationResult(
                    evaluator_name=self.name,
                    layer=self.layer,
                    scores=scores,
                    explanations=explanations,
                    metadata={
                        "comparison_mode": "pairwise",
                        "total_outputs": len(outputs),
                        "output_index": i
                    }
                ))
        
        return results
    
    async def _ranking_comparison(
        self, 
        outputs: List[Any], 
        request: EvaluationRequest
    ) -> List[EvaluationResult]:
        """Conduct ranking comparison of all outputs."""
        print("\n" + "="*60)
        print("📊 A/B TESTING - RANKING COMPARISON")
        print("="*60)
        print("Please rank all outputs from most creative (1) to least creative.")
        
        # Display all outputs
        for i, output in enumerate(outputs):
            print(f"\n--- Option {i+1} ---")
            self._display_content_brief(output)
        
        # Collect ranking
        ranking = self._collect_ranking(len(outputs))
        
        # Convert to evaluation results
        results = []
        for i, output in enumerate(outputs):
            rank = ranking[i]  # 1-based rank
            normalized_score = 1.0 - (rank - 1) / (len(outputs) - 1)  # Convert to 0-1 scale
            
            scores = {
                "rank": rank,
                "normalized_rank_score": normalized_score,
                "relative_creativity": normalized_score,
            }
            
            explanations = {
                "ranking_result": f"Ranked #{rank} out of {len(outputs)} options",
                "performance": "Excellent" if rank <= 2 else "Good" if rank <= len(outputs)//2 else "Needs improvement"
            }
            
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores=scores,
                explanations=explanations,
                metadata={
                    "comparison_mode": "ranking",
                    "total_outputs": len(outputs),
                    "output_index": i
                }
            ))
        
        return results
    
    async def _tournament_comparison(
        self, 
        outputs: List[Any], 
        request: EvaluationRequest
    ) -> List[EvaluationResult]:
        """Conduct tournament-style comparison."""
        print("\n" + "="*60)
        print("🏆 A/B TESTING - TOURNAMENT COMPARISON")
        print("="*60)
        print("Tournament-style elimination to find the most creative output.")
        
        # Simplified tournament: just find winner for now
        # In a full implementation, this would do bracket-style elimination
        current_outputs = list(enumerate(outputs))
        
        while len(current_outputs) > 1:
            next_round = []
            
            # Pair up outputs for this round
            for i in range(0, len(current_outputs), 2):
                if i + 1 < len(current_outputs):
                    # Compare pair
                    idx1, output1 = current_outputs[i]
                    idx2, output2 = current_outputs[i + 1]
                    
                    print(f"\n🥊 Tournament Round - Comparing Options {idx1+1} vs {idx2+1}")
                    winner_idx = self._compare_pair(output1, output2, idx1, idx2)
                    
                    if winner_idx == idx1:
                        next_round.append((idx1, output1))
                    else:
                        next_round.append((idx2, output2))
                else:
                    # Odd one out advances automatically
                    next_round.append(current_outputs[i])
            
            current_outputs = next_round
        
        # Winner found
        winner_idx, winner_output = current_outputs[0]
        
        # Create results with tournament placement
        results = []
        for i, output in enumerate(outputs):
            if i == winner_idx:
                scores = {"tournament_winner": 1.0, "relative_creativity": 1.0}
                explanations = {"result": "🏆 Tournament Winner!"}
            else:
                scores = {"tournament_winner": 0.0, "relative_creativity": 0.5}
                explanations = {"result": "Did not win tournament"}
            
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores=scores,
                explanations=explanations,
                metadata={
                    "comparison_mode": "tournament",
                    "total_outputs": len(outputs),
                    "output_index": i,
                    "winner_index": winner_idx
                }
            ))
        
        return results
    
    def _compare_pair(self, output1: Any, output2: Any, idx1: int, idx2: int) -> int:
        """Compare two outputs and return the index of the more creative one."""
        print(f"\n📝 Option {idx1+1}:")
        self._display_content_brief(output1)
        
        print(f"\n📝 Option {idx2+1}:")
        self._display_content_brief(output2)
        
        while True:
            choice = input(f"\nWhich is MORE CREATIVE? (1 for Option {idx1+1}, 2 for Option {idx2+1}, t for tie): ").strip().lower()
            
            if choice == "1":
                return idx1
            elif choice == "2":
                return idx2
            elif choice == "t":
                return -1  # Tie
            else:
                print("Please enter 1, 2, or 't' for tie")
    
    def _collect_ranking(self, num_outputs: int) -> List[int]:
        """Collect ranking from user input."""
        print(f"\nPlease enter rankings (1 = most creative, {num_outputs} = least creative):")
        
        rankings = [0] * num_outputs
        used_ranks = set()
        
        for i in range(num_outputs):
            while True:
                try:
                    rank = int(input(f"Rank for Option {i+1}: "))
                    if 1 <= rank <= num_outputs and rank not in used_ranks:
                        rankings[i] = rank
                        used_ranks.add(rank)
                        break
                    else:
                        print(f"Please enter a unique rank between 1 and {num_outputs}")
                except ValueError:
                    print("Please enter a valid number")
        
        return rankings
    
    def _display_content_brief(self, output: Any) -> None:
        """Display content in brief format for comparison."""
        content = str(output.content)
        
        # Truncate for easier comparison
        if len(content) > 200:
            content = content[:200] + "..."
        
        print(content)
        
        if hasattr(output, 'prompt') and output.prompt:
            prompt = str(output.prompt)
            if len(prompt) > 100:
                prompt = prompt[:100] + "..."
            print(f"  💭 Prompt: {prompt}")
    
    def _create_insufficient_outputs_results(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Create results when there are insufficient outputs for comparison."""
        results = []
        for output in request.outputs:
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores={},
                explanations={"error": "Need at least 2 outputs for A/B testing"},
                metadata={"comparison_mode": self.comparison_mode, "total_outputs": len(request.outputs)}
            ))
        return results
    
    def _create_unsupported_mode_results(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Create results for unsupported comparison modes."""
        results = []
        for output in request.outputs:
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores={},
                explanations={"error": f"Unsupported comparison mode: {self.comparison_mode}"},
                metadata={"comparison_mode": self.comparison_mode}
            ))
        return results