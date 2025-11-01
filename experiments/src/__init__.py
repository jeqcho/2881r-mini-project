"""
SNIP Set Difference Experiments - Source Modules

This package contains modules for running SNIP set difference experiments:
- snip_calculator: SNIP score dumping
- em_evaluator: Emergent misalignment evaluation
- collect_results: Results collection and parsing
- plot_results: Plot generation
"""

__version__ = "1.0.0"

from .snip_calculator import dump_snip_scores, check_snip_scores_exist
from .collect_results import collect_results_from_pairs, collect_diagonal_results, collect_custom_results
from .plot_results import create_all_plots, create_scatter_plot, create_heatmap
from .em_evaluator import evaluate_em_score, LlamaModelInterface, EM_LIBRARY_AVAILABLE

__all__ = [
    "dump_snip_scores",
    "check_snip_scores_exist",
    "collect_results_from_pairs",
    "collect_diagonal_results",
    "collect_custom_results",
    "create_all_plots",
    "create_scatter_plot",
    "create_heatmap",
    "evaluate_em_score",
    "LlamaModelInterface",
    "EM_LIBRARY_AVAILABLE",
]

