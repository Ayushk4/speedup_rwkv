"""Iteratively Prune a model based on the magnitude of weights."""
import collections # pylint: disable=syntax-error
import copy

from typing import Union, List # pylint: disable=syntax-error
import torch
from torch.nn.utils import prune as torch_prune # pylint: disable=wrong-import-position


class PruneRecipe:
    def __init__(self):
        # self.layer_patterns = [
        #     'module.blocks.*.att.key',
        #     'module.blocks.*.att.value',
        #     'module.blocks.*.att.receptance',
        #     'module.blocks.*.att.output',
        #     'module.blocks.*.ffn.key',
        #     'module.blocks.*.ffn.receptance',
        #     'module.blocks.*.ffn.value']
        self.layer_patterns = [
            'blocks.*.att.key',
            'blocks.*.att.value',
            'blocks.*.att.receptance',
            'blocks.*.att.output',
            'blocks.*.ffn.key',
            'blocks.*.ffn.receptance',
            'blocks.*.ffn.value']
        self.amounts, self.breakpoints = zip(*[
            (0.1, 100),
            (0.08, 800),
            (0.08, 3000),
            (0.07, 6000),
            (0.07, 10000),
            (0.06, 15000),
            (0.06, 21000),
            (0.05, 28000),
            (0.05, 38000),
            (0.04, 50000),
            (0.04, 65000),
            (0.03, 82000),
            (0.03, 100000),
        ])
        self.amounts = list(self.amounts)
        self.breakpoints = list(self.breakpoints)

def _get_attr_from_custom_pattern(layer, attr_pattern: Union[str, List[str]], idx=0):
    """Get the attribute of the layer that matches the regex.

    Only regex "*" pattern are supported.
    Here

    For example:
        module.blocks.*.att.key.weight:
        Here * is a wildcard that matches all the keys for one attr
        under the assumption that module.blocks is an iterable.
    """
    if isinstance(attr_pattern, str):
        attr_pattern = attr_pattern.split(".")
        idx = 0
    if len(attr_pattern) <= idx:
        assert idx == len(attr_pattern), "Invalid index {idx} for pattern {attr_pattern}"
        return [layer]
    if "*" in attr_pattern[idx]:
        assert attr_pattern[idx] == "*", "Only * is supported as a wildcard for entire layer."
        assert isinstance(layer, collections.abc.Iterable), "Layer is not iterable."
        return [param for iterate in layer
                for param in _get_attr_from_custom_pattern(iterate, attr_pattern, idx + 1)]
    return _get_attr_from_custom_pattern(
        getattr(layer, attr_pattern[idx]), attr_pattern, idx + 1)

def log_prune_statistics(parameters_to_prune):
    """Logs the prune statistics."""
    total_num_pruned = 0
    total_num_params = 0
    for param_layer, _ in parameters_to_prune:
        this_layer_pruned = float(torch.sum(param_layer.weight == 0)) # pylint: disable=no-member
        this_num_params = float(param_layer.weight.nelement())
        total_num_pruned += this_layer_pruned
        total_num_params += this_num_params

        sparsity_percent = round(100. * this_layer_pruned / this_num_params, 3)
        print(f"Sparsity in {param_layer}: {sparsity_percent}%")

    print(f"Global sparsity: {round(100. * total_num_pruned / total_num_params, 3)}%")

class Pruner:
    """Class to Prune a model based on the pruning recipe.

    prune_recipe = {"layer_patterns": [...],
                    "method": "L1Unstructured",
                    "amounts": [...],
                    "breakpoints": [...],
                    "scorer": [...],
                    "num_datapoints_score": 1000,
                   }
    Scorer has values from "magnitude", "firstOrderOptimizer", "secondOrder"
    """

    def __init__(self, iter_steps=0):
        prune_recipe = PruneRecipe()
        self.prune_recipe = prune_recipe
        self.layer_patterns = self.prune_recipe.layer_patterns
        self.breakpoints = self.prune_recipe.breakpoints
        self.amounts = self.prune_recipe.amounts

        assert len(self.amounts) == len(self.breakpoints), (
            "Amounts and breakpoints should be of same length")

        self.iter_steps = iter_steps

        self.times_pruned = 0
        while self.breakpoints[self.times_pruned] < self.iter_steps:
            self.times_pruned += 1
        self.model_initialized = False

    def init_prune(self, model):
        """Initializes the model for pruning.

        Detects if model already has zero weights, and uses it to generate masks.
        """
        print(type(model))
        parameters_to_prune = [(layer, 'weight') for layer_pattern in self.layer_patterns
                               for layer in _get_attr_from_custom_pattern(model, layer_pattern)]
        num_prune_params = [layer.weight.data.nelement() - torch.sum(layer.weight.data == 0) # pylint: disable=no-member
                            for layer, _ in parameters_to_prune]
        fraction_init_prune = 1.0 - sum(num_prune_params) / sum(
            [layer.weight.data.nelement() for layer, _ in parameters_to_prune]) # pylint: disable=no-member
        fraction_init_prune = min(max(0.0, fraction_init_prune), 1.0)
        if self.times_pruned != 0:
            assert abs(fraction_init_prune - self.breakpoints[self.times_pruned - 1]) < 0.01, (
                "Pruning recipe is not consistent with the model. "
                "The model is already pruned to a different level.")

        if fraction_init_prune > 0:
            torch_prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch_prune.L1Unstructured,
                amount=fraction_init_prune)
        self.model_initialized = True
        return model

    def update_iter_steps(self, iter_steps):
        """Updates the iteration steps."""
        self.iter_steps = iter_steps

    def is_it_time_to_prune(self):
        """Decides whether to prune the model or not and accordingly prunes."""
        if self.times_pruned >= len(self.breakpoints):
            return False
        if self.iter_steps < self.breakpoints[self.times_pruned]:
            return False
        assert self.model_initialized, "Model not initialized for pruning."
        return True

    def prune_model(self, model):
        """Prunes the model if called."""
        if not self.is_it_time_to_prune():
            return

        print(type(model))
        parameters_to_prune = [(layer, "weight") for layer_pattern in self.layer_patterns
                               for layer in _get_attr_from_custom_pattern(model, layer_pattern)]
        parameters_to_prune = tuple(parameters_to_prune)
        print([(type(x[0]), x[1]) for x in parameters_to_prune])
        print("Sparsity before pruning:")
        log_prune_statistics(parameters_to_prune)

        torch_prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch_prune.L1Unstructured,
            amount=self.amounts[self.times_pruned])

        print("Sparsity after pruning:")
        log_prune_statistics(parameters_to_prune)

        self.times_pruned += 1

    def remove_prune(self, model):
        """Updates the model weights based on the prune mask, zeroing them out."""
        for layer_pattern in self.layer_patterns:
            for layer in _get_attr_from_custom_pattern(model, layer_pattern):
                torch_prune.remove(layer, 'weight')
        return model
