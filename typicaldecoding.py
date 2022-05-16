from typing import List

from scipy.special import softmax
from numpy import log, exp, inf, argmax
from numpy.random import choice

class TypicalDecoding:
    """
    Typical Sampling is an efficient, high-Quality 
    Decoding method founded in Information-Theory 
    by https://arxiv.org/pdf/2202.00666.pdf
    """

    @staticmethod 
    def normalise(logits:List[float]) -> List[float]:
        return softmax(logits)

    @staticmethod
    def log_normalise(logits:List[float]) -> List[float]:
        return log(TypicalDecoding.normalise(logits))

    @staticmethod 
    def shift_logits(log_normalised_scores:List[float]) -> List[float]:
        return abs(
            -log_normalised_scores 
            - TypicalDecoding.entropy(log_normalised_scores)
        )

    @staticmethod
    def sort_logits(logits:List[float],shifted_logits:List[float]) -> List[float]:
        return sorted(logits, key=lambda logit: shifted_logits[logits.index(logit)])
    
    @staticmethod 
    def cumulative_probability(sorted_logits:List[float]) -> List[float]:
        return TypicalDecoding.normalise(sorted_logits).cumsum()

    @staticmethod
    def entropy(log_normalised_scores:List[float]) -> float:
        entropies = -(log_normalised_scores * exp(log_normalised_scores))
        return sum(entropies)
    
    @staticmethod
    def typical_threshold(logits:List[float], shifted_logits:List[float], mass_threshold:float) -> float:
        keep = TypicalDecoding.cumulative_probability(TypicalDecoding.sort_logits(logits,shifted_logits)) < mass_threshold
        return sorted(shifted_logits)[sum(keep)]

    @staticmethod
    def filter_logits(logits:List[float], mass_threshold:float, filtered_value:float=-inf) -> List[float]:
        shifted_logits = TypicalDecoding.shift_logits(TypicalDecoding.log_normalise(logits))
        typical_threshold = TypicalDecoding.typical_threshold(logits,shifted_logits,mass_threshold)
        return list(map(
            lambda logit, shifted_logit: filtered_value if shifted_logit > typical_threshold else logit, 
            logits, shifted_logits
        ))

    @staticmethod
    def sample_index(probabilities:List[float], mass_threshold:float=.9) -> int:
        filtered_probabilities = TypicalDecoding.filter_logits(probabilities, mass_threshold, filtered_value=.0)
        total_probability = sum(filtered_probabilities)
        adjusted_probabilties = list(map(lambda probability:probability/total_probability,filtered_probabilities))
        value = choice(probabilities,p=adjusted_probabilties)
        return probabilities.index(value)


    @staticmethod 
    def best_index(probabilities:List[float], mass_threshold:float=.9) -> int:
        return argmax(TypicalDecoding.filter_logits(probabilities, mass_threshold))
