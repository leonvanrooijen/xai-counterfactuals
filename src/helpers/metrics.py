from pandas import DataFrame

class CounterfactualAnalysis:
    
    """
    
    CounterfactualAnalysis evaluates the quality of a counterfactual explanation, given the factual.
    
    The class computes the following metrics:
    - Validity
    - Categorical Proximity
    - Continuous Proximity
    - Sparsity
    - Categorical Diversity
    - Continuous Diversity
    - Continuous Counterfactuals Diversity
    
    """
    
    def __init__(self, model: str, factual: DataFrame, counterfactual: DataFrame, rounding: int = None):
        
        #Check if the input DataFrames are valid
        if len(factual) != 1:
            raise ValueError("Factual DataFrame should have only one instance")
        
        if len(counterfactual) == 0:
            raise ValueError("Counterfactual DataFrame should have at least one instance")
        
        self.rounding = rounding
        self.factual = factual
        self.counterfactual = counterfactual
        
    def validity(self):  
        ...
    
    def cat_prox(self):
        ...
        
    def cont_prox(self):
        ...
        
    def sparsity(self):
        ...
        
    def cat_diver(self):
        ...
        
    def cont_diver(self):
        ...
        
    def cont_count_divers(self):
        ...
    
    def __round(self, value):
        if self.rounding is not None:
            return round(value, self.rounding)
        return value
        
    
    def evaluate(self):
    
        #Compute all metrics
        
        return DataFrame(
            [[
              self.model,
              self.validity(),
              self.cat_prox(),
              self.cont_prox(),
              self.sparsity(),
              self.cat_diver(),
              self.cont_diver(),
              self.cont_count_divers()
              ]], columns =
            [
                'model_name',
                'validity',
                'cat_prox',
                'cont_prox',
                'sparsity',
                'cat_diver',
                'cont_diver',
                'cont_count_divers'
            ])
        