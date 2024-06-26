from pandas import DataFrame
import numpy as np

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
    
    def __init__(self, model: str, counterfactuals: DataFrame, rounding: int = None):
        
        # Check if the input DataFrames are valid (there must be at least 1 counterfactual and 1 factual in the DF)
        if len(counterfactuals) <= 1:
            raise ValueError("Counterfactual DataFrame should have at least one solution")
        
        self.rounding = rounding
        self.nr_counterfactuals = len(counterfactuals) - 1
        self.counterfactuals = counterfactuals


    def categorical_prox_score(self, F_cat):

        if len(F_cat) > 0:
            cat_prox = 1 - sum(sum((self.counterfactuals.loc['original', F_cat]) != (self.counterfactuals.loc[i, F_cat])) for i in self.counterfactuals.index[1:]) /
            (self.nr_counterfactuals  * len(F_cat))
        else:
            cat_prox = None

        return cat_prox

    def continuous_prox_score(self, F_r):
        
        if len(F_r) > 0:
            cont_prox = 1 - sum(sum(np.abs(self.counterfactuals.loc['original', i] - self.counterfactuals.loc[j, i]) for i in F_r) for j in self.counterfactuals.index[1:]) / self.nr_counterfactuals -sum(sum(np.abs(self.counterfactuals.loc['original', i] -
            self.counterfactuals.loc[j, i]) for i in F_r) for j in self.counterfactuals.index[1:]) / self.nr_counterfactuals 
        else:
            cont_prox = None
            
        return cont_prox

    def sparsity_score(self):
        
        orig = self.counterfactuals[:1]
        
        sparsity = 1 - sum(
            sum(1 if orig.squeeze()[i] != self.counterfactuals.loc[j, i] else 0 for i in self.counterfactuals.columns) for j in self.counterfactuals.index[1:]) / (
                        self.nr_counterfactuals  * len(self.counterfactuals.columns))
        
        return sparsity
        
        
    def continuous_diversity_score(self, F_cat, F_r):
        # continuous variables
        cont_diver = None
        
        if self.nr_counterfactuals  > 1:
            
            cont_diver_numerator = 0
            
            for jx, j in enumerate(self.counterfactuals.index[1:-1]):
                for i in self.counterfactuals.index[jx + 1:]:
                    if i != j:
                        cont_diver_numerator += sum(
                            np.abs(np.round(self.counterfactuals.loc[i, F_r].astype(float), 4) - np.round(self.counterfactuals.loc[j, F_r].astype(float), 4)))
            
            cont_diver_denominator = self.nr_counterfactuals  * (self.nr_counterfactuals  - 1) / 2
            cont_diver = cont_diver_numerator / cont_diver_denominator
            
            
    def categorical_diversity_score(self, F_cat, F_r):
        
        cat_diver = None
        
        if self.nr_counterfactuals  > 1:
            
            if len(F_cat) > 0:
                
                cat_diver_numerator = 0
                
                for jx, j in enumerate(self.counterfactuals.index[1:-1]):
                    for i in self.counterfactuals.index[jx + 1:]:
                        if i != j:
                            cat_diver_numerator += sum(self.counterfactuals.loc[i, F_cat] != self.counterfactuals.loc[j, F_cat])
                
                cat_diver_denominator = self.nr_counterfactuals  * (self.nr_counterfactuals  - 1) / 2 * len(F_cat)
                
                cat_diver = cat_diver_numerator / cat_diver_denominator

        return cat_diver  
    
    
    def continuous_count_diversity(self):
        
        cont_count_divers = None
        
        if self.nr_counterfactuals  > 1:
            
            sparsity_diver_numerator = 0
            
            for jx, j in enumerate(self.counterfactuals.index[1:-1]):
                for i in self.counterfactuals.index[jx + 1:]:
                    if i != j:
                        sparsity_diver_numerator += sum(np.abs(self.counterfactuals.loc[i, :] != self.counterfactuals.loc[j, :]))
            sparsity_diver_denominator = self.nr_counterfactuals  * (self.nr_counterfactuals  - 1) / 2 * len(self.counterfactuals.columns)
            cont_count_divers = sparsity_diver_numerator / sparsity_diver_denominator

        return cont_count_divers   
    
    def __round(self, value):
            if self.rounding is not None:
                return round(value, self.rounding)
            return value    
     
    def evaluate(self):
        
        return DataFrame(
            [[
              self.model,
              self.__round(self.validity()),
              self.__round(self.cat_prox()),
              self.__round(self.cont_prox()),
              self.__round(self.sparsity_score()),
              self.__round(self.categorical_diversity_score()),
              self.__round(self.continuous_diversity_score()),
              self.__round(self.continous_count_divers())
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
        