import itertools
import src.helpers.metrics as evaluation
import time
from pandas import DataFrame



class GridsSearchCF:
    
    """
    
    GridSearchCounterfactual: A class to perform grid search for counterfactuals.
    Inspired by the GridSearchCV class from sklearn.
    
    Iterates over all possible combinations of hyperparameters and fits the model to the data.
    
    Use:
    
    model = SomeModel()
    param_grid = {
        'param1': [value1, value2],
        'param2': [value3, value4]
    }
    
    grid_search = GridSearchCounterfactual(model, param_grid)
    
    grid_search.optimize(X, y)
    
    
    """
    
    
    
    def __init__(self, model, param_grid, scoring='accuracy', cv=5):
        self.model = model
        self.param_grid = param_grid # dictionary with hyperparameters to test
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
    
    #Creates a grid with ALL possible combinations for all the parameters.
    def __generate_param_combinations(params):
        
        #Create a list of parameter names and their corresponding values
        keys = list(params.keys())
        values = [params[key] for key in keys]

        #Generate all combinations using itertools.product
        combinations = itertools.product(*values)

        #Create a list of dictionaries for each combination
        combination_dicts = []
        for combination in combinations:
            # Map each combination to the corresponding parameter name
            combination_dict = dict(zip(keys, combination))
            combination_dicts.append(combination_dict)

        return combination_dicts
    
    
    
    def __evaluate(self, X, y):
        
        #Check if scoring is a string or a function
        
        if isinstance(self.scoring, str) and self.scoring in ['proximity', 'diversity', 'precision', 'recall']:
            return evaluation.evaluate(self.best_estimator_, X, y)[self.scoring]
        
        elif callable(self.scoring):
            return self.scoring(self.best_estimator_, X, y)
        
        raise ValueError("Invalid scoring metric")

    def optimize(self, X_test: DataFrame, y_test: DataFrame):
        
        optimal_score = 0
        optimal_params = None
        
        for param_combination in self.__generate_param_combinations(self.param_grid):
            
            score_subtotal = 0
            metric_collection = None
            
            #Track run-time
            start_time = time.time()
            
            # For every factual, we generate one or multiple counterfactuals
            for factual in X_test:   
            
                cf_model = self.model.set_params(**param_combination)
                counterfactuals = cf_model.optimize(X_test, y_test)
                
                if len(counterfactuals) > 0:
                    analysis = evaluation.CounterfactualAnalysis(model_instance, dataset)
                    
                    # add metrics to the metric collection as a dataframe
                    
                    if metric_collection is None:
                        
                        metric_collection = analysis.evaluate()
                        
                    else:
                        
                        metric_collection.append(analysis.evaluate())
        
            
            end_time = time.time()
            
            total_run_time = end_time - start_time
            
            if self.__evaluate(X, counterfactuals) > optimal_score:
                optimal_score = self.__evaluate(X, counterfactuals)
                optimal_params = param_combination
                self.best_metric_collection = metric_collection
                
        self.best_params_ = optimal_params
        self.best_score_ = optimal_score
        self.best_run_time
        
        

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)