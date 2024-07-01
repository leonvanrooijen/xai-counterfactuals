import itertools
import src.helpers.metrics as evaluation
import time
import pandas as pd



class GridSearchCF:
    
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
    
    best_score = 0 # best score based on metric
    best_params = None # best hyper parameters based on metric
    best_params_evaluation_results = None # all results
    best_params_runtime = None # runtime of the best hyper parameters    
    
    
    
    def __init__(self, model, param_grid, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid # dictionary with hyperparameters to test
        self.scoring = scoring
    
    #Creates a grid with ALL possible combinations for all the parameters.
    def __generate_param_combinations(self, params):
        
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
    
    
    
    def __evaluate(self, counterfactuals: pd.DataFrame, scores):
        
        #Check if scoring is a string or a function
        
        whitelist = [
                'validity',
                'cat_prox',
                'cont_prox',
                'sparsity',
                'cat_diver',
                'cont_diver',
                'cont_count_divers'
        ]
        
        if isinstance(self.scoring, str) and self.scoring in whitelist:
            return scores[self.scoring]
        
        elif callable(self.scoring):
            return self.scoring(scores)
        
        raise ValueError("Invalid scoring metric")

    def optimize(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        
        optimal_score = 0
        optimal_params = None

        for param_combination in self.__generate_param_combinations(self.param_grid):
            
            score_subtotal = 0
            metric_collection = None
            
            #Track run-time
            start_time = time.time()
            
            # For every factual, we generate one or multiple counterfactuals
            for i in range(len(X_test)):
            
                
                X_factual = X_test.iloc[[i]]    
                y_factual = y_test.iloc[i]
            
                self.model.set_params(**param_combination)
                counterfactuals = self.model.generate(X_factual, y_factual)
                
                if len(counterfactuals) > 0:
                    print("we found a cf")
                    analysis = evaluation.CounterfactualAnalysis(counterfactuals, rounding=3)
                    
                    # add metrics to the metric collection as a dataframe
                    
                    if metric_collection is None:
                        metric_collection = analysis.evaluate()
                        
                    else:
                        metric_collection = pd.concat([metric_collection, analysis.evaluate()])
        
            
            end_time = time.time()
            
            total_run_time = end_time - start_time
            
            if metric_collection is None:
                print("No counterfactuals generated for this parameter combination: ", param_combination)
                continue
            
            metrics_avg = metric_collection.mean()
            metrics_to_dict = metrics_avg.to_dict()
            
            if self.__evaluate(metrics_to_dict) > optimal_score:
                optimal_score = self.__evaluate(counterfactuals, metrics_to_dict)
                optimal_params = param_combination
                self.best_params_evaluation_results = metric_collection
                
        self.best_params = optimal_params
        self.best_score = optimal_score
        self.best_params_run_time = total_run_time
        
        

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)