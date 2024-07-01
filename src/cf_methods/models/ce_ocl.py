import src.cf_methods.cf_method as boilerplate
import src.cf_methods.models.helpers.ce_ocl_helper as em
from pyomo.environ import *
from pandas import DataFrame
import dice_ml
import numpy as np
import pandas as pd


class CE_OCL(boilerplate.CounterfactualMethod):
    
    """
    
    A simple wrapper for the CE-OCL algorithm.
    
    Use:
    
    ce_ocl = CE_OCL()
    
    
    """
    
    re_run = 0
    
    def set_data_restrictions(self,
                   real_features = [],
                   binary_features = [],
                   integer_features = [],
                   categorical_encodings = {},
                   only_positive_features = [],
                   only_increasing_features = [],
                   immutable_features = [],
                   conditionally_mutable_features = []
                   ):
        
        self.real_features = real_features
        self.binary_features = binary_features
        self.integer_features = integer_features
        
        self.categorical_encodings = categorical_encodings
        #e.g. {'gender': ['gender_female', 'gender_male', 'gender_non-binary]}
        
        self.only_positive_features = only_positive_features
        self.only_increasing_featues = only_increasing_features
        
        self.immutable_features = immutable_features
        self.conditionally_mutable_features = conditionally_mutable_features
        
    def set_metric_constraints(self, sparsity_constraint = False, trust_region_constraint = False, trust_region_reference = None):
        
        self.sparsity_constraint = sparsity_constraint
        self.trust_region_constraint = trust_region_constraint
        self.trust_region_reference = trust_region_reference
        
        self.enlarge_trust_region_once = False
        
    
    def generate(self, factual: DataFrame, nr_cfs=1):
        
        #X_train = self.df.drop(self.target, axis=1)
        
        try:
            CEs, CEs_, final_model = self._opt_api(
                X = self.df,  
                X1 = self.trust_region_reference,
                u = factual.iloc[0, :],
                F_r = self.real_features,
                F_b = self.binary_features,
                F_int = self.integer_features,
                F_coh = self.categorical_encodings,
                I = self.immutable_features,
                L = self.only_increasing_featues,
                Pers_I = self.conditionally_mutable_features,
                P = self.only_positive_features,
                sp = self.sparsity_constraint,
                mu = self.sparsity, # hyperparameter
                tr_region = self.trust_region_constraint,
                enlarge_tr = self.enlarge_trust_region_once,
                num_counterfactuals = nr_cfs,
                model_master=self.model_master   
            )
            
            self.enlarge_trust_region_once = False
            print("FOUND")
            return CEs
            
        except Exception as e:
            
            print('Optimization error: ', e)
            
            # Enlarge the trust region ONCE if the optimization fails
            # (if the trust region constraint is active)
            if self.re_run != 1 and self.trust_region_constraint:
                
                self.re_run += 1
                self.enlarge_trust_region_once = True
                return self.generate(nr_cfs)
            
            self.re_run = 0
            print('Optimization failed twice. Returning empty DataFrame')
            return DataFrame()
        
        


    # Algorithm taken from https://github.com/tabearoeber/CE-OCL/blob/main/src/ce_helpers.py
    # Algorithm has been changed as little as possible in order to maintain the original author's intent
    # Nice algorithm, by the way :)
    # But it does not contribute to mass adoption of CFs since it is "harder" to understand than the other ones
    
    def _opt_api(self, X, X1, u, F_r, F_b, F_int, F_coh, I, L, Pers_I, P, sp, mu, tr_region, enlarge_tr, num_counterfactuals,
            model_master, scaler=None, obj='l2'):
        sparsity_RHS = len(X.columns)

        conceptual_model = self.optimization_model(X, u, F_r, F_b, F_int, F_coh, mu, sparsity_RHS, I, L,
                                                                Pers_I, P, obj=obj, sparsity=sp, tr=tr_region)
        
        
        MIP_final_model = em.optimization_MIP(conceptual_model, conceptual_model.x, model_master, X1, tr=tr_region,
                                            enlarge_tr=enlarge_tr)
        
        opt = SolverFactory('gurobi_persistent')
        opt.set_instance(MIP_final_model)
        opt.set_gurobi_param('PoolSolutions', num_counterfactuals + 100)
        # opt.set_gurobi_param('PoolSolutions', num_counterfactuals)
        opt.set_gurobi_param('PoolSearchMode', 1)

        # opt.options['Solfiles'] = 'solution'+version
        results = opt.solve(MIP_final_model, load_solutions=True, tee=False)
        print('OBJ:', value(MIP_final_model.OBJ))

        solution = []
        for i in X.columns:
            solution.append(value(MIP_final_model.x[i]))
        print(f'The optimal solution is: {solution}')

        number_of_solutions = opt.get_model_attr('SolCount')
        # print(f"########################Num of solutions###################### {number_of_solutions}")
        CEs = pd.DataFrame([u])
        for i in range(number_of_solutions):
            opt.set_gurobi_param('SolutionNumber', i)
            suboptimal_solutions = opt.get_model_attr('Xn')

            vars_name_x = [opt.get_var_attr(MIP_final_model.x[i], 'VarName') for i in X.columns]
            vars_name_ix = [int(vars_name_x[i].replace('x', '')) for i in range(len(vars_name_x))]
            # print(vars_name_ix)
            vars_val_x = [suboptimal_solutions[i - 1] for i in vars_name_ix]
            solution_i = {X.columns[i]: vars_val_x[i] for i in range(len(vars_val_x))}
            solution_i = pd.DataFrame(solution_i, index=[0])
            CEs = pd.concat([CEs, solution_i], ignore_index=True)

        CEs.reset_index(drop=True, inplace=True)

        CEs['scaled_distance'] = [np.round(sum(abs(u[i] - CEs.loc[j, i]) for i in X.columns), 4) for j in CEs.index]
        # CEs['sparsity'] = [sum(1 if np.round(u[i]-CEs.loc[j,i], 3) != 0 else 0 for i in X.columns) for j in CEs.index]
        # CEs['obj value'] = [CEs.loc[j,'sparsity']*mu + CEs.loc[j,'scaled_distance'] for j in CEs.index]
        CEs = CEs.round(4).drop_duplicates()

        ix_names = ['original'] + ['sol' + str(i) for i in range(len(CEs.index))]
        ix = {i: ix_names[i] for i in range(len(CEs.index))}
        CEs = CEs.reset_index(drop=True).rename(index=ix)
        CEs = CEs.iloc[:num_counterfactuals + 1, :]

        # reverse scaling
        CEs_ = CEs.iloc[:, :-1].copy()

        if scaler is not None:
            try: scaled_xdata_inv = scaler['preprocessor'].named_transformers_['num'].inverse_transform(CEs_[F_r])
            except: scaled_xdata_inv = scaler.inverse_transform(CEs_[F_r])
            #        scaled_xdata_inv = scaler.inverse_transform(CEs_[F_r])
            CEs_.loc[:, F_r] = scaled_xdata_inv

        return CEs, CEs_, MIP_final_model
            

def optimization_model(X, u, F_r, F_b, F_int, F_coh, mu, sparsity_RHS, I, L, Pers_I, P, obj='l2', sparsity=True,
                              tr=False):
    '''
    u: given data point
    F_r = set of real features that describe u
    F_b = set of binary features that describe u
    F_i = set of integer features that describe u
    I = set of immutable features
    Pers_I = variables that are conditionally mutable
    L = set of features that should only increase
    P = set of features that must take on a positive value
    mu = positive hyperparameter per the sparsity constraint
    obj = which objective function to use?
    sparsity = True or False. include sparsity constraint or not?
    '''
    # complete set of features
    F = X.columns
    # print(F)
    big_M = 10000

    model = ConcreteModel('CE')

    'Decision variables'
    model.z = Var(F, domain=Binary,
                  name=['Zaux_%s' % str(ce) for ce in F])  # auxiliary vars for the sparsity constraint
    model.t = Var(F, domain=PositiveReals,
                  name=['Taux_%s' % str(ce) for ce in F])  # auxiliary vars for the MAD objective function
    model.x = Var(F, domain=Reals, name=['ce_%s' % str(ce) for ce in F])  # counterfactual features
    model.e = Var(F, domain=Reals, name=['epsilon_%s' % str(x) for x in F])

    for i in F_b:
        model.x[i].domain = Binary

    for i in F_int:
        model.x[i].domain = NonNegativeIntegers

    for cat in F_coh.keys():
        model.add_component('coherence_' + cat, Constraint(expr=sum(model.x[i] for i in F_coh[cat]) == 1))

    'Objective function'

    def obj_function_l2norm(model, sparsity=sparsity):
        return sum((u[i] - model.x[i]) ** 2 for i in F) + mu * sum(model.z[i] for i in F)

    def obj_function_MAD(model):
        MAD = {
            i: stats.median_absolute_deviation(X[i]) if stats.median_absolute_deviation(X[i]) > 1e-4 else 1.48 * np.std(
                X[i]) for i in F}
        return sum(model.t[i] / MAD[i] for i in F_r + F_int) + sum(model.t[i] / MAD[i] for i in F_b) + mu * sum(
            model.z[i] for i in F)

    def obj_function_l1norm(model):
        return sum(model.t[i] for i in F) + mu * sum(model.z[i] for i in F)

    assert obj in ['l2', 'l1MAD'], "Invalid objective function; please choose between l2, l1, l1MAD"
    if obj == 'l2':
        model.OBJ = Objective(rule=obj_function_l2norm, sense=minimize)
    elif obj == 'l1MAD':
        model.OBJ = Objective(rule=obj_function_MAD, sense=minimize)
    elif obj == 'l1':
        model.OBJ = Objective(rule=obj_function_l1norm, sense=minimize)

        'Auxiliary constraint for t'

        def MAD1(model, i):
            return model.t[i] >= u[i] - model.x[i]

        def MAD2(model, i):
            return model.t[i] >= - u[i] + model.x[i]

        model.Constraint01 = Constraint(F, rule=MAD1)
        model.Constraint02 = Constraint(F, rule=MAD2)

    if not tr:
        def constraint_CTR2(model, i):
            return model.e[i] == 0

        model.ConstraintClusteredTrustRegion2 = Constraint(F, rule=constraint_CTR2)

    if sparsity == True:
        'Sparsity constraints'

        def sparsity11(model, i):
            return model.x[i] - u[i] <= big_M * model.z[i]

        model.Constraint11 = Constraint(F, rule=sparsity11)

        def sparsity12(model, i):
            return -(model.x[i] - u[i]) <= big_M * model.z[i]

        model.Constraint12 = Constraint(F, rule=sparsity12)

    # This is not necessary since we already have a penalizing term in the objective function
    #     def sparsity2(model):
    #         return sum(model.z[i] for i in F) <= sparsity_RHS
    #     model.Constraint2 = Constraint(rule=sparsity2)

    'Immutable features constraints'

    for k in Pers_I:
        for j in k:
            # print(j)
            if j in u.index:
                if u[j] == 1:
                    I = I + k[:k.index(j)]
            else:
                I = I + k[:k.index(j)]

    def immutability(model, i):
        # print()
        return model.x[i] == u[i]

    model.Constraint3 = Constraint(I, rule=immutability)

    'Larger or equal to u[i]'

    def larger(model, i):
        return model.x[i] >= u[i]

    model.Constraint4 = Constraint(L, rule=larger)

    def positive(model, i):
        return model.x[i] >= 0

    model.Constraint5 = Constraint(P, rule=positive)

    return model