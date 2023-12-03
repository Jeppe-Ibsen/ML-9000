
import itertools
import math
import os
import timeit
import numpy as np
import logging as logger
# init logger

# Initialize logging
logFilename = 'svm_clsf_grid_search_manuel.log'
# Setup basic logging (console logging in this case)
logger.basicConfig(filename=logFilename, level=logger.INFO)
log = logger.getLogger(__name__)

def add_file_handler(log, filename):
    file_handler = logger.FileHandler(filename)
    file_handler.setLevel(logger.DEBUG)
    formatter = logger.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

def remove_file_handlers(log):
    for handler in log.handlers[:]:
        if isinstance(handler, logger.FileHandler):
            log.removeHandler(handler)
            handler.close()



def svm_clsf_grid_search_manuel(X_train, y_train, param_grid:dict, cv:int=5, processors:int=6,   verbose:int=0, results_to_print:int=5, print_time:int=0, log_level:int=0, LogFilename:str=logFilename, shuffle:bool=True):
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import make_scorer
    
    # get shortdate as string
    from datetime import datetime
    now = datetime.now()
    shortdate = now.strftime("%Y%m%d_%H%M")
    # add shortdate to logfilename
    
    LogFilename = shortdate + "_" + LogFilename + ".log"
    # set LogFilename   
    remove_file_handlers(log)
    add_file_handler(log, LogFilename)

    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LogFilename)
    if(log_level>0):
        print("Log path from logger:", log_file_path)
    # define scoring f1_micro
    scoring = {'Accuracy': make_scorer(accuracy_score)}
    if(verbose>2):
        print("Scoring: ", scoring)
        print("ParamsGrid: ", param_grid)
    if(log_level>2):
        log.info(f"Scoring: {scoring}")
        log.info(f"ParamsGrid: {param_grid}")
    # adjust gamma
    param_grid = adjust_gamma_params(param_grid)
    # define Each model parameter to test 
    paramaters = list(param_grid.keys())
    paramaters_values = [value if isinstance(value, list) or isinstance(value, np.ndarray) else [value] for value in param_grid.values()]

    if(verbose>2):
        print("Parameter: ", paramaters)
        print("Parameter values: ", paramaters_values)
    if(log_level>2):
        log.info(f"Parameter: {paramaters}")
        log.info(f"Parameter values: {paramaters_values}")

    
    paramaters_combinations = list(itertools.product(*paramaters_values))
    paramaters_combinations = [dict(zip(paramaters, paramaters_combinations[i])) for i in range(len(paramaters_combinations))]
    # shuffle the combinations
    if(shuffle):
        np.random.shuffle(paramaters_combinations)
    
    number_of_combinations = len(paramaters_combinations)
    if(verbose>0):
        print("Number of combinations: ", number_of_combinations)
        if(verbose>3):
            print("Parameter combinations: ", paramaters_combinations)
    if(log_level>0):
        log.info(f"Number of combinations: {number_of_combinations}")
        if(log_level>3):
            log.info(f"Parameter combinations: {paramaters_combinations}")

    
    runtimesStarts = []
    # define the grid search
    for parameter in paramaters_combinations:
        
        # on first iteration calculate the time
        if('start' not in locals()):
            start = timeit.default_timer()
            runtimesStarts.append(start)
        
        model = SVC()
        model.set_params(**parameter)
        if(verbose>0):
            print("Testing parameters: ", parameter)
        if(log_level>1):
            log.info(f"Testing parameters: {parameter}")
        # define the cross validation
        from sklearn.model_selection import cross_validate
        #
        cv_verbose = 2
        if(verbose == 0):
            cv_verbose = 0
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=processors, return_train_score=True, verbose=cv_verbose)
        # print the results
        if(verbose>3):
            print("Train Accuracy: ", cv_results['train_Accuracy'].mean())
            print("Test Accuracy: ", cv_results['test_Accuracy'].mean())
        if(log_level>3):
            log.info(f"Train Accuracy: { cv_results['train_Accuracy'].mean()}")
            log.info(f"Test Accuracy:  {cv_results['test_Accuracy'].mean()}")

        # save the results
        parameter['train_Accuracy'] = cv_results['train_Accuracy'].mean()
        parameter['test_Accuracy'] = cv_results['test_Accuracy'].mean()
        # save the best model
        if('best_model' not in locals()):
            best_model = model
            best_parameters = parameter
        elif(cv_results['test_Accuracy'].mean() > best_parameters['test_Accuracy']):
            best_model = model
            best_parameters = parameter
        # save the results
        if('results' not in locals()):
            results = parameter
        else:
            results = np.vstack((results, parameter))
        # print the results
        if(verbose>4):
            print("Train Accuracy: ", cv_results['train_Accuracy'].mean())
            print("Test Accuracy: ", cv_results['test_Accuracy'].mean())
            print("Best Accuracy: ", best_parameters['test_Accuracy'])
            print("Best parameters: ", best_parameters)
            
            print("-------------------------------------------------------")
        if(log_level>4):
            log.info("Train Accuracy: %s", cv_results['train_Accuracy'].mean())
            log.info("Test Accuracy: %s", cv_results['test_Accuracy'].mean())
            log.info("Best Accuracy: %s", best_parameters['test_Accuracy'])
            log.info("Best parameters: %s", best_parameters)
            log.info("-------------------------------------------------------")
        #
        # print progress 
        if(verbose>0):
            print("Progress: ", len(results), "/", number_of_combinations)
        if(log_level>0):
            log.info("Progress: %s / %s", len(results), number_of_combinations)


        # save the time for the first iteration
        if('start' in locals()):
            stop = timeit.default_timer()
            time = stop - start
            # calculate the time left
            time_left = time * (number_of_combinations - len(results))
            # print the time left
            if(print_time>1):
                print("Time left: ", time_left)
                print("Time spent: ", time)
                if(print_time>2):
                    if(len(results)>0):
                        print("Time spent per iteration: ", time/len(results))
                    if((number_of_combinations - len(results))>0):
                        print("Time left per iteration: ", time_left/(number_of_combinations - len(results)))
                print("-------------------------------------------------------")
            if(log_level>1):
                log.info("Time left: %s", time_left)
                log.info("Time spent: %s", time)
                if(log_level>2):
                    if(len(results)>0):
                        log.info("Time spent per iteration: %s", time/len(results))
                    if((number_of_combinations - len(results))>0):
                        log.info("Time left per iteration: %s ", time_left/(number_of_combinations - len(results)))
                log.info("-------------------------------------------------------")
            # delete the start time
            del start
        # 
        
    # print the results
    if(verbose>0):
        print("Best Accuracy: ", best_parameters['test_Accuracy'])
        print("Best parameters: ", best_parameters)
    if(log_level>0):
        log.info("Best Accuracy: %s", best_parameters['test_Accuracy'])
        log.info("Best parameters: %s", best_parameters)
    if(verbose>1):
        
        print("the ", results_to_print,  " best results: ")
        #sort the results by test_Accuracy
        sortedResults = sorted(results, key=lambda x: x[0]['test_Accuracy'], reverse=True)
        for i in range(results_to_print):
            if(i<len(sortedResults)):
                print(sortedResults[i])
    if(log_level>1):
        log.info("the %s best results: ", results_to_print)

        #sort the results by test_Accuracy
        sortedResults = sorted(results, key=lambda x: x[0]['test_Accuracy'], reverse=True)
        for i in range(results_to_print):
            if(i<len(sortedResults)):
                 log.info("%s", sortedResults[i])
    
    if(print_time>0):
        print("Total Time spent: ", timeit.default_timer() - runtimesStarts[0])
    
    if(log_level>0):
        log.info("Total Time spent: %s", timeit.default_timer() - runtimesStarts[0])

    return best_model, best_parameters, results



def adjust_gamma_params(parameter_grid):
    # Make a copy of the original parameter grid to avoid modifying it directly
    adjusted_params = parameter_grid.copy()
    
    # Check if 'gamma' is in the parameter grid
    if 'gamma' in parameter_grid:
        # Initialize an empty list to hold the adjusted gamma values
        adjusted_gammas = []
        
        
        # Iterate over the gamma values
        for gamma in parameter_grid['gamma']:
            # If gamma is not 'scale' or 'auto', convert it to float
            if gamma not in ['scale', 'auto']:
                try:
                    # Convert numeric string to float
                    adjusted_gammas.append(float(gamma))
                except ValueError:
                    # If conversion fails, raise an error
                    raise ValueError(f"Gamma value '{gamma}' is not valid. It must be 'scale', 'auto', or a float.")
            else:
                # If gamma is 'scale' or 'auto', keep it as is
                adjusted_gammas.append(gamma)
        
        # Replace the original 'gamma' list with the adjusted list
        adjusted_params['gamma'] = adjusted_gammas
    
    return adjusted_params

# # Your original parameters
# original_parameters = {
#     'kernel': ['rbf'],
#     'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
#     'gamma': ['0.1', '0.5', 'scale', 'auto']
# }

# # Adjust the 'gamma' values in the parameters
# adjusted_parameters = adjust_gamma_params(original_parameters)

# # Now you can use 'adjusted_parameters' with your grid search

def esitmate_gridsearch_time(model, X_train , y_train, param_grid:dict, printParams:bool=False, cv:int=5, processors:int=6,  pression:int=5):
    times = []

    #print params combinations
    if printParams:
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            print(params)

    #print Paramscounts 
    print("params count = ",len(list(itertools.product(*param_grid.values()))))
    #print Xtrain shape and size
    print("X_train.shape = ",X_train.shape)
    print("X_train.size = ",X_train.size)
    #print ytrain shape and size
    print("y_train.shape = ",y_train.shape)
    print("y_train.size = ",y_train.size)
    #print cv and processors
    print("cv = ",cv)
    print("processors = ",processors)
    #print pression
    print("pression/iterations for fits = ",pression)

    
    for _ in range(pression):
        start = timeit.default_timer()
        print("Fitting # ",_)
        model.fit(X_train, y_train)
        print("Score # ",_)
        model.score(X_train, y_train)
        timedif = timeit.default_timer() - start
        print("Time # ",_, " = ",timedif)
        times.append(timedif)

    single_train_time = np.array(times).mean() # seconds
    print("single_train_time = ",single_train_time)

    combos = 1
    for vals in param_grid.values():
        combos *= len(vals)

    num_models = combos * cv / processors
    seconds = num_models * single_train_time
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    print(hours, " hours ", minutes, " minutes ", seconds, " seconds ")
    #make it pretty
    seconds = seconds % 60
    minutes = minutes % 60
    hours = hours % 24
    days = math.floor(days)
    
    print("Grid Search will take {:.0f} days {:.0f} hours {:.0f} minutes {:.0f} seconds to complete.".format(days,hours, minutes, seconds))

    
def esitmate_gridsearch_time_by_runtime_array(Runtimes, X_train , y_train, param_grid:dict, printParams:bool=False, cv:int=5, processors:int=6,  pression:int=5):
    times = Runtimes #np.array(Runtimes)

    #print params combinations
    if printParams:
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            print(params)

    #print Paramscounts 
    print("params count = ",len(list(itertools.product(*param_grid.values()))))
    #print Xtrain shape and size
    print("X_train.shape = ",X_train.shape)
    print("X_train.size = ",X_train.size)
    #print ytrain shape and size
    print("y_train.shape = ",y_train.shape)
    print("y_train.size = ",y_train.size)
    #print cv and processors
    print("cv = ",cv)
    print("processors = ",processors)
    #print pression
    print("pression/iterations for fits = ",pression)

    single_train_time = np.array(times).mean() # seconds
    print("single_train_time = ",single_train_time)

    combos = 1
    for vals in param_grid.values():
        combos *= len(vals)

    num_models = combos * cv / processors
    seconds = num_models * single_train_time
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    print(hours, " hours ", minutes, " minutes ", seconds, " seconds ")
    #make it pretty
    seconds = seconds % 60
    minutes = minutes % 60
    hours = hours % 24
    days = math.floor(days)
    
    print("Grid Search will take {:.0f} days {:.0f} hours {:.0f} minutes {:.0f} seconds to complete.".format(days,hours, minutes, seconds))

    