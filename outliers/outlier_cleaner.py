#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    #print(type(predictions))
    cleaned_data = []
    
    errors = predictions - net_worths
    errors = errors.tolist()
    ages = ages.tolist()
    net_worths = net_worths.tolist()
    for x in range(len(errors)):
        cleaned_data.append((ages[x][0],net_worths[x][0],errors[x][0]))
    
    cleaned_data =  sorted(cleaned_data,key = lambda x: x[2],reverse=False)
    #print(cleaned_data[:80])
    return cleaned_data[:81]

