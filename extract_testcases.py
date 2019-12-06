
import pandas as pd
from vector import text_to_vector,get_cosine

req = "create a log in page with user name and password"


def recommend_testcases( requirement ) :

    recommendation = {'testcase':[]}

    test_cases = pd.read_csv('data\\train\\testcases.csv', header=None)


    for test_case in test_cases[0]:
        vector1=text_to_vector(test_case)
        vector2=text_to_vector(req)
        cosine =get_cosine(vector1,vector2)
        if (cosine >0.1) :
            recommendation['testcase'].append(test_case)
    
    return recommendation


print(recommend_testcases(req))