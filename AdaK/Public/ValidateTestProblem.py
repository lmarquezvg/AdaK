"""
Validate test problem.
"""

def validateTestProblem(problem):
    """Returns True if a valid test problem is given, returns False otherwise"""
    valid_problems = ['DTLZ1', 
                      'DTLZ2', 
                      'DTLZ3', 
                      'DTLZ4', 
                      'DTLZ5', 
                      'DTLZ6', 
                      'DTLZ7', 
                      'DTLZ1_MINUS', 
                      'DTLZ2_MINUS', 
                      'DTLZ3_MINUS', 
                      'DTLZ4_MINUS', 
                      'DTLZ5_MINUS', 
                      'DTLZ6_MINUS', 
                      'DTLZ7_MINUS', 
                      'WFG1', 
                      'WFG2', 
                      'WFG3', 
                      'WFG4', 
                      'WFG5', 
                      'WFG6', 
                      'WFG7', 
                      'WFG8', 
                      'WFG9', 
                      'WFG1_MINUS', 
                      'WFG2_MINUS', 
                      'WFG3_MINUS', 
                      'WFG4_MINUS', 
                      'WFG5_MINUS', 
                      'WFG6_MINUS', 
                      'WFG7_MINUS', 
                      'WFG8_MINUS', 
                      'WFG9_MINUS']
    if (problem in valid_problems):
        return True
    else:
        return False
