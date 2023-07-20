"""
Validate objectives.
"""

def validateObjectives(problem, m):
    """Validates objectives per problem"""
    two_objectives = ['IMOP1', 'IMOP2', 'IMOP3']
    three_objectives = ['IMOP4', 'IMOP5', 'IMOP6', 'IMOP7', 'IMOP8', 'VNT1', 'VNT2', 'VNT3']
    if problem in two_objectives:
        return 2
    elif problem in three_objectives:
        return 3
    else:
        return m
