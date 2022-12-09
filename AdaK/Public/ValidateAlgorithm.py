"""
Validate algorithm.
"""

def validateAlgorithm(algorithm):
    """Returns True if a valid algorithm is given, returns False otherwise"""
    valid_algorithms = ['MOEAD',
                        'AdaW',
                        'MOEAD-AdaRSE',
                        'MOEAD-AdaGAE',
                        'MOEAD-AdaCOU',
                        'MOEAD-AdaPT',
                        'MOEAD-AdaMPT',
                        'MOEAD-AdaKRA',
                        'NSGA-III',
                        'A-NSGA-III',
                        'NSGA-III-AdaRSE', 
                        'NSGA-III-AdaGAE', 
                        'NSGA-III-AdaCOU', 
                        'NSGA-III-AdaPT', 
                        'NSGA-III-AdaMPT', 
                        'NSGA-III-AdaKRA',
                        'RVEA',
                        'RVEAa',
                        'RVEA-AdaRSE',
                        'RVEA-AdaGAE',
                        'RVEA-AdaCOU',
                        'RVEA-AdaPT',
                        'RVEA-AdaMPT',
                        'RVEA-AdaKRA']
    if (algorithm in valid_algorithms):
        return True
    else:
        return False
