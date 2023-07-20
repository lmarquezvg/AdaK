import sys

from Public.ValidateArguments import validateArguments
from Public.ValidateObjectives import validateObjectives
from Public.UploadAlgorithm import uploadAlgorithm
from Public.SaveApproximationSet import saveApproximationSet
from Public.SaveGenerations import saveGenerations

if __name__ == '__main__':
    if (str(sys.argv[1]) == '--help'):
        f = open('../README.txt', 'r')
        contents = f.read()
        f.close()
        print(contents)
    else:
        if (len(sys.argv) != 7):
            sys.exit('Incorrect number of arguments. Use: main.py --help')
        algorithm = str(sys.argv[1])
        N = int(sys.argv[2])
        problem = str(sys.argv[3])
        m = int(sys.argv[4])
        max_generations = int(sys.argv[5])
        runs = int(sys.argv[6])
        
        validateArguments(algorithm, N, problem, m, max_generations, runs)
        m = validateObjectives(problem, m)
        
        for run in range(1, runs+1):
            print('Algorithm:', algorithm, '| Population size:', N, 
                  '| Problem:', problem, '| Objectives:', m, '| Generations:', 
                  max_generations, '| Run:', run)
            if (algorithm == 'MOEAD'):
                main = uploadAlgorithm(algorithm)
                P, Data_gen = main(N, problem, m, max_generations)
            elif (algorithm == 'AdaW'):
                main = uploadAlgorithm(algorithm)
                P, Data_gen = main(N, problem, m, max_generations)
            elif (algorithm == 'MOEAD-AdaRSE'):
                main = uploadAlgorithm('MOEAD-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 1)
            elif (algorithm == 'MOEAD-AdaGAE'):
                main = uploadAlgorithm('MOEAD-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 2)
            elif (algorithm == 'MOEAD-AdaCOU'):
                main = uploadAlgorithm('MOEAD-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 3)
            elif (algorithm == 'MOEAD-AdaPT'):
                main = uploadAlgorithm('MOEAD-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 4)
            elif (algorithm == 'MOEAD-AdaMPT'):
                main = uploadAlgorithm('MOEAD-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 5)
            elif (algorithm == 'MOEAD-AdaKRA'):
                main = uploadAlgorithm('MOEAD-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 6)
            elif (algorithm == 'NSGA-III'):
                main = uploadAlgorithm(algorithm)
                P, Data_gen = main(N, problem, m, max_generations)
            elif (algorithm == 'A-NSGA-III'):
                main = uploadAlgorithm(algorithm)
                P, Data_gen = main(N, problem, m, max_generations)
            elif (algorithm == 'NSGA-III-AdaRSE'):
                main = uploadAlgorithm('NSGA-III-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 1)
            elif (algorithm == 'NSGA-III-AdaGAE'):
                main = uploadAlgorithm('NSGA-III-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 2)
            elif (algorithm == 'NSGA-III-AdaCOU'):
                main = uploadAlgorithm('NSGA-III-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 3)
            elif (algorithm == 'NSGA-III-AdaPT'):
                main = uploadAlgorithm('NSGA-III-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 4)
            elif (algorithm == 'NSGA-III-AdaMPT'):
                main = uploadAlgorithm('NSGA-III-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 5)
            elif (algorithm == 'NSGA-III-AdaKRA'):
                main = uploadAlgorithm('NSGA-III-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 6)
            elif (algorithm == 'RVEA'):
                main = uploadAlgorithm(algorithm)
                P, Data_gen = main(N, problem, m, max_generations)
            elif (algorithm == 'RVEAa'):
                main = uploadAlgorithm(algorithm)
                P, Data_gen = main(N, problem, m, max_generations)
            elif (algorithm == 'RVEA-AdaRSE'):
                main = uploadAlgorithm('RVEA-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 1)
            elif (algorithm == 'RVEA-AdaGAE'):
                main = uploadAlgorithm('RVEA-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 2)
            elif (algorithm == 'RVEA-AdaCOU'):
                main = uploadAlgorithm('RVEA-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 3)
            elif (algorithm == 'RVEA-AdaPT'):
                main = uploadAlgorithm('RVEA-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 4)
            elif (algorithm == 'RVEA-AdaMPT'):
                main = uploadAlgorithm('RVEA-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 5)
            elif (algorithm == 'RVEA-AdaKRA'):
                main = uploadAlgorithm('RVEA-AdaK')
                P, Data_gen = main(N, problem, m, max_generations, 6)
            saveApproximationSet(P.obj, algorithm, problem, run, 'save_txt')
            saveGenerations(Data_gen, algorithm, problem, run)
            del P, Data_gen, main
