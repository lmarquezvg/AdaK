Documentation of the module main.py used for the experimentation of: On the 
adaptation of reference sets using niching and pair-potential energy functions 
for multi-objective optimization.

NAME
       main.py - test a multi-objective evolutionary algorithm (MOEA)

SYNOPSIS
       main.py MOEA N MOP OBJS GENS RUNS
       main.py OPTION

DESCRIPTION
       This module is used to test a MOEA on a selected multi-objective problem 
       (MOP) with a given number of objective functions for a specific number 
       of independent runs. The required arguments are described as follows:

       MOEA
              It must be a valid MOEA name. The valid MOEA names are: NSGA-III, 
              A-NSGA-III, NSGA-III-AdaRSE, NSGA-III-AdaGAE, NSGA-III-AdaCOU, 
              NSGA-III-AdaPT, NSGA-III-AdaMPT, NSGA-III-AdaKRA, RVEA, RVEAa, 
              RVEA-AdaRSE, RVEA-AdaGAE, RVEA-AdaCOU, RVEA-AdaPT, RVEA-AdaMPT, 
              RVEA-AdaKRA, MOEAD, AdaW, MOEAD-AdaRSE, MOEAD-AdaGAE, 
              MOEAD-AdaCOU, MOEAD-AdaPT, MOEAD-AdaMPT, and MOEAD-AdaKRA.

       N
              It must be an integer greater than zero. It represents the 
              population size. In our experiments, we use 100, 91, 210, 156, 
              and 275 for MOPs with 2, 3, 5, 8, and 10 objective functions, 
              respectively.

       MOP
              It must be a valid MOP name. The valid MOP names are: DTLZ1, 
              DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ1_MINUS, 
              DTLZ2_MINUS, DTLZ3_MINUS, DTLZ4_MINUS, DTLZ5_MINUS, DTLZ6_MINUS, 
              DTLZ7_MINUS, WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, 
              WFG9, WFG1_MINUS, WFG2_MINUS, WFG3_MINUS, WFG4_MINUS, WFG5_MINUS, 
              WFG6_MINUS, WFG7_MINUS, WFG8_MINUS, WFG9_MINUS, IMOP1, IMOP2, 
              IMOP3, IMOP4, IMOP5, IMOP6, IMOP7, IMOP8, VNT1, VNT2, and VNT3.

       OBJS
              It must be an integer greater than one. It represents the number 
              of objective functions of the MOP. In our experiments, all MOPs 
              from the DTLZ, WFG, Minus-DTLZ, and Minus-WFG test suites are 
              scaled to 3, 5, 8, and 10 objective functions. The MOPs from the
              IMOP and VNT test suites are not scalable in the objective space.
              IMOP1-IMOP3 have 2 objective functions, while IMOP4-IMOP8 and 
              VNT1-VNT3 have 3 objective functions.

       GENS
              It must be an integer greater than or equal to zero. It 
              represents the maximum number of generations for the execution of 
              the MOEA. For DTLZ1, DTLZ5, and DTLZ6 the values 400, 600, 750, 
              and 1000 are selected for the versions with 3, 5, 8, and 10 
              objective functions, respectively. For DTLZ2 the values 250, 350, 
              500, and 750 are selected for the versions with 3, 5, 8, and 10 
              objective functions, respectively. For DTLZ3 and DTLZ7 the values 
              1000, 1000, 1000, and 1500 are selected for the versions with 3, 
              5, 8, and 10 objective functions, respectively. For DTLZ4 the 
              values 600, 1000, 1250, and 2000 are selected for the versions 
              with 3, 5, 8, and 10 objective functions, respectively. For WFG1-
              WFG9 the values 400, 750, 1500, and 2000 are selected for the 
              versions with 3, 5, 8, and 10 objective functions, respectively. 
              The same values are used for the MINUS versions of each MOP. For 
              IMOP1-IMOP3 we use 500 generations, while for IMOP4-IMOP8 and 
              VNT1-VNT3 we use 550 generations.

       RUNS
              It must be an integer greater than zero. It represents the number 
              of independent runs that the MOEA will be tested on the selected 
              MOP. In our experiments, all MOEAs are executed for 30 
              independent runs in each MOP.

       The following option can be used:

       --help 
              Display this help and exit.

REQUIREMENTS
       A computer with the installation of Python 3.8 is needed. The modules 
       numpy, matplotlib, and scipy are required.

EXAMPLE
       For running the module main.py, go to AdaK/ and write:

       IPython console users:
              %run main.py NSGA-III-AdaRSE 91 DTLZ1_MINUS 3 400 1

       Windows users:
              python main.py NSGA-III-AdaRSE 91 DTLZ1_MINUS 3 400 1

       Linux users:
              python3 main.py NSGA-III-AdaRSE 91 DTLZ1_MINUS 3 400 1

       The previous line executes the module main.py to test the 
       NSGA-III-AdaRSE on the DTLZ1_MINUS with 3 objective functions for one
       independent run. A population size of 91 is selected. The maximum number 
       of generations is set to 400.

RESULTS
       On success, the output files containing the final populations are 
       generated in AdaK/Results/Approximations/. Also, the output files 
       containing the evolutionary process are generated in 
       AdaK/Results/Generations/.

CONTENTS
       The folder "AdaK" contains the source code of the module main.py. The 
       folder "PlatEMO additional problems" contains additional problems for 
       PlatEMO which are employed for the comparison against state-of-the-art 
       MOEAs.

REFERENCE
       L.A. Márquez-Vega, J.G. Falcón-Cardona, and E. Covantes Osuna, On the 
       adaptation of reference sets using niching and pair-potential energy 
       functions for multi-objective optimization, Swarm and Evolutionary 
       Computation (2023), doi: 10.1016/j.swevo.2023.101408
       
       Or you can use the following BibTex entry:
       
       @article{MarquezEtAl2023,
              author = {Luis A. M{\'a}rquez-Vega and Jes{\'u}s Guillermo Falc{\'o}n-Cardona and Edgar {Covantes Osuna}},
              title = {{On the adaptation of reference sets using niching and pair-potential energy functions for multi-objective optimization}},
              journal = {Swarm and Evolutionary Computation},
              year = {2023},
              pages = {101408},
              issn = {2210-6502},
              doi = {10.1016/j.swevo.2023.101408}
       }

