"""
Save generations.
"""

import numpy as np

def saveGenerations(Data_gen, algorithm, problem, run):
    """Saves the evolutionary process"""
    gens = len(Data_gen)
    N_gen = []
    for i in range(0, gens):
        N_gen.append(len(Data_gen[i]))
        if i == 0:
            P_gen = Data_gen[i]
        else:
            P_gen = np.vstack((P_gen, Data_gen[i]))
    N, m = np.shape(P_gen)
    np.savetxt('Results/Generations/'+algorithm+'_'+problem+'_{0:0=2d}D'.format(m)+'_R{0:0=2d}'.format(run)+'.ngen', N_gen, fmt='%d', header=str(gens)+' 1')
    np.savetxt('Results/Generations/'+algorithm+'_'+problem+'_{0:0=2d}D'.format(m)+'_R{0:0=2d}'.format(run)+'.pgen', P_gen, fmt='%.6e', header=str(N)+' '+str(m))
    return
