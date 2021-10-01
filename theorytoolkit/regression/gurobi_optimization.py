__author__ = "Peichen Zhong"


import numpy as np
from gurobipy import *


def l0l1_diretct_optimize(A, f, mu0, mu1, M=1000.0, cutoff=300):
    """
    Brute force solving of L0L1-regularization problem using Gurobi.
    All ECIs are regarded as mathematical programming problem.
    No real physical constrains added here.

    Parameters
    ----------
    A: feature matrix as numpy array of shape mxn
    f: scalar property vector as numpy array of shape mx1
    mu0: regularization parameter of L0 term as float
    mu1: regularization parameter of L1 term as float
    M: upper bound of absolute value of ECI used to build relevant constraints, e.g. For all ECIs, |ECI| < M.
    cutoff: cutoff for solving the optimization problem in seconds


    Returns
    -------
    Fitted ECI values as numpy array of shape A.shape[1]
    """
    n = A.shape[0]
    d = A.shape[1]
    ATA = A.T @ A
    fTA = f.T @ A

    l1l0 = Model()
    w = l1l0.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    z0 = l1l0.addVars(d, vtype=GRB.BINARY)
    z1 = l1l0.addVars(d)
    for i in range(d):
        l1l0.addConstr(M * z0[i] >= w[i])
        l1l0.addConstr(M * z0[i] >= (-1.0 * w[i]))
        l1l0.addConstr(z1[i] >= w[i])
        l1l0.addConstr(z1[i] >= (-1.0 * w[i]))

    L = QuadExpr()
    for i in range(d):
        L = L + mu0 * z0[i]
        L = L + mu1 * z1[i]
        L = L - 2 * fTA[i] * w[i]

        for j in range(d):
            L = L + w[i] * w[j] * ATA[i][j]

    l1l0.setObjective(L, GRB.MINIMIZE)
    l1l0.setParam(GRB.Param.TimeLimit, cutoff)
    l1l0.setParam(GRB.Param.PSDTol, 1e-5)  # Set a larger PSD tolerance to ensure success
    l1l0.setParam(GRB.Param.OutputFlag, 0)
    # Using the default algorithm, and shut gurobi up.
    l1l0.update()
    l1l0.optimize()
    w_opt = np.array([w[v_id].x for v_id in w])
    return w_opt



def l0l2_hierarchy_optimize_quicksum(A, f, correlateID, A_ub = None, f_ub = None,
                                     mu0 = 1e-4, mu2 = 1e-4, M=1000.0, cutoff=300,
                                     if_use_Ewald = True,
                           ):
    """
    Sovler of L0L2-regularization problem using Gurobi.
    Hierarchy constraints can be introduced by correlateID, each entry contains the index of higher ECIs to the current ECI


    Parameters
    ----------
    A: feature matrix as numpy array of shape mxn
    f: scalar property vector as numpy array of shape mx1
    A_ub: feature matrix of unbalanced/fail relaxed, the A_ub @ ECIs > f_ub
    f_ub: scalar property (energy) of boundary condition

    mu0: regularization parameter of L0 term as float
    mu2: regularization parameter of L2 term as float
    M: upper bound of absolute value of ECI used to build relevant constraints, e.g. For all ECIs, |ECI| < M.
    cutoff: cutoff for solving the optimization problem in seconds


    !!!!!!!!
    Check carefully if you are using the Ewald term. The Ewald constraint condition is listed in the middle of this function
    !!!!!!!!

    Returns
    -------
    Fitted ECI values as numpy array of shape A.shape[1]
    """
    n = A.shape[0]
    d = A.shape[1]
    ATA = A.T @ A
    fTA = f.T @ A

    l0l2 = Model()
    w = l0l2.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    z0 = l0l2.addVars(d, vtype=GRB.BINARY)
    z1 = l0l2.addVars(d)

    # Cost function
    L = QuadExpr()

    # Add correlated ID related L0 group type regularization

    for ii in range(len(correlateID)):
        higherList = correlateID[ii]

        for highIdx in higherList:
            l0l2.addConstr(z0[ii] >= z0[highIdx])




    for i in range(d):
        l0l2.addConstr(M * z0[i] >= w[i])
        l0l2.addConstr(M * z0[i] >= (-1.0 * w[i]))


    if not (f_ub is None):
        for i in range(len(f_ub)):
            constrain = quicksum(w[j]*A_ub[i,j] for j in range(d))

            l0l2.addConstr(constrain >= f_ub[i])



    ## dielectric constant constrain##
    if if_use_Ewald:
        l0l2.addConstr( w[d-1] >= 0)
        l0l2.addConstr( w[d-1] <= 1)
        l0l2.addConstr(z0[d-1] >=1 )


    L = L + quicksum(mu0 * z0[i] + mu2 * w[i]*w[i] -2*fTA[i]*w[i] for i in range(d))
    L = L + quicksum(w[i]*w[j]*ATA[i][j] for i in range(d) for j in range(d))


    l0l2.setObjective(L, GRB.MINIMIZE)
    l0l2.setParam(GRB.Param.TimeLimit, cutoff)
    l0l2.setParam(GRB.Param.PSDTol, 1e-6)  # Set a larger PSD tolerance to ensure success
    l0l2.setParam(GRB.Param.OutputFlag, 0)
    # Using the default algorithm, and shut gurobi up.
    l0l2.update()
    l0l2.optimize()
    w_opt = np.array([w[v_id].x for v_id in w])
    z0_opt = np.array([z0[v_id].x for v_id in z0])
    return w_opt, z0_opt
