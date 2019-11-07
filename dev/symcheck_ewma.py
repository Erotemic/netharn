def _symcheck():
    # import sympy as sym
    # # Create the index for our indexable sequence objects
    # i = sym.symbols('i', cls=sym.Idx)

    # measured = list(map(sym.IndexedBase, ['A_m', 'B_m', 'C_m']))
    # smoothed = list(map(sym.IndexedBase, ['A_s', 'B_s', 'C_s']))

    # for S in smoothed:
    #     S[0] = 0

    # for M, S in zip(measured, smoothed):
    #     M[i]
    #     pass

    # for r in raw_values:
    #     z = r[i]
    #     .is_nonnegative = True
    #     r.is_real = True

    ###########
    # I'm not sure how to used index objects correctly, so lets hack it
    import sympy as sym
    kw = dict(is_nonnegative=True, is_real=True)

    measured = {'A_m': [], 'B_m': [], 'C_m': [], 'T_m': []}
    smoothed = {'A_s': [], 'B_s': [], 'C_s': [], 'T_s': []}

    for i in range(3):
        _syms = sym.symbols('A_m{i}, B_m{i}, C_m{i}, T_m{i}'.format(i=i), **kw)
        for _sym in _syms:
            prefix = _sym.name[0:-1]
            measured[prefix].append(_sym)

        _syms = sym.symbols('A_s{i}, B_s{i}, C_s{i}, T_s{i}'.format(i=i), **kw)
        for _sym in _syms:
            prefix = _sym.name[0:-1]
            smoothed[prefix].append(_sym)

    # alpha, beta = sym.symbols('α,  β', is_real=1, is_positive=True)
    # constraints.append(sym.Eq(beta, 1 - alpha))
    # constraints.append(sym.Le(alpha, 1))
    # constraints.append(sym.Ge(alpha, 0))

    constraints = []
    N = 2
    alpha = 0.6
    beta = 1 - 0.6

    for i in range(N):
        # Total measure constraints
        constr = sym.Eq(measured['T_m'][i], measured['A_m'][i] + measured['B_m'][i] + measured['C_m'][i])

        # EWMA constraints
        for s_key in smoothed.keys():
            m_key = s_key.replace('_s', '_m')
            S = smoothed[s_key]
            M = measured[m_key]
            if i == 0:
                constr = sym.Eq(S[i], M[i])
                constraints.append(constr)
            else:
                constr = sym.Eq(S[i], M[i] * alpha + beta * S[i - 1])
                constraints.append(constr)

        constr = sym.Eq(measured['T_m'][i], measured['A_m'][i] + measured['B_m'][i] + measured['C_m'][i])
        constraints.append(constr)

    from sympy.solvers.solveset import linear_eq_to_matrix
    from sympy.solvers.solveset import linsolve
    import itertools as it
    import ubelt as ub

    all_symbols = list(ub.flatten(it.chain(measured.values(), smoothed.values())))

    # WE WANT TO KNOW IF THIS IS POSSIBLE
    constraints.append(
        sym.Gt(measured['A_m'][-1], measured['T_m'][-1]),
    )

    A, b = linear_eq_to_matrix(constraints, all_symbols)
    linsolve((A, b))
