"""
Tools for choosing poles for closed loop control systems
"""
import cmath


def bessel_spoles(n, Ts=1):
    """ Return the roots of the reverse Bessel polynomial normalized the given
    settling time. The settling time is 1 second by default. Adapted from 
    Digital Control: A State-Space Approach, Table 6.3.
    
    Args:
        n: The order of the Bessel polynomial.
        Ts (optional): The settling time to scale to.

    Returns:
        list: The roots of the Bessel polynomial.
    """

    spoles = [0]
    
    if n == 1:
        spoles = [-4.6200 + 0j]
    elif n == 2:
        spoles = [-4.0530 + 2.3400j, -4.0530 - 2.3400j]
    elif n == 3:
        spoles = [-5.0093 + 0j,
                  -3.9668 + 3.7845j, -3.9668 - 3.7845j]
    elif n == 4:
        spoles = [-4.0156 + 5.0723j, -4.0156 - 5.0723j,
                  -5.5281 + 1.6553j, -5.5281 - 1.6553j]
    elif n == 5:
        spoles = [-6.4480 + 0j,
                  -4.1104 + 6.3142j, -4.1104 - 6.3142j,
                  -5.9268 + 3.0813j, -5.9268 - 3.0813j]
    elif n == 6:
        spoles = [-4.2169 + 7.5300j, -4.2169 - 7.5300j,
                  -6.2613 + 4.4018j, -6.2613 - 4.4018j,
                  -7.1205 + 1.4540j, -7.1205 - 1.4540j]
    elif n == 7:
        spoles = [-8.0271 + 0j,
                  -4.3361 + 8.7519j, -4.3361 - 8.7519j,
                  -6.5714 + 5.6786j  -6.5714 - 5.6786j,
                  -7.6824 + 2.8081j  -7.6824 - 2.8081j]
    elif n == 8:
        spoles = [-4.4554 + 9.9715j, -4.4554 - 9.9715j,
                  -6.8554 + 6.9278j, -6.8554 - 6.9278j,
                  -8.1682 + 4.1057j, -8.1682 - 4.1057j,
                  -8.7693 + 1.3616j, -8.7693 - 1.3616j]
    elif n == 9:
        spoles = [-9.6585 + 0j,
                  -4.5696 + 11.1838j, -4.5696 - 11.1838j,
                  -7.1145 +  8.1557j, -7.1145 -  8.1557j,
                  -8.5962 +  5.3655j, -8.5962 -  5.3655j,
                  -9.4013 +  2.6655j, -9.4013 -  2.6655j]
    elif n == 10:
        spoles = [-4.6835 + 12.4022j, -4.6835 - 12.4022j,
                  -7.3609 +  9.3777j, -7.3609 -  9.3777j,
                  -8.9898 +  6.6057j, -8.9898 -  6.6057j,
                  -9.9657 +  3.9342j, -9.9657 -  3.9342j,
                  -10.4278 + 1.3071j, -10.4278 - 1.3071j]

    return [ spole/Ts for spole in spoles ]


def spoles_to_zpoles(spoles, T):
    """ Convert the continouous s-plane poles to the discrete z-plane poles
    with the given sampling interval T.
    
    Args:
        spoles (list): The s-plane poles
        T: The sampling interval in seconds.
    """

    return [ cmath.exp(spole * T) for spole in spoles ]

import numpy as np
from numpy import linalg as LA

def fbg(A, B, p, q=None):
    """
    %FBG	Feedback gain matrices.
    %
    % FUNCTION CALLS (2):
    %
    % (1) L = fbg(A,B,p);
    %
    % This calculates a matrix L such that the eigenvalues of A-B*L are
    % those specified in the vector p.  Any complex eigenvalues in the 
    % vector p must appear in consecutive complex-conjugate pairs.
    % The numbers in the vector p must be distinct (see below for repeated roots).
    %
    % (2) L = fbg(A,B,p,q);
    %
    % This form allows the user to specify repeated eigenvalues by indicating
    % the desired multiplicity in the vector q.  For example, to have a single
    % eigenvalue at -2, another at -3 with multiplicity 3, and  complex conjugate 
    % eigenvalues at -1+j, -1-j with multiplicity 2,  set p=[-2 -3 -1+j -1-j], 
    % and q=[1 3 2 2].  The multiplicity of any eigenvalue cannot be greater
    % than the number of columns of B.

    """

    if q is None:
        q = np.ones((1, len(p)))
    (n, m) = B.shape
    I = np.eye(n)
    npp = len(p)
    cvv = [n.imag==0 for n in p ]
    npoles = int(npp - sum([ not n for n in cvv])/2)

    for i in range(0, npp):
        if i < npoles:
            if not cvv[i]:
                cvv[i+1]=[]
                p[i+1]=[]
                q[i+1]=[]

    if n <= m:
        d1=[]
        d2=[]
        for i in range(0, npoles):
            if cvv[i]:
                d1.extend([p[i]])
                if i < npoles:
                    d2.extend([0])
            else:
                d1.extend([p[i].real, p[i].real])
                if i < np:
                    d2.extend([p[i].imag, 0])
                else:
                    d2.extend([p[i].imag])
        if n > 1:
            L = LA.lstsq(B, A - (np.diag(d1) + np.diag(d2, 1) - np.diag(d2, -1)))[0]
        else:
            L = LA.lstsq(B, A - d1)[0]
        return L

    sq = sum(q)
    AT=[]
    ATT=[]
    X=[]
    X1=[]
    Y=[]
    Y1=[]
    cv=[]
    Xb=[]
    for i in range(0, npoles):
        print(cvv[i])
        cv.extend([cvv[i] * n for n in [1]*q[i]])
    print(cv)
    for i in range(0, npoles):
        T = np.null(np.concatenate((p[i]*I-A, B), axis=1))
        TT = np.orth(T[1:n,:])
        AT = np.concatenate((AT, T), axis=1)
        ATT = np.concatenate((ATT, TT), axis=1)
        X[:, i] = TT[:, 1:q[i]]
        if q[i] == 1 and i > 1:
            cvt = cv[1:i]
            In = find(cvt==0)
            c = cond(np.concatenate((X, np.conj(X[:,In])), axis=1))
            for j in range(1, m+1):
                Y = np.concatenate((X[:,1:i-1], TT[:,j]), axis=1)
                cc = np.cond(np.concatenate((Y, np.conj(Y[:,In])), axis=1))
                if cc < c:
                    c = cc
                    X[:, i] = TT[:, j]

    Xt = X
    cd = 1.e15

    if m == n:  # can calculate L to get orthogonal eigenvectors
        Ab = np.zeros((n, n))
        for i in range(0, npoles):
            Ab[i, i] = p[i]
            if not cv[i]:
                Ab[i, i+1] = -p[i].imag
                Ab[i+1, i] = p[i].imag
                Ab[i+1, i+1] = p[i].real

        L = LA.lstsq(B, (A-Ab))[0]
        return L

    if m > 1:
        for k in range(5):
            X2 = []
            kk = 0
            for i in range(0, npoles):
                Pr = ATT[:,(i-1)*m+1:i*m]
                Pr = Pr * np.transpose(Pr)
                for j in range(0, q[i]):
                    kk = kk + 1
                    S = np.concatenate((Xt[:,1:kk-1], Xt[:,kk+1:sq]), axis=1)
                    S = np.concatenate((S, np.conj(S)), axis=1)
                    if not cv[kk]:
                        S = np.concatenate((S, np.conj(Xt[:, kk])), axis=1)
                    (Us, Ss, Vs) = np.svd(S)
                    Xt[:,kk] = Pr * Us[:,n]
                    Xt[:,kk] = Xt[:, kk] / np.norm(Xt[:, kk])
                    if not cv(kk):
                        X2 = np.concatenate((X2, np.conj(Xt[:,kk])), axis=1)

            c = np.cond(np.concatenate((Xt, X2), axis=1));
            if c < cd:
                Xtf = Xt
                cd = c
    else:
        Xtf = X

    kkk = 0
    X1=[]
    X2=[]
    for i in range(0, npoles):
        for j in range(0, q[i]):
            kkk = kkk + 1
            if cv[kkk]:
                x = Xtf[:,kkk].real
                y = Xtf[:,kkk].imag
                if np.norm(x) > np.norm(y):
                    Xtf[:,kkk] = x/np.norm(x)
                else:
                    Xtf[:,kkk] = y/np.norm(y)
            a = LA.lstsq(AT[1:n, i*m+1:(i+1)*m], Xtf[:,kkk])[0]
            t = AT[n+1:n+m, i*m+1:(i+1)*m] * a
            x = t.imag
            Xb = np.concatenate((Xb, t.real), axis=1)
            if not cv[kkk]:
                X2 = np.concatenate((X2, x), axis=1)
                X1 = np.concatenate((X1, Xtf[:,kkk].imag), axis=1)
                Xtf[:,kkk] = Xtf[:,kkk].real

    L = np.concatenate((Xb, X2), axis=1)/np.concatenate((Xtf, X1), axis=1)
    return L
