import physical as ph
import numpy as np

def rS(M):
    return 2. * ph.G * M / ph.c**2

def omega(M,r):
    return (ph.G * M/r**3)**(1/2)

def rhill(r,q):
    return r * (q/3.)**(1/3)

def mdotEdd(M):
    eps = 0.1
    return 4.*np.pi*ph.G*M/(eps * ph.kappaES*ph.c)

def ptot(rho,T,mu,tau,Teff):
    '''total pressure with effective radiation pressure'''
    pgas = rho * ph.kB/ph.mH/mu * T
    prad = tau/2 * ph.sigB/ph.c*Teff**4
    return pgas+prad

def kappaPW(rho,T):
    # USE THIS ONE
    # For a list of rho,T, this returns a kappa that is most similar to SG/Pan solutions
    Tes = 1.79393e8*rho**(2/5)
    # LIMIT FOR ELECTRON SCATTERING? (inconsistent w current transitions)
    Teslimit = 2.e4
    #Teslimit = 0.
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    condlist = [((T>=Tes)&(T>Teslimit)),((T<Tes) & (T>=Tbf)),((T<Tbf) & (T>=Thyd)),\
        ((T<Thyd)& (T>=Tmol)),((T<Tmol)& (T>=Tds)),((T<Tds) & (T>=Td)),((T<Td) & (T>=Tis)),((T<Tis))]
    funclist = [ph.kappaES, ph.kappabf*rho*T**(-5/2), ph.kappahyd*rho**(1/3)*T**(10), \
        ph.kappamol*rho**(2/3)*T**(3),ph.kappads*rho*T**(-24), ph.kappad*T**(1/2), \
            ph.kappais*T**(-7), ph.kappai*T**(2)]
    return np.piecewise(T,condlist,funclist)

def kappaPWnoES(rho,T):
    # PW opacities with no electron scattering for outer disc
    # For a list of rho,T, this returns a kappa that is most similar to SG/Pan solutions
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    condlist = [((T>Tbf)),((T<Tbf) & (T>=Thyd)),\
        ((T<Thyd)& (T>=Tmol)),((T<Tmol)& (T>=Tds)),((T<Tds) & (T>=Td)),((T<Td) & (T>=Tis)),((T<Tis))]
    funclist = [ ph.kappabf*rho*T**(-5/2), ph.kappahyd*rho**(1/3)*T**(10), \
        ph.kappamol*rho**(2/3)*T**(3),ph.kappads*rho*T**(-24), ph.kappad*T**(1/2), \
            ph.kappais*T**(-7), ph.kappai*T**(2)]
    return np.piecewise(T,condlist,funclist)
  

def kappaZhu(rho,T,tau,Teff):
    # from Zhu et al 2009
    # assuming constant mu
    # includes updates from pressure dependence of dust opacities
    # currently doesn't reproduce ES kappa in inner region 
    logP = np.log10(ptot(rho,T,0.62,tau,Teff))
    logT = np.log10(T)
    logTbf = 0.28*logP + 3.69
    logThyd = 0.04*logP + 3.91
    logTmol2 = 0.015*logP + 3.7
    logTmol = 0.00832*logP + 3.41
    logTwv = 0.03*logP + 3.28
    logTds = 0.0281*logP + 3.19
    logTd = 0.03*logP + 3.12
    condlist = [(logT<2.9),((logT<logTd)&(logT>=2.9)),((logT<logTds)&(logT>=logTd)),\
        ((logT<logTwv)&(logT>=logTds)),\
        ((logT<logTmol)&(logT>=logTwv)),((logT<logTmol2)&(logT>=logTmol)),\
            ((logT<logThyd)&(logT>=logTmol)),((logT<logTbf)&(logT>=logThyd)),(logT>=logTbf)]
    funclist = [0.738*logT-1.277, 0.738*logT - 1.277, -42.98*logT+1.312*logP+135.1, \
        4.063*logT-15.013,\
        -18.48*logT+0.676*logP+58.93, 2.905*logT+0.498*logP-13.995, \
            10.19*logT+0.382*logP-40.936, -3.36*logT+0.928*logP+12.026, -0.48  ]
    logk = np.piecewise(T,condlist,funclist)
    # one more conditions for boundary: 
    if ((logT<4)&(logk<3.586*logT-16.85)): logk = 3.586*logT-16.85
    return 10**logk


def kappaPWrev(rho,T):
    # For a list of rho,T, this returns a kappa that is most similar to SG/Pan solutions
    Tes = 1.79393e8*rho**(2/5)
    #Teslimit = 2.e4
    Teslimit = 0.
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    condlist = [((T<Tis)),((T<Td) & (T>=Tis)),((T<Tds) & (T>=Td)),((T<Tmol)& (T>=Tds)),\
        ((T<Thyd)& (T>=Tmol)),((T<Tbf) & (T>=Thyd)),((T<Tes) & (T>=Tbf)),((T>=Tes) & (T>=Teslimit))]
    funclist = [ph.kappai*T**(2),ph.kappais*T**(-7),ph.kappad*T**(1/2),ph.kappads*rho*T**(-24),\
        ph.kappamol*rho**(2/3)*T**(3),ph.kappahyd*rho**(1/3)*T**(10),ph.kappabf*rho*T**(-5/2),ph.kappaES]
    return np.piecewise(T,condlist,funclist)
  


def kappaBL(rho,T):
    # Tes : transition to ES from bf+ff
    Tes = 1.79393e8*rho**(2/5)
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    return (
    (T>Tes)               *   ( ph.kappaES )     +
    ((T<=Tes) & (T>Tbf) ) *   ( ph.kappabf * rho * T**(-5/2)       )  +
    ((T<=Tbf) & (T>Thyd)) *   ( ph.kappahyd * rho**(1/3) * T**(10) )  +
    ((T<=Thyd)& (T>Tmol)) *   ( ph.kappamol * rho**(2/3) * T**(3)  )  +
    ((T<=Tmol)& (T>Tds) ) *   ( ph.kappads * rho * T**(-24)        )  +
    ((T<=Tds) & (T>Td)  ) *   ( ph.kappad * T**(1/2)               )  +
    ((T<=Td) & (T>Tis)  ) *   ( ph.kappais * T**(-7)               )  +
    ((T<=Tis)           ) *   ( ph.kappai * T**(2)                 ) 
    )



def kappaBLbf(rho,T):
    # BL opacities as boolean sum of inverse
    # Combine ES and BF/ff 
    Tes = 1.79393e8*rho**(2/5)
    Teslimit = 2.e4
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    return 1./(
    ((T<=Tis)           ) *   1./( 2.e-4 * T**(2)                 )  + 
    ((T<=Td)  & (T>Tis) ) *   1./( 2.e16 * T**(-7)               )  +
    ((T<=Tds) & (T>Td)  ) *   1./( 0.1 * T**(1/2)                )  +
    ((T<=Tmol)& (T>Tds) ) *   1./( 2.e81 * rho * T**(-24)        )  +
    ((T<=Thyd)& (T>Tmol)) *   1./( 1.e-8 * rho**(2/3) * T**(3)  )  +
    ((T<=Tbf) & (T>Thyd)) *   1./( 1.e-36 * rho**(1/3) * T**(10) )  +
    ((T<=Tes) & (T>Tbf) ) *   1./( 0.348 + 1.5e20 * rho * T**(-5/2)     )  
    )

def kappaPWtest(rho,T):
    # works in solver but does not work with lists
    Tes = max(1.79393e8*rho**(2/5),1.e4)
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    if ((T>=Tes)):          return   ( ph.kappaES )
    elif((T<Tes) & (T>=Tbf) ): return   ( ph.kappabf * rho * T**(-5/2) ) 
    elif ((T<Tbf) & (T>=Thyd)): return   ( ph.kappahyd * rho**(1/3) * T**(10) )
    elif ((T<Thyd)& (T>=Tmol)): return   ( ph.kappamol * rho**(2/3) * T**(3)  )  
    elif ((T<Tmol)& (T>=Tds) ): return   ( ph.kappads * rho * T**(-24)        )
    elif ((T<Tds) & (T>=Td)  ): return   ( ph.kappad * T**(1/2)               )
    elif ((T<Td) & (T>=Tis)  ): return   ( ph.kappais * T**(-7)               )
    elif ((T<Tis)           ): return   ( ph.kappai * T**(2)                 ) 
    else: return 0



   

def kappaFloor(rho,T):
    # Tes : transition to ES from bf+ff
    Tes = 1.79393e8*rho**(2/5)
    Tbf = 31195.2*rho**(4/75)
    Thyd = 1.e4 * rho**(1/21)
    Tmol = 2029.76*rho**(1/81)
    Tds = 2286.77 * rho**(2/49)
    Td = 202.677
    Tis = 166.81
    if (T>=Tes):          return   ( ph.kappaES )
    elif((T<Tes) & (T>=Tbf) ): return   ( ph.kappabf * rho * T**(-5/2) ) 
    elif ((T<Tbf)): return   ( 10**0.76 ) 
    else: return 0


#def kappaPW(rho,T):
#    numpy.piecewise(r, [(r<r1), (r==r1), (r>r1)], [0, 1, 2])
#   return 

def kappaFF(rho,T):
    return ph.kappaES + 4.e25*rho*T**(-7/2)

def kappaRP(rho,T):
    # Tes : transition to ES from bf+ff
    Tes = 1.e4
    Tbf = 3200.
    Thyd = 1.025e4*rho**(2/27)
    Tmol = 2780.*rho**(1/48)
    Tds = 2.e3*rho**(1/50)
    Td = 180.
    Tis = 150
    return (
    (T>Tes)               *   ( ph.kappaES )     +
    ((T<=Tes) & (T>Tbf) ) *   ( 0.645e23 * rho * T**(-7/2)       )  +
    ((T<=Tbf) & (T>Thyd)) *   ( 2.e34 * rho**(2/3) * T**(-9) )  +
    ((T<=Thyd)& (T>Tmol)) *   ( 1.6e-2  )  +
    ((T<=Tmol)& (T>Tds) ) *   ( 1.57e60 * rho**(3/8) * T**(-18)        )  +
    ((T<=Tds) & (T>Td)  ) *   ( 2.e-2 * T**(3/4)              )  +
    ((T<=Tds) & (T>Tis) ) *   ( 1.15e18 * T**(-8)               )  +
    ((T<=Tis)           ) *   ( 2.e-4 * T**(2)                 ) 
    )




