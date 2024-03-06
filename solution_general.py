import numpy as np
import sympy as sp
from scipy.optimize import root
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import physical as ph
import functions as f
import warnings
# This warning is misleading given our units (i.e. when solving the set of equations, differences between large numbers will always produce this):
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


# Choose parameters 
# the MBH mass = Mbh_f x 10^Mbh_exp * Msun
Mbh_f = 1.
Mbh_exp = 6 
# Accretion rate = a fraction fEdd * the Eddington rate
fEdd = 0.1 
# constant value of disk viscosity
alpha = 0.1
# critical Toomre Q, below which the disk is gravitationally unstable
qcrit = 1.4
# When saving files, add a descriptor in the name
#name = 'irr_alb0'
name = 'nofloor'

Mbh = (Mbh_f * 10.**Mbh_exp) * ph.Msun
# rmin is the radius of the marginally stable orbit around the SMBH, which we set to rS/r/radiative efficiency of accretion (Sirko & Goodman 2003)
rS = f.rS(Mbh) 
rmin = rS/4./0.1
mdotEdd = 4.*np.pi*ph.G * Mbh / (ph.kappaES * 0.1 * ph.c)
# mean molecular mass (depends on composition and ionization fraction, but we are simplifying):
mu = 0.62
gamma = 5./3
# for irradiation
albedo = 0

# starting point: 
rinit = 10*rS
# final point:
rmax = .5*ph.pctocm
#rmax = 1000 * rS
# choose data spacing for inner and outer disk solutions 
dr = 1*rS
dr_o = 10*rS
# factor to vary dr_o if solution is not behaving: not utilized for now
#fdr = 1.

print('\n BH mass = ',Mbh_f,'x 10^',Mbh_exp,' Msun\n',\
    'f_Edd = ',fEdd,'\n',\
    'alpha = ',alpha\
    )

def equations_inner(var, *args):
    '''
    System of Eqs for inner disk
    '''
    sigma, T, cs, kappa, tau, rho = var
    rad = args[0]
    mdot = args[1]
    Teff = args[2]
    om = (ph.G * Mbh/rad**3)**(1/2)
    # return eqs that = 0
    eq1 = cs**2 * sigma - mdot*om / 3. / np.pi / alpha
    eq2 = T**4 - (3 * tau/8 + 1/2 + 1/4/tau)*Teff**4
    eq3 = 2*tau - kappa * sigma
    eq4 = kappa - f.kappaPW(rho,T)
    # other options: 
    #eq4 = kappa - ph.kappaES
    # Here the sound speed is defined using an effective radiation pressure 
    eq5 = cs**2 - ph.kB/ph.mH/mu * T - ph.sigB/ph.c/2 *tau*Teff**4/rho    
    # other options: 
    #eq5 = cs**2 -  ph.kB/ph.mH/mu * T - 1./3 * ph.arad*T**4/rho
    eq6 = 2*rho - om* sigma / cs
    #eq7 = h - cs/om
    return ([eq1, eq2, eq3, eq4, eq5, eq6])

def equations_outer(var, *args):
    '''
    System of Eqs for outer disk where Q=Q_critical
    '''
    sigma, T, Teff, cs, kappa, tau = var
    rad = args[0]
    mdot = args[1]
    rho = args[2]
    #cs = args[3]
    #sigma = args[4]
    om = (ph.G * Mbh/rad**3)**(1/2)
    # return eqs that = 0
    eq1 = cs**2 * sigma - mdot*om / 3. / np.pi / alpha
    eq2 = T**4 - (3 * tau/8 + 1/2 + 1/4/tau)*Teff**4
    eq3 = 2*tau - kappa * sigma
    #eq4 = kappa - f.kappaPWnoES(rho,T)
    eq4 = kappa - f.kappaPW(rho,T)
    #eq4 = kappa - ph.kappaES
    # Here the sound speed is defined using an effective radiation pressure
    eq5 = cs**2 - ph.kB/ph.mH/mu * T - ph.sigB/ph.c/2 *tau*Teff**4/rho
    #eq5 = cs**2 -  ph.kB/ph.mH/mu *T - 1./3 * ph.arad*T**4/rho
    #eq6 = 2*rho - sigma / h
    eq6 = 2*rho - om* sigma / cs 
    #eq7 = h - cs/om
    return ([eq1, eq2, eq3, eq4, eq5, eq6])


# First we need to solve for an initial point. Giving out parameteres, the effective (surface) temperature is known
r0 = rinit
om = f.omega(Mbh,r0)
mdot = fEdd * mdotEdd * (1. - np.sqrt(rmin/r0))
Teffprof = (3./np.pi/8. *mdot * om**2/ph.sigB)**(1/4)
# Irradiated case:
# assuming a guess for h/r
#Teffprof = (3./np.pi/8. *mdot * om**2/ph.sigB + (1.-albedo)* eta*mdot*ph.c**2/(4.*np.pi * ph.sigB * r0**2) *(0.01))**(1/4)


# ------ Solution for initial guess ------
# We'll use sympy to solve analytically a simpler set of eqs
# Define symbols
sigma_s, Tmid_s, cs_s, kappa_s, tau_s, rho_s, h_s \
    = sp.symbols('sigma_s, Tmid_s, cs_s, kappa_s, tau_s, rho_s, h_s',\
        positive=True)

# Inner Eqs for initial solution point 
# assume optically thick, radiation pressure, ES opacity. 
# take caution for lower MBH masses, where this guess is less appropriate
eq1i = sp.Eq(cs_s**2 * sigma_s - mdot*om/3/np.pi/alpha,0)
eq2i = sp.Eq(Tmid_s**4 - (3./8 * tau_s)*Teffprof**4,0)
eq3i = sp.Eq(2*tau_s - kappa_s*sigma_s,0)
eq4i = sp.Eq(kappa_s - ph.kappaES,0)
#eq4i = sp.Eq(kappa_s - ph.kappabf*rho_s*Tmid_s**(-5./2),0)  # this hangs at first guess
eq5i = sp.Eq(cs_s**2 - (1./3 * ph.arad)/rho_s*Tmid_s**4,0)
eq6i = sp.Eq(2*rho_s*h_s - sigma_s,0)
eq7i = sp.Eq(cs_s - om*h_s,0)
system_init = [eq1i,eq2i,eq3i,eq4i,eq5i,eq6i,eq7i]

soln = sp.solvers.solve(system_init, \
    (sigma_s, Tmid_s, cs_s, kappa_s, tau_s, rho_s, h_s),\
        verify=False)

# Need to convert sympy matrix output to a float type suitable for scipy
guesses = np.array(soln).astype(np.float64)
# take out h: it is actually a redundant variable here, but I will modify later
guessesT = np.delete(guesses,[6])
# Now start with numerical solution, given the initial guess:
froot = root(equations_inner, args=(r0,mdot,Teffprof), x0=guessesT)

radius = np.array([r0])
om_arr = np.array([om])
sigman = np.array([froot.x[0]])
Tmidn = np.array([froot.x[1]])
csn = np.array([froot.x[2]])
kappan = np.array([froot.x[3]])
taun = np.array([froot.x[4]])
rhon = np.array([froot.x[5]])
#hn = np.array([froot.x[6]])

# check indexing above! 
Teffn = np.array([Teffprof])
hn = csn/om
# other interesting quantities:
pgasn = rhon * ph.kB/ph.mH/mu * Tmidn
pradn = taun/2 * ph.sigB/ph.c*Teffn**4
toomreQ = csn * om / (np.pi * ph.G * sigman)
tcool = 1./(gamma-1.)*3./16 * sigman*csn**2 / (ph.sigB * Tmidn**4) \
    * (taun + 1./taun)
alphan = alpha


# - - - - - - - Now solve over radii - - - - - - - - -
i=1
r0 += (dr)    
om = f.omega(Mbh,r0)
mdot = fEdd * mdotEdd * (1. - np.sqrt(rmin/r0))
Teffprof = (3./np.pi/8. *mdot * om**2/ph.sigB)**(1/4)
#Teffprof = (3./np.pi/8. *mdot*om**2/ph.sigB + (1.-albedo)*eta*mdot*ph.c**2/(4.*np.pi*ph.sigB*r0**2)*(csn/om/r0))**(1/4)
# - - - - - INNER DISK TILL Q = QCRIT - - - - - - 
print('solving inner disk...')
while ((toomreQ[i-1]>=qcrit) and (r0 < rmax)):
    # guesses for: sigma, T, cs, kappa, tau, rho
    # take solutions from last point
    guesses = np.array([sigman[i-1], Tmidn[i-1], \
        csn[i-1], kappan[i-1], taun[i-1], rhon[i-1]])
    froot = root(equations_inner, args=(r0,mdot,Teffprof), x0=guesses)

    # check change in Tmid before appending
    #if abs(froot.x[1]-Tmidn[i-1])/Tmidn[i-1]>.9: 
    #    print('deviating: dr_o = ',dr_o)
        # go back and reduce dr
    #    r0 -= (fdr*dr)
    #    fdr = 0.5*fdr
    #    if fdr < 1.e-10:
    #        print('UNCONVERGED at r = ', r0/ph.pctocm,' pc')
    #        r0 = rmax
    #else: 
    # save solution:
    radius = np.append(radius,r0)
    om_arr = np.append(om_arr,om)
    sigman = np.append(sigman,froot.x[0])
    Tmidn = np.append(Tmidn,froot.x[1])
    csn = np.append(csn,froot.x[2])
    kappan = np.append(kappan,froot.x[3])
    taun = np.append(taun,froot.x[4])
    rhon = np.append(rhon,froot.x[5])

    Teffn = np.append(Teffn,(3./np.pi/8. *mdot * om**2/ph.sigB)**(1/4))
    # irradiated case: 
    #Teffn = np.append(Teffn,(3./np.pi/8. *mdot*om**2/ph.sigB + (1.-albedo)*eta*mdot*ph.c**2/(4.*np.pi*ph.sigB*r0**2)*(csn[i]/om/r0))**(1/4))

    hn = np.append(hn,csn[i]/om)
    pgasn = np.append(pgasn,rhon[i] * ph.kB/ph.mH/mu * Tmidn[i])
    pradn = np.append(pradn,taun[i]/2 * ph.sigB/ph.c*Teffn[i]**4) # This is the effective radiation pressure
    # Calculate Toomre Q to determine if inner or outer disk
    toomreQ = np.append(toomreQ,csn[i] * om / (np.pi * ph.G * sigman[i]))
    tcool = np.append(tcool,1./(gamma-1.)*3./16 * sigman[i]*csn[i]**2 / (ph.sigB * Tmidn[i]**4) \
        * (taun[i] + 1./taun[i]))
    alphan = np.append(alphan,alpha)
    r0 += (dr)    
    om = f.omega(Mbh,r0)
    mdot = fEdd * mdotEdd * (1. - np.sqrt(rmin/r0))
    Teffprof = (3./np.pi/8. *mdot * om**2/ph.sigB)**(1/4)
    # irradiated case:
    #Teffprof = (3./np.pi/8. *mdot*om**2/ph.sigB + (1.-albedo)*eta*mdot*ph.c**2/(4.*np.pi*ph.sigB*r0**2)*(csn[i]/om/r0))**(1/4)
    i+=1

rtransition = r0
indtransition = len(radius)-1
cscrit = csn[i-1]

# - - - - - - - OUTER DISK FIXED Q = QCRIT - - - - - - - 
print('solving outer disk... ' ,rtransition/f.rS(Mbh),' rS')
while (r0 < rmax): 
    print('transition to gravitationally unstable: r = ',r0/ph.pctocm,' pc')
    mdot = fEdd * mdotEdd * (1. - np.sqrt(rmin/r0))
    # guesses for: sigma, T, Teff, cs, kappa, tau, rho
    # rho, cs, sigma are now constrained by qcrit
    rhocrit = om**2 / 2./np.pi/ph.G/qcrit
    cscrit = (ph.G * qcrit * mdot/3./alpha)**(1/3)
    #sigcrit = mdot*om / cscrit**2 / 3. / np.pi / alpha
    sigcrit = 2*rhocrit*cscrit/om

    guesses = np.array([sigman[i-1], Tmidn[i-1],Teffn[i-1],csn[i-1], kappan[i-1],taun[i-1]])
    froot = root(equations_outer, args=(r0,mdot,rhocrit), x0=guesses)

    # sometimes there were NaNs, but we took care of it
    #nanind = np.isnan(froot.x[:])
    #froot.x[nanind] = 0.

    # check Q solution before appending
    #if abs(((cscrit*om/(np.pi*ph.G*sigcrit)-qcrit)/qcrit>0.1)): 
    #    print('deviating ')
    #    i-=1
    #    fdr = 0.5*fdr
    #    if fdr < 1.e-5:
    #        print('UNCONVERGED at r = ', r0/ph.pctocm,' pc')
    #        r0 = rmax
    #else: 
    #fdr = 1.
    # save solution
    radius = np.append(radius,r0)
    om_arr = np.append(om_arr,om)
    sigman = np.append(sigman,froot.x[0])
    Tmidn = np.append(Tmidn,froot.x[1])
    Teffn = np.append(Teffn,froot.x[2])
    csn = np.append(csn,froot.x[3])
    kappan = np.append(kappan,froot.x[4])
    taun = np.append(taun,froot.x[5])

    # if putting this back then check the above indexing!
    hn = np.append(hn,cscrit/om)
    rhon = np.append(rhon,rhocrit)
    #csn = np.append(csn,cscrit)
    #sigman = np.append(sigman,sigcrit)
    #Teffn = np.append(Teffn,Tmidn[i]*(3*taun[i]/8+1/2+1/4/taun[i])**(-1/4))
    pgasn = np.append(pgasn,rhon[i] * ph.kB/ph.mH/mu * Tmidn[i])
    pradn = np.append(pradn,taun[i]/2 * ph.sigB/ph.c*Teffn[i]**4) 
    toomreQ = np.append(toomreQ,csn[i]*om/(np.pi*ph.G*sigman[i]))
    #toomreQ = np.append(toomreQ,qcrit)
    tcool = np.append(tcool,1./(gamma-1.)*3./16 \
        * sigman[i]*csn[i]**2 / (ph.sigB * Tmidn[i]**4) \
        * (taun[i] + 1./taun[i]))
    alphan = np.append(alphan,1./(om*tcool[i]))
  
    r0 += (fdr*dr_o)
    om = f.omega(Mbh,r0) 
    i+=1



# --------- P l o t -----------
num=5

plt.figure(figsize=(12,7.5))
plt.subplot(341)
plt.scatter(radius[::num]/ph.pctocm,sigman[::num],marker='.')
#plt.plot(r_M8/ph.pctocm,sigma_M8,color='grey')
#plt.plot(r_M6/ph.pctocm,sigma_M6,color='darkgrey')
#plt.scatter(rtransition/ph.pctocm,sigman[indtransition],marker='x',color='r')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\Sigma$')

plt.subplot(342)
plt.scatter(radius[::num]/ph.pctocm,Tmidn[::num],marker='.')
#plt.plot(r_M8/ph.pctocm,Tmid_M8,color='grey')
#plt.scatter(rtransition/ph.pctocm,Tmidn[indtransition],marker='x',color='r')
#plt.plot(r_M6/ph.pctocm,Tmid_M6,color='darkgrey')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$T_{\rm mid}$')

plt.subplot(343)
plt.scatter(radius[::num]/ph.pctocm,Teffn[::num],marker='.')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$T_{\rm eff}$')

plt.subplot(344)
plt.scatter(radius[::num]/ph.pctocm,kappan[::num],marker='.')
#plt.scatter(rtransition/ph.pctocm,kappan[indtransition],marker='x',color='r')
#plt.plot(r_M8/ph.pctocm,kappa_M8,color='grey')
#plt.plot(r_M6/ph.pctocm,2*tau_M6/sigma_M6,color='darkgrey')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\kappa$')

plt.subplot(345)
plt.scatter(radius[::num]/ph.pctocm,taun[::num],marker='.')
#plt.scatter(rtransition/ph.pctocm,taun[indtransition],marker='x',color='r')
#plt.plot(r_M8/ph.pctocm,tau_M8,color='grey')
#plt.plot(r_M6/ph.pctocm,tau_M6,color='darkgrey')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\tau$')

plt.subplot(346)
plt.scatter(radius[::num]/ph.pctocm,csn[::num],marker='.')
#plt.plot(r_M8/ph.pctocm,cs_M8,color='grey')
#plt.scatter(rtransition/ph.pctocm,csn[indtransition],marker='x',color='r')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$c_s$')

plt.subplot(347)
plt.scatter(radius[::num]/ph.pctocm,rhon[::num],marker='.')
#plt.scatter(rtransition/ph.pctocm,rhon[indtransition],marker='x',color='r')
#plt.plot(r_M8/ph.pctocm,rho_M8,color='grey')
#plt.plot(r_M6/ph.pctocm,rho_M6,color='darkgrey')
#plt.xlim([1.e3,5.e5])
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\rho$')

plt.subplot(348)
plt.scatter(radius[::num]/ph.pctocm,hn[::num]/radius[::num],marker='.')
#plt.scatter(radius/ph.pctocm,csn/omega/radius,marker='.')
#plt.plot(r_M8/ph.pctocm,hor_M8,color='grey')
#plt.plot(r_M6/ph.pctocm,hor_M6,color='darkgrey')
#plt.xlim([1.e3,5.e5])
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$h/r$')


plt.subplot(349)
plt.scatter(radius[::num]/ph.pctocm,radius[::num]/hn[::num],marker='.')
#plt.scatter(rtransition/ph.pctocm,toomreQ[indtransition],marker='x',color='r')
plt.axhline(qcrit,-1,1,linestyle='--')
#plt.xlim([1.e3,5.e5])
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathcal{M}$')


plt.subplot(3,4,10)
plt.scatter(radius[::num]/ph.pctocm,om_arr[::num]*tcool[::num],marker='.')
#plt.xlim([1.e3,5.e5])
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\Omega t_{\rm cool}$')

plt.subplot(3,4,11)
plt.scatter(radius[::num]/ph.pctocm,pgasn[::num]/(pgasn[::num]+pradn[::num]),marker='.')
#plt.xlim([1.e3,5.e5])
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\beta$')

plt.subplot(3,4,12)
plt.scatter(radius[::num]/ph.pctocm,toomreQ[::num],marker='.')
plt.scatter(rtransition/ph.pctocm,toomreQ[indtransition],marker='x',color='r')
plt.axhline(qcrit,-1,1,linestyle='--')
#plt.xlim([1.e3,5.e5])
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$Q$')

plt.tight_layout()
plt.show()


# - - - - - ask to save solution to file - - - - - 
var = input("Save solution? (type 'y'):  ")
print("You entered: " + var)

# - - - - - - - - save data - - - - - - - 
if (var=='y'):
    solutionarray = np.array([radius,om_arr,sigman,Tmidn,Teffn,csn,kappan,taun,rhon,hn,\
        toomreQ,tcool]).T
    savedir = "solutions/M%.1fe%i_a%.2f_f%.2f_%s.txt"%(Mbh_f,Mbh_exp,alpha,fEdd,name)
    np.savetxt(savedir,\
        solutionarray,'%.5e',delimiter='   ',\
        header=' r [cm]   omega   sigma   Tmid   Teff   cs   kappa   tau   rho   h [cm]   Q   tcool') 
    print('DONE: output saved in ',savedir)
else:
    print('DONE: output not saved')