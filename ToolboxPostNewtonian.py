# Copyright (C) 2023
#   Author: Javier M. Antelis <mauricio.antelis@gmail.com>
#   Revision: 1.01
#   Date: 2023/08/04 14:11:37
#   



# -------------------------------
# Import packages
import numpy as np
import matplotlib.pyplot as plt



# -------------------------------
# Define constants


# Gravitational constant
G              = 6.67430e-11       # m^3 kg^−1 s^−2

# Velocity of light (m/s)
c              = 2.99792458e8      # m/s
c2             = np.power(c,2)     # c^2
c3             = np.power(c,3)     # c^3
c4             = np.power(c,4)     # c^4

# Sun mass
msun           = 1.9885e+30        # kg
esun           = msun*c**2         # J

# Conversiones de distance
pc2m           = 3.08567758128e+16 # m
kpc2m          = 3.08567758128e+19 # m
Mpc2m          = 3.08567758128e+22 # m

pc2cm          = 3.08567758128e+18 # cm
kpc2cm         = 3.08567758128e+21 # cm
Mpc2cm         = 3.08567758128e+24 # cm

m2pc           = 3.240779289666357e-17
m2kpc          = 3.240779289666357e-20
m2Mpc          = 3.240779289666357e-23

cm2pc          = 3.240779289666357e-19
cm2kpc         = 3.240779289666357e-22
cm2Mpc         = 3.240779289666357e-25



# -------------------------------
# Plotting settings
plt.rcParams.update(plt.rcParamsDefault)
size=17
# 'legend.fontsize': 'large'
params = {'figure.figsize': (8,5),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'legend.fontsize': size*0.8,
          'axes.titlepad': 7,
          'font.family': 'serif',
          'font.weight': 'medium',
          'xtick.major.size': 10,
          'ytick.major.size': 10,
          'xtick.minor.size': 5,
          'ytick.minor.size': 5
          }
plt.rcParams.update(params)



# -------------------------------
# Compute physical information
def GWphysicalinformation(DMpc, m1Mo, m2Mo, flow, fs, PNorder, doprint):
   
    # -------------------------------
    # Convert distance from Mpc to m
    D              = DMpc * Mpc2m      # m

    # -------------------------------
    # Convert masses to units of kg
    m1             = m1Mo*msun         # kg
    m2             = m2Mo*msun         # kg

    # -------------------------------
    # Total mass of the two companions
    Mtotal         = m1 + m2           # kg

    # Reduced mass ($\mu$)
    mu             = (m1*m2) / Mtotal  # kg

    # -------------------------------
    # Ratio of reduced mass and total mass or symmetric mass ratio  ($\nu$)
    nu            = mu/Mtotal         # dimenssionless

    # -------------------------------
    # Chirp mass
    Mchirp         = np.power(mu,3/5)*np.power(Mtotal,2/5) # kg (or np.power(nu,3/5)*Mtotal)

    # -------------------------------
    # Chirp maximum frequency
    fisco          = c3/(6*np.sqrt(6)*np.pi*G*Mtotal) # Hz

    # -------------------------------
    # Number of cycles in the chip signal
    Ncyc           = 1/(32*np.power(np.pi,8/3)) * np.power(G*Mchirp/c3,-5/3) * (np.power(flow,-5/3)-np.power(fisco,-5/3)) # dimenssionless

    # -------------------------------
    # Compute Tchirp
    Tchirp, Nsamples = GWchirpduration(PNorder, nu, Mtotal, flow, fs, False) # seconds and samples

    # -------------------------------
    # Create time vector
    t              = np.linspace( 0, Tchirp-(1/fs), Nsamples )   # From 0 to Tchirp

    # Coalescence time $t_{coal}$
    t_coal         = t[-1] + (1/fs) # at the end time (one sample after to avoid "divide by zero encountered in power negative")

    # Coalescence phase  $\phi_{coal}$
    phi_coal       = 0        # rad
    
    # Plot
    if doprint:
        print("Distance:                        {0:1.2e} m".format(D) )
        
        print("Mass 1:                          {0:1.2e} kg".format(m1) )
        print("Mass 2:                          {0:1.2e} kg".format(m2) )
        print("Total mass:                      {0:1.2e} kg".format(Mtotal) )
        print("Reduced mass:                    {0:1.2e} kg".format(mu) )
        print("Symmetric mass ratio:            {0:1.2f}".format(nu) )
        print("Chirp mass:                      {0:1.2e} kg".format(Mchirp) )

        print("Frequency isco:                  {0:1.2f} Hz".format(fisco) )
        print("Number of cycles:                {0:1.2f}".format(Ncyc) )
        print("Duration of the GW signal:       {0:1.2f} s".format(Tchirp) )
        print("Number of samples in the signal: {0:d}".format(Nsamples) )

        #print("Coalescence time (s):                  {0:1.2f}".format(t_coal) )
        #print("Coalescence phase (rad):               {0:1.2f}".format(phi_coal) )
      
    # Return
    return D, m1, m2, Mtotal, mu, nu, Mchirp, fisco, Ncyc, Tchirp, Nsamples, t, t_coal, phi_coal



# -------------------------------
# Compute chirp duration "Tchirp"
def GWchirpduration(PNorder, nu, Mtotal, flow, fs, doprint):
    """
    Check https://arxiv.org/pdf/0903.0338.pdf

    Chirp times: PN contributions at different orders to the duration of a 
    signal starting from a time when the instantaneous GW frequency has a 
    fiducial value fL to a time when the GW frequency formally diverges and 
    system coalesces
    """
    
    # -------------------------------
    # Compute tau0 and tau1
    xlow        = np.power( (G*Mtotal/c3)*(np.pi*flow) , 1/3)
    tau0        = ( 5/(256*np.pi*nu*flow)) * np.power(xlow,-5)
    tau1        = ( 1/(8*nu*flow) ) * np.power(xlow,-2)
    #print("xlow                                   {0:1.2f}".format(xlow ) )
    #print("tau0                                   {0:1.2f}".format(tau0 ) )
    #print("tau1                                   {0:1.2f}".format(tau1 ) )
    
    # -------------------------------
    # Compute these terms
    gammaE      = 0.577216
    xlow_8      = np.power(xlow, -8)
    xlow_6      = (743/252 + (11/3)*nu) * np.power(xlow,-6)
    xlow_5      = - (32*np.pi/5) * np.power(xlow, -5)
    xlow_4      = ( 3058673/508032 + (5429/504)*nu + (617/72)*nu**2 ) * np.power(xlow,-4)
    xlow_3      = ( (13/3)*nu - (7729/252) ) * np.pi * np.power(xlow,-3)
    xlow_2      = ( \
            ( (6848*gammaE/105)-(10052469856691/23471078400)+(128*(np.pi**2)/3) ) \
        +   ( ( (3147553127/3048192) - (451*(np.pi**2)/12) ) *nu ) \
        -   ( (15211/1728) *nu**2 ) \
        +   ( (25565/1296) *np.power(nu,3) ) \
        +   ( (6848/105) *np.log(4*xlow) ) \
        ) * np.power(xlow,-2)
    xlow_1      = ( (14809/378)*nu**2 - (75703/756)*nu - 15419335/127008 ) * np.pi * np.power(xlow,-1)
    
    # -------------------------------
    # Calculate Tchirp according to the PN order
    if   PNorder==0.0: Tchirp = tau0
    elif PNorder==1.0: Tchirp = tau0+tau1
    elif PNorder==1.5: Tchirp = tau0+tau1 # PILAS: this need to be confirmed
    elif PNorder==2.0: Tchirp = (5/(256*nu)) * (G*Mtotal/c3) * ( xlow_8 + xlow_6 + xlow_5 + xlow_4 )
    elif PNorder==2.5: Tchirp = (5/(256*nu)) * (G*Mtotal/c3) * ( xlow_8 + xlow_6 + xlow_5 + xlow_4 + xlow_3 + xlow_2 + xlow_1) # PILAS: this is not correct
    elif PNorder==3.0: Tchirp = (5/(256*nu)) * (G*Mtotal/c3) * ( xlow_8 + xlow_6 + xlow_5 + xlow_4 + xlow_3 + xlow_2 + xlow_1) # PILAS: this is not correct
    elif PNorder==3.5: Tchirp = (5/(256*nu)) * (G*Mtotal/c3) * ( xlow_8 + xlow_6 + xlow_5 + xlow_4 + xlow_3 + xlow_2 + xlow_1)
    
    # -------------------------------
    # Number of samples (at the chosen sampling frequency)
    Nsamples       = int(Tchirp * fs)
    
    # Plot
    if doprint:
        print("Duration of the GW signal:       {0:1.2f} s".format(Tchirp) )
        print("Number of samples in the signal: {0:d}".format(Nsamples) )
    
    # Return
    return Tchirp, Nsamples



# -------------------------------
# Compute $\Theta(t)$
def GWtheta(nu, Mtotal, t_coal, t, doplot):
    """
    $\Theta(t):= \frac{c^3\nu^2}{5Gm}(t_c-t)$

    See Magiore's book [Pag. 291, Equ. 5.242] or Thesis machos [Pag. 36, Equ. 2.99]
    """
    
    # Compute Thetat
    Thetat         = ( c3*nu/(5*G*Mtotal) ) * (t_coal-t)
    
    # Plot
    if doplot:
        plt.figure()
        plt.plot(t-t[-1], Thetat, label='$\Theta(t)$')
        plt.title("$\Theta(t)$")
        plt.xlabel('Time [s]')
        plt.ylabel('Dimensionless')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
      
    # Return
    return Thetat



# -------------------------------
# Compute orbital phase and frequency
def GWorbitalphaseandfrequency(PNorder, Thetat, nu, Mtotal, t, doplot):
    """
    For the case of circular orbit, named here 0 PN, (see Magiore's book [Pag. XX, Equ. XX]):
    $\phi(t) = ...$
    $w(t) = ...$
    
    For the case of 1 PN, (see Magiore's book [Pag. XX, Equ. XX]):
    $\phi(t) = ...$
    $w(t) = ...$
    
    For the case of 1.5 PN, (see Magiore's book [Pag. XX, Equ. XX]):
    $\phi(t) = ...$
    $w(t) = ...$
      
    Recall that
    $w(t) = \int \phi(t) dt$ COREGIR    
    and    
    $w(t) = 2 \pi f(t)$
    """
    
    # -------------------------------
    # Terms for 0 PN and the rest
    P_k0           = 1
    P_t0           = np.power(Thetat,5/8)
    
    W_k0           = 1
    W_t0           = np.power(Thetat,-3/8)
    
    # -------------------------------
    # Terms for 1 PN and the rest
    P_k1           = (3715/8064) + (55/96)*nu
    P_t1           = np.power(Thetat,3/8)
    
    W_k1           = (743/2688) + (11/32)*nu
    W_t1           = np.power(Thetat,-5/8)
    
    # -------------------------------
    # Terms for 1.5 PN and the rest
    P_k15         = -3*np.pi/4
    P_t15         = np.power(Thetat,1/4)
    
    W_k15         = -3*np.pi/10
    W_t15         = np.power(Thetat,-3/4)
    
    # -------------------------------
    # Terms for 2 PN and the rest
    P_k2          = (9275495/14450688) + (284875/258048)*nu + (1855/2048)*nu**2
    P_t2          = np.power(Thetat,1/8)
    
    W_k2          = (1855099/14450688) + (56975/258048)*nu + (371/2048)*nu**2
    W_t2          = np.power(Thetat,-7/8)
    
    # -------------------------------
    # Terms for 2.5 PN and the rest
    P_k25         = ( (-38645/172032) + (65/2048)*nu ) * np.pi
    P_t25         = 1*np.log(Thetat/Thetat[0])
    
    W_k25         = ( (-7729/21504) + (13/256)*nu ) * np.pi
    W_t25         = 1*np.power(Thetat,-1)
    
    # -------------------------------
    # Terms for 3 PN and the rest
    P_C           = 0.577
    P_k3          = ( \
        + (831032450749357/57682522275840) \
        - ((53/40)*np.pi**2) \
        - (107/56)*P_C \
        + (107/448)*np.log(Thetat/256) \
        + ( (-123292747421/4161798144)+(2255/2048)*np.pi**2 )*nu \
        + (154565/1835008)*nu**2 \
        - (1179625/1769472)*np.power(nu,3) \
        )
    # PILAS: P_k3 in Magiori is 126510089885 while in Blanket is 123292747421
    
    P_t3          = np.power(Thetat,-1/8)
    W_k3          = -(1/5)*( \
        + (831032450749357/57682522275840) \
        - ((53/40)*np.pi**2) \
        - (107/56)*P_C \
        + (107/448)*np.log(Thetat/256) \
        + ( (-123292747421/4161798144)+(2255/2048)*np.pi**2 )*nu \
        + (154565/1835008)*nu**2 \
        - (1179625/1769472)*np.power(nu,3) \
        - (107/56) \
        )
    # PILAS: W_k3 in Magiori is 126510089885 while in Blanket is 123292747421
    W_t3          = np.power(Thetat,-9/8)
    
    # -------------------------------
    # Terms for 3.5 PN and the rest
    P_k35         =  ( (188516689/173408256) + (488825/516096)*nu - (141769/516096)*nu**2 ) * np.pi
    P_t35         = np.power(Thetat,-1/4)
    
    W_k35         = -( (188516689/433520640) + (97765/258048)*nu - (141769/1290240)*nu**2 ) * np.pi
    W_t35         = np.power(Thetat,-5/4)
    
    # -------------------------------
    # Sum terms for the orbital phase phi(t) and frequency w(t)
    if   PNorder==0.0:
        P         = P_k0*P_t0
        W         = W_k0*W_t0
    
    elif PNorder==1.0:
        P         = P_k0*P_t0 + P_k1*P_t1
        W         = W_k0*W_t0 + W_k1*W_t1
    
    elif PNorder==1.5:
        P        = P_k0*P_t0 + P_k1*P_t1 + P_k15*P_t15
        W        = W_k0*W_t0 + W_k1*W_t1 + W_k15*W_t15
    
    elif PNorder==2.0:
        P        = P_k0*P_t0 + P_k1*P_t1 + P_k15*P_t15 + P_k2*P_t2
        W        = W_k0*W_t0 + W_k1*W_t1 + W_k15*W_t15 + W_k2*W_t2
    
    elif PNorder==2.5:
        P        = P_k0*P_t0 + P_k1*P_t1 + P_k15*P_t15 + P_k2*P_t2 + P_k25*P_t25
        W        = W_k0*W_t0 + W_k1*W_t1 + W_k15*W_t15 + W_k2*W_t2 + W_k25*W_t25
    
    elif PNorder==3.0:
        P        = P_k0*P_t0 + P_k1*P_t1 + P_k15*P_t15 + P_k2*P_t2 + P_k25*P_t25 + P_k3*P_t3
        W        = W_k0*W_t0 + W_k1*W_t1 + W_k15*W_t15 + W_k2*W_t2 + W_k25*W_t25 + W_k3*W_t3
    
    elif PNorder==3.5:
        P        = P_k0*P_t0 + P_k1*P_t1 + P_k15*P_t15 + P_k2*P_t2 + P_k25*P_t25 + P_k3*P_t3 + P_k35*P_t35
        W        = W_k0*W_t0 + W_k1*W_t1 + W_k15*W_t15 + W_k2*W_t2 + W_k25*W_t25 + W_k3*W_t3 + W_k35*W_t35
    
    # -------------------------------
    # Final calculation of the orbital phase phi(t)
    P             = (-1/nu) * P
    
    # -------------------------------
    # Final calculation of the orbital phase w(t) and f(t)
    W             = (c3/(8*G*Mtotal)) * W
    F             = W / (2*np.pi)
    
    # Plot
    if doplot:
        plt.figure()
        plt.plot(t-t[-1], P, label=str(PNorder)+" PN")
        plt.title("Orbital phase")
        plt.xlabel('Time [s]')
        plt.ylabel('Phase [rad]')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        plt.figure()
        plt.plot(t-t[-1], F, label=str(PNorder)+" PN")
        plt.title("Orbital frequency")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
      
    # Return
    return P, F



# -------------------------------
# Compute orbital phase and frequency
def GWphaseandfrequency(PNorder, Thetat, nu, Mtotal, phi_0, t, doplot):
    """
    REFERENCIAR
    """
    
    # -------------------------------
    # Compute orbital phase and frequency
    P, F = GWorbitalphaseandfrequency(PNorder, Thetat, nu, Mtotal, t, False)
    
    # -------------------------------
    # Compute GW phase and frequency
    phit          = 2*P - phi_0
    ft            = 2*F
    
    # Plot
    if doplot:
        plt.figure()
        plt.plot(t-t[-1], phit, label=str(PNorder)+" PN")
        plt.title("GW phase")
        plt.xlabel('Time [s]')
        plt.ylabel('Phase [rad]')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        plt.figure()
        plt.plot(t-t[-1], ft, label=str(PNorder)+" PN")
        plt.title("GW frequency")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
      
    # Return
    return phit, ft    



# -------------------------------
# Compute GW hp and hc
def GWamplitude(ft, Mchirp, t, doplot):
    """
    For $A(t)$ see Magiore's book [Pag. 291, Equ. 5.242] or Thesis machos [Pag. 36, Equ. 2.99]
    """
    
    # Compute A(t)
    At            = 4 * np.power(G*Mchirp/c2,5/3) * np.power(np.pi*ft/c,2/3)
    
    # Plot
    if doplot:
        plt.figure()
        plt.plot(t-t[-1], At, label="$A(t)$")
        plt.title("$A(t)$")
        plt.xlabel('Time [s]')
        plt.ylabel('Envelope amplitude')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Return
    return At


# -------------------------------
# Compute GW hp and hc
def GWhphc(phit, At, D, iota, t, doplot):
    """
    For $h_{+}$ and $h_{\times}$ see Magiore's book [Pag. XXX, Equ. XXX] or Thesis machos [Pag. XXX, Equ. XXX]
    """
    
    # Compute GW: hp and hc polarizations
    hp            = (1/D) * At * (1+np.cos(iota)**2)/2  * np.cos(phit)
    hc            = (1/D) * At * np.cos(iota)           * np.sin(phit)
    
    # Plot        
    if doplot:
        plt.figure()
        plt.plot(t-t[-1], hp, label=r"$h_{+}(t)$")
        plt.plot(t-t[-1], hc, label=r"$h_{\times}$(t)")
        plt.title(r"$h_{+}(t)$ and $h_{\times}$(t)")
        plt.xlabel('Time [s]')
        plt.ylabel('Strain')
        plt.grid()
        #plt.xlim(-0.1,0.001)
        plt.legend()
        plt.tight_layout()
        plt.show()
          
    # Return
    return hp, hc



# -------------------------------
# Compute GW hp and hc from physical parameters
def GWcomputehphc(DMpc, m1Mo, m2Mo, iota, phi_0, flow, fs, PNorder, doprint, doplot):
    """
    Compute GW hp and hc from physical parameters
    """
    
    # -------------------------------
    # 1) Compute all information
    D, m1, m2, Mtotal, mu, eta, Mchirp, fisco, Ncyc, Tchirp, Nsamples, \
    t, t_coal, phi_coal = GWphysicalinformation(DMpc, m1Mo, m2Mo, flow, fs, PNorder, doprint)
    
    # -------------------------------
    # 2) Compute $\Theta(t)$
    Thetat       = GWtheta(eta, Mtotal, t_coal, t, doplot)
    
    # -------------------------------
    # 3) Compute $\phi(t)$ and $f(t)$
    phit, ft     = GWphaseandfrequency(PNorder, Thetat, eta, Mtotal, phi_0, t, doplot)
    
    # -------------------------------
    # 4) Compute $A(t)$
    At           = GWamplitude(ft, Mchirp, t, doplot)
    
    # -------------------------------
    # 5) Compute $h_{+}(t)$ and $h_{\times}(t)$
    hp, hc       = GWhphc(phit, At, D, iota, t, doplot)
          
    # Return
    return hp, hc, t
