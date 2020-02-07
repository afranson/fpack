"""
Custom, commonly used functions or expressions.
"""

# from __future__ import division
import numpy as np

if __name__ == 'main':
    import sympy as sym
    import sympy.physics.vector as spv


def absorption(x, aHeight=1, FWHM=0.1, x0=3):
    """Constructs a lorentzian with height 'aHeight', full width half max
    'FWHM', and position 'x0'.  This lorentzian is constructed for
    height and FWHM to have independent effects on the lorentzian.

                        2
                    FWHM ⋅aHeight
    lorentzian(x) = ───────────────────
                        2             2
                    FWHM  + 4⋅(x - x₀)

    """
    return aHeight*FWHM**2/(FWHM**2 + 4*(x - x0)**2)


def absorption_derivative(x, aHeight=1, FWHM=0.1, x0=3):
    """Constructs a lorentzian derivative with height 'aHeight', full
    width half max 'FWHM', and position 'x0'.  This lorentzian
    derivative is constructed for height and FWHM to have independent
    effects on the lorentzian.

                                          3
                                -32⋅√3⋅FWHM ⋅aHeight⋅(x - x₀)
    absorption_derivative(x) =  ──────────────────────────────
                                                       2
                                 ⎛    2             2⎞
                                9⋅⎝FWHM  + 4⋅(x - x₀) ⎠

    """
    return -32*np.sqrt(3)*FWHM**3 * \
        aHeight*(x - x0)/(9*(FWHM**2 + 4*(x - x0)**2)**2)


def dispersion(x, dHeight=1, FWHM=0.1, x0=3):
    """Constructs a dispersion from a lorenztian with height 'dHeight',
    full width half max 'FWHM', and position 'x0'.  This dispersion is
    constructed for height and FWHM to have independent effects on the
    dispersion.

                    -4⋅FWHM⋅dHeight⋅(x - x₀)
    dispersion(x) = ─────────────────────────
                        2             2
                    FWHM  + 4⋅(x - x₀)

    """
    return -4*FWHM*dHeight*(x - x0)/(FWHM**2 + 4*(x - x0)**2)


def dispersion_derivative(x, dHeight=1, FWHM=0.1, x0=3):
    """Constructs a dispersion derivative from a lorenztian with height
    'dHeight', full width half max 'FWHM', and position 'x0'.  This
    dispersion is constructed for height and FWHM to have independent
    effects on the dispersion derivative.


                                    2         ⎛    2             2⎞
                                -FWHM ⋅dHeight⋅⎝FWHM  - 4⋅(x - x₀) ⎠
    dispersion_derivative(x) =  ─────────────────────────────────────
                                                              2
                                        ⎛    2             2⎞
                                        ⎝FWHM  + 4⋅(x - x₀) ⎠

    """
    return -FWHM**2 * \
        dHeight*(FWHM**2 - 4*(x - x0)**2)/(FWHM**2 + 4*(x - x0)**2)**2


def absorption_dispersion_mixed(x, aHeight=1, dHeight=1, FWHM=0.1, x0=3):
    """Constructs an absorption + dispersion from an absorption with
    height 'aHeight', dispertion with height 'dHeight', full width
    half max 'FWHM', and position 'x0'.  These function are
    constructed for height and FWHM to have independent effects on the
    dispersion derivative.  See functions 'absorption' and
    'dispersion' for more details.

    """
    return absorption(x, aHeight, FWHM, x0) + dispersion(x, dHeight, FWHM, x0)


def absorption_dispersion_derivative_mixed(x, aHeight=1, dHeight=1,
                                           FWHM=0.1, x0=3):
    """Constructs an absorption + dispersion derivative from an
    absorption with height 'aHeight', dispertion with height
    'dHeight', full width half max 'FWHM', and position 'x0'.  These
    function are constructed for height and FWHM to have independent
    effects on the dispersion derivative.  See functions
    'absorption_derivate' and 'dispersion_derivate' for more details.

    """
    return absorption_derivative(x, aHeight, FWHM, x0) + \
        dispersion_derivative(x, dHeight, FWHM, x0)


def mixing_FMR_function(x, height=1, FWHM=0.1, x0=3, theta=0.12):
    return np.cos(theta) * absorption(x, x0, FWHM, height) + \
        np.sin(theta) * dispersion(x, x0, FWHM, height)


def bandpassfilterLorentzian(x0, FWHM=0.1, height=1,
                             low_freq_cutoff=100, high_freq_cutoff=1000):
    """Constructs a lorentzian with position 'x0', full width half max
    'FWHM', and height 'height'. Returns a function of one
    variable. This lorentzian is not normalized, rather is constructed
    for height and FWHM to have independent effects on the
    lorentzian. The lorentzian is missing all Fourier components
    (frequencies) below low_freq_cutoff and above high_freq_cutoff.
    Note: There is sometimes a factor of 2pi(f+1) involved due to
    differences between using f and omega.

    :param x0: Resonant position of lorentzian
    :param FWHM: Full Width Half Max of lorentzian
    :param height: Height of lorentzian
    :param low_freq_cutoff: Lowest allowed frequency component.
    :param high_freq_cutoff: Highest allowed frequency component.
    :return: Analytic lorentzian function with selected Fourier components.

    """
    def return_lorentzian(x):
        return (1/(FWHM**2 + 4 * (x-x0)**2)) * height * np.exp(-0.5 * (low_freq_cutoff + high_freq_cutoff) * FWHM)\
               * FWHM * (np.exp((high_freq_cutoff * FWHM)/2) * FWHM * np.cos(low_freq_cutoff * (x-x0)) - 2
                         * np.exp((high_freq_cutoff * FWHM)/2) * (x-x0) * np.sin(low_freq_cutoff * (x-x0))
                         + np.exp((low_freq_cutoff * FWHM)/2) * (-FWHM * np.cos(high_freq_cutoff * (x-x0))
                                                                 + 2 * (x-x0) * np.sin(high_freq_cutoff * (x-x0))))
    return return_lorentzian


# Equation derived below
def theta_path(t=0, t_0=0, theta_1=0, theta_2=0, phi_1=0, phi_2=0):
    """Returns theta along a path defined by v1=(1,theta_1,phi_1) to
    v2=(1,theta_2,phi_2). The path reaches v2 when t=the angle between
    them.  t = 2 pi will give v1, i.e. a full rotation will be
    achieved.

    """
    return np.arctan(np.sqrt((np.sin(phi_1)*np.sin(theta_1)*np.sin((-(t - t_0)/np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)) + 1)*np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2))) + np.sin(phi_2)*np.sin(theta_2)*np.sin(t - t_0))**2/(-(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2))**2 + 1) + (np.sin(theta_1)*np.sin((-(t - t_0)/np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)) + 1)*np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)))*np.cos(phi_1) + np.sin(theta_2)*np.sin(t - t_0)*np.cos(phi_2))**2/(-(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2))**2 + 1))*np.sqrt(-(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2))**2 + 1)/(np.sin((-(t - t_0)/np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)) + 1)*np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)))*np.cos(theta_1) + np.sin(t - t_0)*np.cos(theta_2)))


# Equation derived below
def phi_path(t=0, t_0=0, theta_1=0, theta_2=0, phi_1=0, phi_2=0):
    """Returns phi along a path defined by v1=(1,theta_1,phi_1) to
    v2=(1,theta_2,phi_2). The path reaches v2 when t= the angle
    between them.  t = 2 pi will give v1, i.e. a full rotation will be
    achieved.

    """
    return np.arctan((np.sin(phi_1)*np.sin(theta_1)*np.sin((-(t - t_0)/np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)) + 1)*np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2))) + np.sin(phi_2)*np.sin(theta_2)*np.sin(t - t_0))/(np.sin(theta_1)*np.sin((-(t - t_0)/np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)) + 1)*np.arccos(np.sin(phi_1)*np.sin(phi_2)*np.sin(theta_1)*np.sin(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(phi_1)*np.cos(phi_2) + np.cos(theta_1)*np.cos(theta_2)))*np.cos(phi_1) + np.sin(theta_2)*np.sin(t - t_0)*np.cos(phi_2)))


# Equation derived below
def fmr_resonant_field(omega_res=9.83e3, M=100, gamma=2.8, Nx=0, Ny=0,
                       Nz=1, theta_H=0, phi_H=0):
    """Returns resonant field for an FMR measurement in a Bruker; derived
    in the funcs.py file. Assumed sample is well saturated.

    kwargs:
    omega_res  -- The resonant frequency that the measurement is taken at.
    M          -- The magnetization of the sample, will be Meff if two demagnetizing components are 0, will be M if one or less are 0.
    gamma      -- Gyromagnetic ratio of sample
    Nx, Ny, Nz -- Demagnetizing components along the x, y, and z direction.
    theta_H    -- Polar angle of magnetization (angle between z axis and magnetization)
    phi_H      -- Azimuthal angle of magnetization (angle between x axis and magnetization)

    Make sure the units of omega_res, M, and gamma have the same
    relative units.  i.e. omega_res [=] MHz; M [=] G; gamma [=] MHz/G

    """
    return (-M*gamma*(Nx + Ny - 2*Nz + 3*(-Nx + Nz + (Nx - Ny)*np.sin(phi_H)**2)*np.sin(theta_H)**2) + np.sqrt(M**2*gamma**2*(Nx + Ny - 2*Nz + 3*(Nx*np.sin(phi_H)**2 - Nx - Ny*np.sin(phi_H)**2 + Nz)*np.sin(theta_H)**2)**2 - 4*M**2*gamma**2*(2*Nx**2*np.cos(phi_H)**4*np.cos(theta_H)**4 - 4*Nx**2*np.cos(phi_H)**4*np.cos(theta_H)**2 + 2*Nx**2*np.cos(phi_H)**4 + Nx**2*np.cos(phi_H)**2*np.cos(theta_H)**2 - Nx**2*np.cos(phi_H)**2 - 2*Nx*Ny*np.sin(theta_H)**2 + Nx*Ny - 4*Nx*Nz*np.sin(theta_H)**4 + 4*Nx*Nz*np.sin(theta_H)**2 - Nx*Nz - Ny*Nz*np.cos(2*theta_H) - 2*Ny*(2*Nx - Ny)*np.sin(phi_H)**4*np.sin(theta_H)**4 + Nz**2*np.cos(theta_H)**2*np.cos(2*theta_H) + (4*Nx*Ny*np.sin(theta_H)**2 + 4*Nx*Nz*np.sin(theta_H)**2 - 2*Nx*Nz - Ny**2 - 4*Ny*Nz*np.sin(theta_H)**2 + 2*Ny*Nz)*np.sin(phi_H)**2*np.sin(theta_H)**2) + 4*omega_res**2))/(2*gamma)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sym.init_printing(pretty_print=True, wrap_line=False)

    print("\nDeriving equations for arbitrary paths on a unit sphere.")
    N = spv.ReferenceFrame('N')
    v_1, v_2, v_m, theta_1, theta_2, phi_1, phi_2, eta, d, s_1, s_2, full_circle, t, t_0 = sym.symbols('v_1 v_2 v_m theta_1 theta_2 phi_1 phi_2 eta d s_1 s_2 full_circle t t_0', real=True)
    v_1 = N.x * sym.sin(theta_1) * sym.cos(phi_1) + N.y * sym.sin(theta_1) * sym.sin(phi_1) + N.z * sym.cos(theta_1)
    v_2 = N.x * sym.sin(theta_2) * sym.cos(phi_2) + N.y * sym.sin(theta_2) * sym.sin(phi_2) + N.z * sym.cos(theta_2)
    eta = sym.acos(spv.dot(v_1, v_2))
    half_circle = sym.pi / eta
    full_circle = 2*sym.pi / eta
    d = sym.sin(eta)
    s_1 = sym.sin(eta * (1-(t-t_0)/eta))
    s_2 = sym.sin(eta * (t-t_0)/eta)
    v_m = (s_1*v_1 + s_2*v_2)/d
    v_m_x, v_m_y, v_m_z = spv.dot(v_m, N.x), spv.dot(v_m, N.y), spv.dot(v_m, N.z)
    v_m_theta = sym.atan((sym.sqrt(v_m_x**2 + v_m_y**2) / v_m_z).collect(sym.sin(theta_1)).collect(sym.sin(theta_2)))
    v_m_phi = sym.atan((v_m_y / v_m_x).collect(sym.sin(theta_1)).collect(sym.sin(theta_2)))

    # print(str(full_circle).replace('sin', 'np.sin').replace('cos', 'np.cos').replace('anp.', 'np.arc').replace('sqrt', 'np.sqrt').replace('pi','np.pi'))
    # print()
    print(str(v_m_theta).replace('sin', 'np.sin').replace('cos', 'np.cos').replace('anp.', 'np.arc').replace('sqrt', 'np.sqrt').replace('atan', 'np.arctan'))
    print()
    print(str(v_m_phi).replace('sin', 'np.sin').replace('cos', 'np.cos').replace('anp.', 'np.arc').replace('sqrt', 'np.sqrt').replace('atan', 'np.arctan'))
    print()
    exit()

    # Check the magnitude is equal to 1
    # sym.pprint(sym.mathematica_code(v_m.magnitude().subs(((theta_1, 0),(theta_2, sym.pi/2),(phi_1,0),(phi_2,sym.pi/2)))))
    # sym.pprint(sym.mathematica_code(v_m_sphere_theta.subs(((theta_1, 0.119),(theta_2, 0.543),(phi_1, 0.13),(phi_2, 1.442)))))
    # para = ((theta_1, 0.5),(theta_2, np.pi/2),(phi_1, 0.2),(phi_2, 0.7))
    # theta_path = sym.lambdify(t, v_m_theta.subs(para))
    # phi_path = sym.lambdify(t, v_m_phi.subs(para))
    # x = np.linspace(0, 2*np.float(half_circle.subs(para)), 1000)
    # y1 = theta_path(x)
    # y2 = phi_path(x)
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.show()
    # exit()
    print("Done.")

    print("\nDeriving H_res as a function of shape anisotropy for generic rectangular shapes.")
    M, Heff, gamma, H, theta_H, theta_M, phi_H, phi_M, Nx, Ny, Nz, theta_0, phi_0, omega_res, H_res, omega_sq = sym.symbols("M Heff gamma H theta_H theta_M phi_H phi_M Nx Ny Nz theta_0 phi_0 omega_res H_res, omega_sq", real=True, positive=True)
    # Sets up the free energy expression to be evaluated
    free_energy = -M*H*(sym.sin(theta_M) * sym.sin(theta_H) * sym.cos(phi_M-phi_H) + sym.cos(theta_M)*sym.cos(theta_H)) +\
    (M**2/sym.S(2))*((Nz*sym.cos(theta_M))**2/(Nz) + (Ny*sym.sin(theta_M)*sym.sin(phi_M))**2/(Ny) + (Nx*sym.sin(theta_M)*sym.cos(phi_M))**2/(Nx))
    print("Free energy expression:")
    sym.pprint(free_energy, wrap_line=False)
    print()
    
    # Uses RGT method to acquire harmonic solutions to the equation of
    # motion -- outputs the square as it's easier to
    # test/compute/simplify with
    print("Working on solving for omega**2.")
    omega_sq = (gamma/M)**2 *(\
    sym.diff(free_energy, theta_M, theta_M) * (sym.diff(free_energy, phi_M, phi_M)/sym.sin(theta_M)**2 + sym.cos(theta_M)*sym.diff(free_energy, theta_M)/sym.sin(theta_M)) -\
    (sym.diff(free_energy, theta_M, phi_M)/sym.sin(theta_M) - sym.cos(theta_M)*sym.diff(free_energy, phi_M)/sym.sin(theta_M)**2)**2)
    # Here is the full expression for w(H, ...) with magnetization
    # angles made equivalent to applied field angles (assumes well
    # saturated)
    collected_omega_sq = omega_sq.subs(((theta_M, theta_H), (phi_M, phi_H))).expand().collect(H)
    collected_omega_sq_H_sq = collected_omega_sq.coeff(H, 2).simplify()
    collected_omega_sq_H = collected_omega_sq.coeff(H, 1).expand().trigsimp().expand().collect(sym.sin(phi_H)).trigsimp().collect(M).collect(gamma).collect(sym.sin(theta_H))
    collected_b_sq = (collected_omega_sq_H**2).expand().trigsimp().expand().collect(sym.sin(phi_H)).trigsimp().collect(M).collect(gamma).collect(sym.sin(theta_H))
    collected_omega_sq_H_0 = collected_omega_sq.coeff(H, 0).collect(Nx).collect(Ny).collect(Nz).simplify()
    collected_4ac = (4*collected_omega_sq_H_sq*(collected_omega_sq_H_0 - omega_res**2)).expand().trigsimp().expand().collect(sym.sin(phi_H)).trigsimp().collect(M).collect(gamma).collect(sym.sin(theta_H))
    # sym.pprint(collected_4ac)
    omega_sq = H**2 * collected_omega_sq_H_sq + H * collected_omega_sq_H + collected_omega_sq_H_0
    print("Finished deriving omega_sq.")
    # sym.pprint(sym.mathematica_code(omega_sq)) ## For comparing with Mathematica results
 
    # Solve for H(w, ...)
    print("Working on solving for H_res.")
    # H_res = sym.solve(omega_res**2 - omega_sq, H)[0]
    H_res = (-collected_omega_sq_H + sym.sqrt(collected_b_sq - collected_4ac))/(2*collected_omega_sq_H_sq)
    # H_res = H_res.subs(((theta_H, v_m_theta), (phi_H, v_m_phi)))
    H_res_str = str(H_res.simplify())
    H_res_str = H_res_str.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('sqrt', 'np.sqrt')
    print(H_res_str)
    print("Finished deriving H_res.")
    exit()
    para_system = ((gamma, 2.8), (M, 95), (Nx, 0.1), (Ny, 0.3), (Nz, 0.6), (omega_res, 10000), (t_0, 0.3))
    para_phi_0 = ((theta_1, 0), (theta_2, np.pi/2), (phi_1, 0), (phi_2, 0))
    # para_phi_pi2 = ((theta_1, 0),(theta_2, np.pi/2),(phi_1, np.pi/2),(phi_2, np.pi/2))
    para_phi_pi2 = ((theta_1, 0.2), (theta_2, np.pi/2), (phi_1, 0), (phi_2, np.pi/2))
    # para_theta_0 = ((theta_1, 0),(phi_1, 0),(phi_2, np.pi/2))
    para_theta_pi2 = ((theta_1, np.pi/2), (theta_2, np.pi/2), (phi_1, 0), (phi_2, np.pi/2))

    Hphi0 = sym.lambdify(t, H_res.subs((*para_system, *para_phi_0)))
    Hphipi2 = sym.lambdify(t, H_res.subs((*para_system, *para_phi_pi2)))
    # Htheta0 = sym.lambdify(t, H_res.subs((*para_system, *para_theta_0)).simplify().subs(theta_2,0))
    Hthetapi2 = sym.lambdify(t, H_res.subs((*para_system, *para_theta_pi2)))

    t = np.linspace(0, np.float(half_circle.subs(para_phi_pi2)), 100)
    y1 = Hphi0(t)
    y2 = Hphipi2(t)
    # y3 = list(map(Htheta0, t))
    y4 = Hthetapi2(t)

    plt.plot(t,y1)
    plt.plot(t,y2)
    # plt.plot(t,y3)
    plt.plot(t,y4)
    plt.show()
    exit()

    print("Deriving Lorentzian expressions found in funcs.py using sympy.")
    # aHeight is for absorptive scaling, dHeight is for dispersive
    # scaling, FWHM is the Full-Width at Half Max, x0 is the resonant
    # position
    aHeight, dHeight, FWHM, x, x0 = sym.symbols('aHeight dHeight FWHM x x0')

    symb_lorentz = aHeight*FWHM**2/(4*(x-x0)**2+FWHM**2)
    symb_disp = -4*dHeight*FWHM*(x-x0)/(4*(x-x0)**2+FWHM**2)

    symb_lorentz_deriv = sym.simplify(sym.diff(symb_lorentz, x)*4*FWHM/(3 * sym.sqrt(3)))
    # Additional factors to normalize peak height
    symb_lorentz_secondderiv = sym.simplify(sym.diff(symb_lorentz_deriv, x))
    symb_disp_deriv = sym.simplify(sym.diff(symb_disp, x)*FWHM/4)
    # Additional factors to normalize peak height
    symb_disp_secondderiv = sym.simplify(sym.diff(symb_disp_deriv, x))

    r, c = sym.solveset(symb_lorentz_deriv, x).args
    half_max_pair, c = sym.solveset(symb_lorentz - symb_lorentz.subs(x, r.args[0])/2, x).args
    # print("Absorptive Lorentzian | Max Value: ({}) at x = ({}) | FWHM of ({})".format(symb_lorentz.subs(x, r.args[0]), r.args[0], half_max_pair.args[1]-half_max_pair.args[0]))

    # print(symb_lorentz)
    # sym.pprint(symb_lorentz)

    r, c = sym.solveset(symb_disp_deriv, x).args
    # print("\nDispersion Lorenztian : Max Value: ({}) at x = ({})".format(symb_disp.subs(x, r.args[0]), r.args[0]))
    # print(symb_disp)
    # sym.pprint(symb_disp)

    r, c = sym.solveset(symb_lorentz_secondderiv, x).args
    # print("\nAbsorption Derivative : Max Value: ({}) at x = ({}) | Peak to Peak of ({})".format(symb_lorentz_deriv.subs(x, r.args[0]), r.args[0], r.args[1]-r.args[0]))
    # print(symb_lorentz_deriv)
    # sym.pprint(symb_lorentz_deriv)

    r, c = sym.solveset(symb_disp_secondderiv, x).args
    # print("\nDispersion Derivative : Max Value: ({}) at x = ({})".format(symb_disp_deriv.subs(x, r.args[0]), r.args[0]))
    # print(symb_disp_deriv)
    # sym.pprint(symb_disp_deriv)
