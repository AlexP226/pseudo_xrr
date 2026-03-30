# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import trapezoid, simpson,dblquad,quad # ChatGPT contribution to replace the integration with quad for better performance
from scipy.special import kv as besselk, jv as besselj, gamma
from scipy.constants import pi, Boltzmann as kb
from scipy.optimize import curve_fit
#from numba import njit
#@jit(nopython=True, parallel=True)  # Use Numba for performance optimization
from joblib import Parallel, delayed


"""
changes 2025 november by Chen:
calc_film_DS_RRF_integ: input DSqxy_HWHM changes to DSphi_HWHM
film_integral_delta_beta_delta_phi: merge with the approx together
"""

# -------------------------------------------------------------
# extended capillary wave model : roughness factor calculation
# -------------------------------------------------------------

def eCWM_correlation_integrand_replacement(r, qxy, eta, Lk, amin): # Changes made
    Lk = np.maximum(Lk, 0.001)  # safeguard for divide-by-zero. Qk = 1000 when Lk = 0.001 therefore is irelevant
    r = np.asarray(r)  # Ensure r is a numpy array
    rad_term = np.sqrt(r[None, None, :]**2 + amin**2)  # r turned into r[None, None, :] b/c sizing errors when integrating and the [..., None] at the end was removed
    term1 = rad_term**(1 - eta[..., None])
    term2 = np.exp(-eta[..., None] * besselk(0, rad_term / Lk)) - 1
    term3 = besselj(0, rad_term * qxy[..., None])
    return term1 * term2 * term3


def eCWM_diffPsi_red(beta_rad, 
                     phi_rad, 
                     kbT_gamma, 
                     wave_number, 
                     alpha, 
                     Lk, 
                     amin, 
                     use_approx = False):
    """
    Calculate the reduced differential roughness factor psi_red(Qxy, Qz).

    This function evaluates the reduced differential roughness factor
    psi_red(Qxy, Qz) for x-ray scattering from a liquid surface or thin film
    according to the extended capillary wave model (eCWM).

    The full differential roughness factor psi(Qxy, Qz) is defined in Eq. (9)
    of the paper. In the present function, the geometric prefactor

        Qz^4 / (16 * pi^2 * sin(alpha))

    is intentionally left out. Therefore, this function returns the reduced
    quantity

        psi_red(Qxy, Qz) = psi(Qxy, Qz) / [Qz^4 / (16*pi^2*sin(alpha))]

    so that the full differential roughness factor can be reconstructed as

        psi(Qxy, Qz) =
            psi_red(Qxy, Qz) * Qz^4 / (16*pi^2*sin(alpha))

    This reduced form is convenient for numerical angular integration, where
    the prefactor may be applied afterwards at the level of the final
    roughness-factor integral.

    Parameters
    ----------
    beta_rad : array-like or float
        Exit angle beta in radians. May be a scalar or NumPy array.

    phi_rad : array-like or float
        In-plane scattering angle phi in radians. May be a scalar or NumPy
        array. Must be broadcast-compatible with beta_rad.

    kbT_gamma : float
        Thermal capillary prefactor k_B*T/gamma in Å^2.

    wave_number : float
        Incident wave number k0 = 2*pi/lambda in 1/Å.

    alpha : float
        Incident angle alpha in degrees.
        Note: alpha is given in degrees here, while beta_rad and phi_rad are
        given in radians.

    Lk : float
        Characteristic bending-rigidity length in Å,
        Lk = sqrt(kappa * k_B * T / gamma).

    amin : float
        Molecular cutoff length in Å, used to define
        Qmax = pi / amin.

    use_approx : bool, optional
        If True, use the approximate eCWM form.
        If False, use the more complete / accurate form based on the
        correlation-function integral.
        Default is False.

    Returns
    -------
    result : ndarray or float
        Reduced differential roughness factor psi_red(Qxy, Qz), with the same
        broadcasted shape as the input beta_rad / phi_rad arrays.

    Notes
    -----
    - The scattering-vector components are calculated internally as

          Qxy = k0 * sqrt((cos(beta) * sin(phi))^2
                          + (cos(alpha) - cos(beta) * cos(phi))^2)

          Qz  = k0 * (sin(alpha) + sin(beta))

      with alpha interpreted in degrees and beta, phi in radians.

    - The returned quantity does not include the prefactor
      Qz^4 / (16*pi^2*sin(alpha)).

    - For use in diffuse roughness-factor calculations, this prefactor is
      typically applied afterwards, for example during the final angular
      integration over beta and phi.

    - In the limit kappa -> 0, the expression approaches the standard
      capillary wave model (CWM) form.

    References
    ----------
    Chen Shen, Honghu Zhang, Beate Kloesgen, and Benjamin M. Ocko,
    "Extending the capillary wave model to include the effect of
    bending rigidity: X-ray reflectivity and diffuse scattering",
    Phys. Rev. Research 7, 043016 (2025).

    See:
    - Eq. (9): full differential roughness factor psi(Qxy, Qz)
    - Eq. (10): simplified eCWM form
    """
    qmax = pi / amin
    Lk = np.maximum(Lk, 0.001)  # [A] safeguard for divide-by-zero. Qk = 1000 when Lk = 0.001 therefore is irelevant
    beta_rad = np.asarray(beta_rad) # converting to numpy array for performance/vectorization
    phi_rad = np.asarray(phi_rad)
    alpha_rad = np.radians(alpha)
    
    cosb = np.cos(beta_rad)
    sinb = np.sin(beta_rad)
    cosp = np.cos(phi_rad)
    sinp = np.sin(phi_rad)
    cosa = np.cos(alpha_rad)
    sina = np.sin(alpha_rad)
    
    qxy = wave_number * np.sqrt((cosb * sinp)**2 + (cosa - cosb * cosp)**2)
    qz = wave_number * (sina + sinb)
    eta = (kbT_gamma / (2 * pi)) * qz**2
    # Safeguard against divide-by-zero or underflow
    qxy = np.maximum(qxy, 1e-12)
    safe_besselk_arg = np.maximum(1 / (Lk * qmax), 1e-12)
    exp_term = np.exp(eta * besselk(0, safe_besselk_arg))
    
    if use_approx:
        '''
        approximation form
        '''
        result = kbT_gamma * (1 / qmax)**eta * exp_term * qxy**eta / (qxy**2 + (Lk**2) * qxy**4)  
    
    else:
        '''
        accurate form
        '''
        r_vals = np.linspace(0.001, 8 * Lk, 300)
        integrand_vals = eCWM_correlation_integrand_replacement(r_vals, qxy, eta, Lk, amin)
        integral_vals = trapezoid(integrand_vals, r_vals, axis=-1) # might need axis = 1
        C_prime = 2 * pi * integral_vals
        xi = 2 ** (1 - eta) * gamma(1 - 0.5 * eta) / gamma(0.5 * eta) * (2 * pi) / (qz ** 2)    # xi used to = (2 * Lk) ** eta, but in MATLAB looks like: xi = (2.^(1-eta).*gamma(1-0.5*eta)./gamma(0.5*eta)) *2*pi./qz.^2;
        result = (xi * qxy ** (eta - 2) + C_prime / qz**2) * (1 / qmax) ** eta * exp_term 
    
    # reduced differential roughness factor:
    # full psi(Qxy, Qz) with prefactor Qz^4 / (16*pi^2*sin(alpha)) removed
    return result

# -------------------- Main Calculation -------------------- #
# Naming convention used in this file:
# psi(Qxy, Qz)      : full differential roughness factor
# psi_red(Qxy, Qz)  : reduced differential roughness factor
# Psi_DS(Qz, Qxy0)  : diffuse roughness factor after angular integration
# Psi_R(Qz)         : specular roughness factor after angular integration
# r_red             : reduced ratio Psi_DS / Psi_R

def calc_eCWM_roughness_factor_DS(alpha, beta_space, phi, 
                                  energy = None, 
                                  DSphi_HWHM = None, 
                                  DSbeta_HWHM = None,
                                  tension = 0.073, 
                                  temp = 295, 
                                  kappa = 0, 
                                  amin = 3.1, 
                                  use_approx=False):
    """
    Calculate the diffuse roughness factor Psi_DS by angular integration
    of the eCWM differential roughness factor over a finite detector window.

    This function evaluates the diffuse/off-specular roughness factor
    Psi_DS(Qz, Qxy0) according to the extended capillary wave model (eCWM).
    It numerically integrates the differential roughness factor over a finite
    detector acceptance centered at a chosen off-specular position defined by
    (beta, phi).

    In contrast to the specular roughness factor, this function is intended
    for scattering measured away from the specular ridge, i.e. for diffuse
    scattering (R*). The integration corresponds to the angular roughness-
    factor formalism of Eq. (16) in the paper, evaluated over a rectangular
    angular window in beta and phi around the selected diffuse condition.

    For each point in beta_space:
    - beta defines the center of the detector window in the out-of-plane
      direction,
    - phi defines the center of the detector window in the in-plane
      direction,
    - the finite acceptance is given by:
          beta ∈ [beta - DSbeta_HWHM, beta + DSbeta_HWHM]
          phi  ∈ [phi  - DSphi_HWHM,  phi  + DSphi_HWHM]

    The function returns the diffuse roughness factor only. To obtain the full
    diffuse scattering intensity or a reduced reflectivity quantity, this term
    must be combined with the corresponding Fresnel / intrinsic structure
    factor terms elsewhere.

    Parameters
    ----------
    alpha : float
        Incident angle in degrees.
        Must be a single scalar value.

    beta_space : array-like
        One-dimensional array of exit angles beta in degrees.
        A diffuse roughness factor is calculated for each value.

    phi : float
        In-plane angular offset from the specular condition in degrees.
        Must be a single scalar value.
    
    -- keyward argument, must be given --
    
    energy : float
        X-ray energy in eV.

    DSphi_HWHM : float
        Half-width at half-maximum (HWHM) of the detector acceptance in
        phi direction, in degrees.

    DSbeta_HWHM : float
        Half-width at half-maximum (HWHM) of the detector acceptance in
        beta direction, in degrees.

    tension : float
        Surface tension gamma in N/m.

    temp : float
        Temperature in K.

    kappa : float
        Bending rigidity in units of k_B T.
        kappa = 0 corresponds to the standard capillary wave model limit.

    amin : float
        Molecular cutoff length in Angstrom, used to define
        Qmax = pi / amin.

    use_approx : bool, optional
        If True, use the approximate form of the eCWM differential roughness
        factor. If False, use the more complete / accurate expression.
        Default is False.

    Returns
    -------
    eCWM_Psi_DS : ndarray
        One-dimensional NumPy array of shape (len(beta_space),)
        containing the diffuse roughness factor evaluated for each beta value.

    Notes
    -----
    - The function performs numerical integration over a rectangular angular
      window using Simpson integration.
    - The diffuse scattering condition is determined by alpha, beta, and phi.
    - The corresponding momentum transfer components are approximately:
          Qz  = k0 * (sin(alpha) + sin(beta))
          Qxy = |Qxy(alpha, beta, phi)|
      where k0 = 2*pi/lambda.
    - The returned quantity contains only the thermal roughness contribution
      from the eCWM.

    References
    ----------
    Chen Shen, Honghu Zhang, Beate Kloesgen, and Benjamin M. Ocko,
    "Extending the capillary wave model to include the effect of
    bending rigidity: X-ray reflectivity and diffuse scattering",
    Phys. Rev. Research 7, 043016 (2025).

    In particular, see:
    - Eq. (16): angular roughness-factor definition
    - Eq. (18): finite detector angular integration form
    """
    
    # ------------------------------------------------------------
    # Validate required inputs for diffuse scattering calculation
    # ------------------------------------------------------------
    required_params = {
        "energy": energy,
        "DSphi_HWHM": DSphi_HWHM,
        "DSbeta_HWHM": DSbeta_HWHM,
    }
    
    for name, value in required_params.items():
        if value is None:
            raise ValueError(f"{name} must be provided for diffuse scattering calculation.")
    
        # check scalar (not array/list)
        if np.ndim(value) != 0:
            raise ValueError(f"{name} must be a scalar (float), not an array.")
    
        # try converting to float
        try:
            required_params[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a float (or convertible to float).")
    
    # overwrite with validated values
    energy = required_params["energy"]
    DSphi_HWHM = required_params["DSphi_HWHM"]
    DSbeta_HWHM = required_params["DSbeta_HWHM"]
    
    # ------------------------------------------------------------
    # start calculation
    # ------------------------------------------------------------
    
    wavelength = 12400.0 / energy
    wave_number = 2 * pi / wavelength
    qz = wave_number * (np.sin(np.radians(alpha)) + np.sin(np.radians(beta_space)))
    # prefactor that converts reduced differential roughness factor
    # psi_red(Qxy, Qz) into the full differential roughness factor psi(Qxy, Qz)
    diffPsi_prefactor = qz**4 /(16* pi**2 * np.sin(np.radians(alpha)))
    
    phi_upper = phi + DSphi_HWHM
    phi_lower = phi - DSphi_HWHM

    beta_upper = beta_space + DSbeta_HWHM
    beta_lower = beta_space - DSbeta_HWHM

    kbT_gamma = kb * temp / tension * 1e20
    Lk = np.sqrt(kappa * kb * temp / tension) * 1e10

    eCWM_Psi_DS = np.zeros(len(beta_space))
    phi_grid = np.linspace(phi_lower, phi_upper, 100)

    for idx, beta in enumerate(beta_space):
        beta_grid = np.linspace(beta_lower[idx], beta_upper[idx], 100)
        beta_mesh, phi_mesh = np.meshgrid(beta_grid, phi_grid, indexing='ij')
        vals = eCWM_diffPsi_red(np.radians(beta_mesh), np.radians(phi_mesh), kbT_gamma, wave_number, alpha, Lk, amin, use_approx = use_approx)
        # integrate reduced differential roughness factor over detector acceptance,
        # then restore the full prefactor to obtain the diffuse roughness factor Psi_DS
        eCWM_Psi_DS[idx] = simpson(simpson(vals, np.radians(phi_grid)), np.radians(beta_grid)) * diffPsi_prefactor[idx]

    return eCWM_Psi_DS

def calc_eCWM_roughness_factor_SP(
    qz_space,
    energy=None,
    sdd=1000,
    resolution_mode=0,
    resolution=0.0002,
    bkg_mode=None,
    bkg_off= 1,
    tension=0.073,
    temp=295,
    kappa=0,
    amin=3.1,
    use_approx = False
    ):

    """
    Calculate the specular roughness factor Psi_R for finite detector resolution
    according to the extended capillary wave model (eCWM).

    This function evaluates the roughness factor Psi_R(Qz) for specular
    x-ray reflectivity from a liquid surface or thin film according to the
    extended capillary wave model (eCWM). The returned roughness factor
    contains the effect of thermally excited height fluctuations including
    the influence of bending rigidity kappa.

    Two resolution descriptions are supported:

    1) resolution_mode = 0
       Circular in-plane Qxy resolution, given directly in reciprocal space
       in units of 1/Angstrom. This corresponds to the circular-resolution
       treatment of the specular roughness factor [Eq. (19) in the paper].

    2) resolution_mode = 1
       Rectangular detector slit resolution, given in real detector-space
       units (mm) as [vertical_half_width, horizontal_half_width]. In this
       mode, the slit size is converted into angular acceptance using the
       x-ray energy and the sample-to-detector distance (sdd), and the
       roughness factor is evaluated from the slit-integrated differential
       roughness factor [Eq. (18) in the paper].

    Special treatment of the singularity at Qxy = 0
    -----------------------------------------------
    In slit mode, the differential roughness factor is singular at the
    specular position Qxy = 0. To handle this robustly, the calculation is
    split into two parts for each beta angle:

    - First, the detector slit edge is mapped into Qxy space.
    - Then, the largest circle around the specular position that fits fully
      inside the slit is determined.
    - The contribution inside this circle is evaluated with the circular
      specular expression [Eq. (19)] to treat the singular part analytically.
    - The remaining slit area outside that circle is evaluated numerically
      using the differential form [Eq. (18)].

    This hybrid treatment avoids numerical problems at Qxy = 0 while still
    preserving the true rectangular slit geometry.

    Optional off-specular background subtraction
    --------------------------------------------
    In slit mode, an off-specular background can optionally be estimated by
    shifting the slit away from the specular position:

    - bkg_mode = None : no background subtraction
    - bkg_mode = 0    : horizontal slit offset (phi direction)
    - bkg_mode = 1    : vertical slit offset (beta direction)

    The offset magnitude is given by bkg_off in mm.

    Parameters
    ----------
    qz_space : array-like
        One-dimensional array of Qz values [1/Angstrom] at which the
        specular roughness factor is calculated.

    energy : float, optional
        X-ray energy in eV. Required when resolution_mode == 1.
        Not used when resolution_mode == 0.

    sdd : float, optional
        Sample-to-detector distance in mm. Required when
        resolution_mode == 1. Default is 1000.

    resolution_mode : int, optional
        Selects the resolution description:
        - 0 : circular Qxy resolution in reciprocal space
        - 1 : rectangular slit resolution in detector space
        Default is 0.

    resolution : float or array-like
        Resolution parameter, interpreted according to resolution_mode.

        If resolution_mode == 0:
            Single float giving the circular Qxy half width
            dQxy_R [1/Angstrom].

        If resolution_mode == 1:
            Two-element array-like [slit_v_HWHM, slit_h_HWHM] in mm,
            giving the detector slit half widths in the vertical and
            horizontal directions.

    bkg_mode : None or int, optional
        Background subtraction mode, only relevant when
        resolution_mode == 1.
        - None : no background subtraction
        - 0    : horizontal slit offset
        - 1    : vertical slit offset
        Default is None.

    bkg_off : float, optional
        Background slit offset in mm when bkg_mode is 0 or 1.
        Ignored when bkg_mode is None or when resolution_mode == 0.
        Default is 1.

    tension : float, optional
        Surface tension gamma in N/m. Default is 0.073.

    temp : float, optional
        Temperature in K. Default is 293.

    kappa : float, optional
        Bending rigidity in units of k_B T. Default is 0.
        kappa = 0 recovers the standard capillary wave model limit.

    amin : float, optional
        Molecular cutoff length in Angstrom, used to define
        Qmax = pi / amin. Default is 5.

    use_approx : bool, optional
        If True, use the approximate form of the eCWM differential
        roughness factor where implemented. If False, use the more
        complete expression. Default is False.

    Returns
    -------
    eCWM_Psi_R : ndarray
        One-dimensional NumPy array with the same length as qz_space,
        containing the specular roughness factor Psi_R(Qz).

    Notes
    -----
    - In circular mode, the function returns the specular roughness factor
      directly from the circular-resolution expression.
    - In slit mode, the function combines an analytical treatment of the
      singular central region with numerical integration over the remaining
      slit area.
    - The returned quantity is the thermal roughness factor only. It must be
      multiplied by the Fresnel reflectivity and the intrinsic structure
      factor terms to obtain the full specular reflectivity.

    References
    ----------
    Chen Shen, Honghu Zhang, Beate Kloesgen, and Benjamin M. Ocko,
    "Extending the capillary wave model to include the effect of
    bending rigidity: X-ray reflectivity and diffuse scattering",
    Phys. Rev. Research 7, 043016 (2025).

    In particular, see:
    - Eq. (16): definition of the roughness-factor integral
    - Eq. (17): specular reflectivity form
    - Eq. (18): slit-integrated specular roughness factor
    - Eq. (19): circular-resolution specular roughness factor
    """
       
    # ------------------------------------------------------------
    # Input preparation and common eCWM parameters
    # ------------------------------------------------------------
    # ensure 1D numpy arrays (avoid broadcasting issues)
    qz_space = np.asarray(qz_space, dtype=float).ravel()
    #qz_space = np.asarray(qz_space, dtype=float)
    # thermal prefactor (k_B T / gamma) in Å^2 units
    kbT_gamma = kb * temp / tension * 1e20
    # molecular cutoff → maximum in-plane wavevector
    qmax = pi / amin
    # characteristic length scale from bending rigidity
    # (reduces to standard CWM when kappa = 0)
    Lk = np.sqrt(kappa * kb * temp / tension) * 1e10 if kappa != 0 else 0.001
    #Lk = np.maximum(np.sqrt(kappa * kb * temp / tension) * 1e10, 0.001) # [A] safeguard for divide-by-zero. Qk = 1000 when Lk = 0.001 therefore is irelevant 
    eta = (kbT_gamma / (2 * pi)) * qz_space**2
    xi = (2 ** (1 - eta)) * (gamma(1 - 0.5 * eta) / gamma(0.5 * eta)) * 2 * pi / qz_space**2
    
    # ----------------------------
    # validate resolution_mode
    # ----------------------------
    if resolution_mode not in (0, 1):
        raise ValueError("resolution_mode must be 0 or 1")
    # ------------------------------------------------------------
    # Resolution model selection
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Mode 0: circular Qxy resolution (Eq. 19)
    # ------------------------------------------------------------
    if resolution_mode == 0:
        if np.ndim(resolution) != 0:
            raise ValueError(
                "When resolution_mode == 0, circular resolution dQxy_R [1/A], resolution must be a single float."
            )
        resolution = float(resolution)

    # ------------------------------------------------------------
    # Mode 1: rectangular slit resolution (Eq. 18)
    # ------------------------------------------------------------
    elif resolution_mode == 1:
        res = np.asarray(resolution, dtype=float)
        if res.shape != (2,):
            raise ValueError(
                "When resolution_mode == 1, slit resolution [mm], resolution must be a 2-element array "
                "[sl_v_HWHM, sl_h_HWHM]."
            )
        if energy is False or energy is None:
            raise ValueError(
                "When resolution_mode == 1, energy [eV] must be given as a single float."
            )
        if sdd is False or sdd is None:
            raise ValueError(
                "When resolution_mode == 1, sdd [mm] must be given as a single float."
            )

        energy = float(energy)
        sdd = float(sdd)
        resolution = res

    # ----------------------------
    # background settings (NEW API)
    # ----------------------------
    # bkg_mode:
    #   None → no background
    #   0    → horizontal offset (phi direction)
    #   1    → vertical offset (beta direction)

    if (bkg_mode is None) or (resolution_mode == 0):
        # background not used
        bkg_mode_use = None
        bkg_off_use = None

    else:
        if bkg_mode not in (0, 1):
            raise ValueError("bkg_mode must be None, 0, or 1")

        # must be a single float now
        if np.ndim(bkg_off) != 0:
            raise ValueError(
                "When bkg_mode is 0 or 1, bkg_off must be a single float."
            )

        bkg_mode_use = int(bkg_mode)
        bkg_off_use = float(bkg_off)
    
    # ------------------------------------------------------------
    # start calculating 
    # ------------------------------------------------------------
    print("start calculating the specular roughness factor")
    
    # ------------------------------------------------------------
    # Resolution model selection
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Mode 0: circular Qxy resolution (Eq. 19)
    # ------------------------------------------------------------
    if resolution_mode == 0:
        # direct evaluation of the specular roughness factor
        # using a circular integration region in Qxy space
        # (analytical expression, no numerical integration needed)
        r_vals = np.linspace(0.001, 8 * round(Lk), 1000)
        r_grid = np.sqrt(r_vals**2 + amin**2)
        C_integrand = np.zeros((len(qz_space), len(r_vals)))
        for idx, eta_val in enumerate(eta):
            C_integrand[idx, :] = 2 * pi * r_grid**(1 - eta_val) * (np.exp(-eta_val * besselk(0, r_grid / Lk)) - 1)
    
        C = trapezoid(C_integrand, r_vals, axis=1)
        eCWM_Psi_R = ((xi / kbT_gamma) * resolution**eta + resolution**2 * C / (4 * pi)) * (1 / qmax)**eta * np.exp(eta * besselk(0, 1 / (Lk * qmax)))
        print("calculate circular resolution done")
    # ------------------------------------------------------------
    # Mode 1: rectangular slit resolution (Eq. 18)
    # ------------------------------------------------------------
    elif resolution_mode == 1:
        # convert detector slit size (mm) into angular / Q-space acceptance
        # using x-ray energy and sample-detector distance
        wavelength = 12400/energy
        wave_number = 2 * pi / wavelength
        beta = np.degrees(np.arcsin(qz_space / 2 / wave_number))
        beta = beta.reshape(-1, 1) # do this, otherwise beta_xrr has shape (46,) instead of (46, 1) which will mess up xrr_config_phi_array_for_qxy_slit_min and make it (46, 46) instead of (46, 1) like MATLAB code
        alpha = beta # xrr: alpha = beta
        # slit half width in angular space [degrees]
        delta_phi_HW = np.degrees(np.arctan(resolution[1] / sdd / np.cos(np.radians(beta))))
        delta_beta_HW =   np.degrees(np.arcsin(resolution[0] / sdd * np.cos(np.radians(beta))))

        # ------------------------------------------------------------
        # Build slit boundary in detector coordinates
        # h → horizontal (phi direction), v → vertical (beta direction)
        # t: top, b: bottom, l: left, r: right
        # ------------------------------------------------------------        
        slit_h_coord = np.arange(-resolution[1], resolution[1] + 0.005, 0.005)
        slit_v_coord = np.arange(-resolution[0], resolution[0] + 0.005, 0.005)
        # coordinate: two column array, each row (h, v) in mm
        slit_t = np.column_stack((slit_h_coord, np.ones(len(slit_h_coord)) * resolution[0]))
        slit_b = np.column_stack((slit_h_coord, np.ones(len(slit_h_coord)) * -resolution[0]))
        slit_l = np.column_stack((np.ones(len(slit_v_coord)) * -resolution[1], slit_v_coord))
        slit_r = np.column_stack((np.ones(len(slit_v_coord)) * resolution[1], slit_v_coord))
        # put all coordinate into one two-column array
        slit_coord = np.concatenate(
            (slit_t, slit_r, np.flipud(slit_b), np.flipud(slit_l)),
            axis=0
        )
        
        # ------------------------------------------------------------
        # Convert slit edges into Qxy space for each beta
        # This defines the accessible in-plane scattering region
        # qx: transversal, qy: longitudinal
        # ------------------------------------------------------------
        # set the array structure (fill with zero)
        qxy_slit = np.zeros((slit_coord.shape[0], 2, beta.shape[0]))
        qxy_slit_min = np.zeros((beta.shape[0], 1))
        # polar angle within the slit, for ease of Qxy coordinate calculation
        ang = np.arange(0, 2 * pi, 0.01) 
        qxy_slit_min_coord = np.zeros((ang.shape[0], 2, qxy_slit_min.shape[0]))
        # ------------------------------------------------------------
        # Handle singularity at Qxy = 0 (specular condition)
        #
        # The differential roughness factor diverges at Qxy → 0.
        # To avoid numerical instability:
        #   1) find the largest Qxy circle fully inside the slit
        #   2) evaluate that central region analytically
        #   3) integrate only the remaining slit area numerically
        #   4) the remaining slit area is divided into two different regions:
        #      (a) for phi larger than the maixmal phi of the slit, this is a whole rectangular area that tangentes the circle and extends until the slit l/r border
        #      (b) the remaining area on the edge of the circle, and should be divided into a few small rectangular areas to be calculated separately
        # ------------------------------------------------------------
        for idx in range(len(beta)):
            # qxy position of the slit edge: (qx, qy)
            qxy_slit[:, :, idx] = wave_number * np.column_stack([
                slit_coord[:, 0] / sdd,
                slit_coord[:, 1] / sdd * np.sin(np.radians(beta[idx]))
            ])
            # minimal Qxy on the slit edge → radius of the inscribed Qxy circle
            qxy_slit_min[idx, 0] = np.min(
                np.sqrt(qxy_slit[:, 0, idx]**2 + qxy_slit[:, 1, idx]**2)
            )
            # the maximal circular Qxy region inside the slit, defined by the circle that tangentes two edges
            # find its coordinate in q space by polar coordinate (qx, qy)
            qxy_slit_min_coord[:, :, idx] = qxy_slit_min[idx] * np.column_stack([
                np.cos(ang), np.sin(ang)
            ])
        
        # find the maximal phi of the qxy circle. to find the border of the (4a) and (4b)
        phi_max_qxy_slit_min = np.degrees(
            np.arctan(qxy_slit_min / wave_number / np.cos(np.radians(beta)))
        )
        # divide the area (4b) into five pieces by their phi angle
        phi_array_for_qxy_slit_min = phi_max_qxy_slit_min * np.array([0, 1/5, 2/5, 3/5, 4/5])
        # calculate the beta angle for each of this small area on the qxy circle
        # this pair beta and phi gives the corner of one small area on the qxy circle. The other corner is at the slit edge
        # dim 0: beta angle points, dim 1 = 5: phi division (see line above)
        delta_beta_array_for_qxy_slit_min = np.degrees(
            np.arcsin(
                (
                    np.sqrt(
                        np.maximum(
                            qxy_slit_min[:, 0:1]**2
                            - (
                                np.tan(np.radians(phi_array_for_qxy_slit_min))
                                * np.cos(np.radians(beta))
                                * wave_number
                            )**2,
                            0
                        )
                    )
                    / (wave_number * np.sin(np.radians(beta)))
                )
                * np.cos(np.radians(beta))
            )
        )
        
        # delta_beta_HW is (n,1) array due to reshape of beta before. This should be reduced to avoid broadcasting
        delta_beta_HW_1d = delta_beta_HW[:, 0]
        # just to make sure that the circle is not exceeding the slit edge but only equal. Should not happen but can due to accuracy
        for idx in range(delta_beta_array_for_qxy_slit_min.shape[1]):
            repidx = delta_beta_array_for_qxy_slit_min[:, idx] >= delta_beta_HW_1d
            delta_beta_array_for_qxy_slit_min[repidx, idx] = delta_beta_HW_1d[repidx]
        
        # the last border should be included, such taht this array can be directly used as phi upper limit for integration
        phi_array_for_qxy_slit_min = np.hstack([
            phi_array_for_qxy_slit_min,
            phi_max_qxy_slit_min
        ])
        
        # finally, the offset angle position of the background slit center.
        if bkg_mode_use == 0:
            bkg_phi = np.degrees(np.arctan(bkg_off_use / (sdd * np.cos(np.radians(beta)))))
        elif bkg_mode_use ==1:
            bkg_beta_u = beta + np.degrees(np.arctan(bkg_off_use / sdd))
            bkg_beta_l = beta - np.degrees(np.arctan(bkg_off_use / sdd))
        else:
            print('no bkg')
        
        # ------------------------------------------------------------
        # Analytical contribution inside the inscribed Qxy circle
        # This regularizes the singular specular region
        # ------------------------------------------------------------
        qxy_slit_min_flat = qxy_slit_min.flatten()
        Psi_specular_qxy_min = (xi / kbT_gamma) * qxy_slit_min_flat**eta * (1 / qmax)**eta * np.exp(eta * besselk(0, 1 / (Lk * qmax)))
        
        # ------------------------------------------------------------
        # Numerical integration over the remaining slit area
        # outside the central Qxy circle
        # ------------------------------------------------------------
        Psi_region_around_radial_u_r = np.zeros((len(beta), delta_beta_array_for_qxy_slit_min.shape[1]))
        Psi_region_around_radial_l_r = np.zeros((len(beta), delta_beta_array_for_qxy_slit_min.shape[1]))
        Psi_region_outside_phi_max = np.zeros(len(beta))
        Psi_slit_bkgoff = np.zeros(len(beta))
        
        # ------------------------------------------------------------
        # Evaluate slit contribution for each beta independently
        # (parallelized over beta index)
        # variable name of Psi at each beta:
        #   upper_vals: Psi_region_around_radial_u_r(beta)
        #   lower_vals: Psi_region_around_radial_l_r(beta)
        #   out_i: Psi_region_outside_phi_max(beta)
        #   bkgoff_i: Psi_slit_bkgoff(beta)
        # ------------------------------------------------------------
        # pre factor of the differential roughness factor. Taken that out to ensure a better integral accuracy
        diffPsi_prefactor = qz_space**4 /(16* pi**2 * np.sin(np.radians(alpha.ravel())))
        # start evaluating contribution for each beta
        def process_idx_rad(idx):
            beta_i = np.radians(float(beta[idx])) # the diffPsi_red expects beta and phi in radian
            alpha_i_deg = float(alpha[idx]) # the function expects alpha in degree
            # reduced differential roughness factor function
            diff_psi = lambda beta_rad, phi_rad: eCWM_diffPsi_red(
                beta_rad, phi_rad, kbT_gamma, wave_number, alpha_i_deg, Lk, amin, use_approx = use_approx
            )
    
            upper_vals = []
            lower_vals = []
            
            # first evaluate the (4b) area: the surrounding area of the qxy circle that are divided into 5 phi steps
            # the upper and lower side are different due to different beta, therefore are calculated separately
            # the left-right is symmetric therefore only one side is calculated
            for phi_idx in range(delta_beta_array_for_qxy_slit_min.shape[1]):
                # Upper
                upper, _ = dblquad(
                    lambda phi, beta: diff_psi(beta, phi),
                    beta_i + np.radians(delta_beta_array_for_qxy_slit_min[idx, phi_idx]),
                    beta_i + np.radians(delta_beta_HW[idx]),
                    lambda _: np.radians(phi_array_for_qxy_slit_min[idx, phi_idx]),
                    lambda _: np.radians(phi_array_for_qxy_slit_min[idx, phi_idx + 1]),
                    epsabs=1e-12, epsrel=1e-10
                )
                upper_vals.append(upper*diffPsi_prefactor[idx])
                # Lower
                lower, _ = dblquad(
                    lambda phi, beta: diff_psi(beta, phi),
                    beta_i - np.radians(delta_beta_HW[idx]),
                    beta_i - np.radians(delta_beta_array_for_qxy_slit_min[idx, phi_idx]),
                    lambda _: np.radians(phi_array_for_qxy_slit_min[idx, phi_idx]),
                    lambda _: np.radians(phi_array_for_qxy_slit_min[idx, phi_idx + 1]),
                    epsabs=1e-12, epsrel=1e-10
                )
                lower_vals.append(lower*diffPsi_prefactor[idx])
    
            # the rectangular region outside of the phi max of the qxy circle till the slit edge
            result, _ = dblquad(
                func=diff_psi,
                a=np.radians(phi_max_qxy_slit_min[idx]),
                b=np.radians(delta_phi_HW[idx]),
                gfun=lambda _: beta_i - np.radians(delta_beta_HW[idx]),
                hfun=lambda _: beta_i + np.radians(delta_beta_HW[idx]),
                epsabs=1e-8, epsrel=1e-6
            )
            out_i = result*diffPsi_prefactor[idx]
            
            # --------------------------------------------------------
            # Optional off-specular background subtraction
            #
            # bkg_mode:
            #   None → no background
            #   0    → horizontal slit offset (phi direction)
            #   1    → vertical slit offset (beta direction)
            # --------------------------------------------------------
            if bkg_mode_use == 0:
                print('bkg by offset phi left and right')
                result2, _ = dblquad(
                    func=diff_psi,
                    a=np.radians(bkg_phi[idx] - delta_phi_HW[idx]),
                    b=np.radians(bkg_phi[idx] + delta_phi_HW[idx]),
                    gfun=lambda _: beta_i - np.radians(delta_beta_HW[idx]),
                    hfun=lambda _: beta_i + np.radians(delta_beta_HW[idx]),
                    epsabs=1e-8, epsrel=1e-6
                )
                bkgoff_i = result2*diffPsi_prefactor[idx]
            elif bkg_mode_use == 1:
                print('bkg by offset beta up and down')
                result2u, _ = dblquad(
                    func=diff_psi,
                    a= -np.radians(delta_phi_HW[idx]),
                    b= np.radians(delta_phi_HW[idx]),
                    gfun=lambda _: np.radians(bkg_beta_u[idx] - delta_beta_HW[idx]),
                    hfun=lambda _: np.radians(bkg_beta_u[idx] + delta_beta_HW[idx]),
                    epsabs=1e-8, epsrel=1e-6
                )
                result2l, _ = dblquad(
                    func=diff_psi,
                    a= -np.radians(delta_phi_HW[idx]),
                    b= np.radians(delta_phi_HW[idx]),
                    gfun=lambda _: np.radians(bkg_beta_l[idx] - delta_beta_HW[idx]),
                    hfun=lambda _: np.radians(bkg_beta_l[idx] + delta_beta_HW[idx]),
                    epsabs=1e-8, epsrel=1e-6
                )
                bkgoff_i = (result2u + result2l)/2 *diffPsi_prefactor[idx]
            else:
                print('no bkg')
                bkgoff_i = 0.0
    
            upper_vals = np.array(upper_vals, dtype=np.float64).flatten()
            lower_vals = np.array(lower_vals, dtype=np.float64).flatten()
    
            return idx, upper_vals, lower_vals, out_i, bkgoff_i
    
        parallel_results  = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_idx_rad)(i) for i in range(len(beta))
        )
        
        # ------------------------------------------------------------
        # Collect parallel results and combine analytical + numerical parts
        # ------------------------------------------------------------
        for idx, upper_vals, lower_vals, out_i, bkgoff_i in parallel_results:
            Psi_region_around_radial_u_r[idx, :] = upper_vals
            Psi_region_around_radial_l_r[idx, :] = lower_vals
            Psi_region_outside_phi_max[idx] = out_i
            Psi_slit_bkgoff[idx] = bkgoff_i
        # ------------------------------------------------------------
        # total specular slit-integrated roughness factor
        # note: Psi_slit_bkgoff is already done above
        # ------------------------------------------------------------
        # within the specular slit
        Psi_slit_SP = Psi_specular_qxy_min + 2 * (
            np.sum(Psi_region_around_radial_u_r + Psi_region_around_radial_l_r, axis=1)
            + Psi_region_outside_phi_max
        )
        # background subtracted roughness factor
        eCWM_Psi_R = Psi_slit_SP - Psi_slit_bkgoff
        print("calculate slit resolution done")
    # integrated specular roughness factor Psi_R(Qz)
    # includes the finite detector resolution around the specular condition
    return eCWM_Psi_R

    

def calc_eCWM_red_r(beta_space, 
                    phi, 
                    alpha=None,                    
                    energy=None,  
                    DSphi_HWHM=None, 
                    DSbeta_HWHM=None,
                    R_resolution_mode=0,
                    R_resolution=0.0002,
                    R_energy = None,
                    R_sdd = 1000,
                    R_bkg_mode=None,
                    R_bkg_off= 1,
                    tension=0.073, 
                    temp=295, 
                    kappa=0, 
                    amin=3.1, 
                    use_approx=False, 
                    show_plot=True):
    """
    Calculate the reduced ratio r_red = Psi_DS / Psi_R of diffuse and specular roughness factors.

    This function computes the ratio between the diffuse-scattering roughness
    factor Psi_DS(Qz, Qxy0) and the specular roughness factor Psi_R(Qz)
    according to the extended capillary wave model (eCWM):

        r_red(Qz, Qxy0) = Psi_DS(Qz, Qxy0) / Psi_R(Qz)

    This quantity is referred to here as the *reduced r*, in contrast to the
    r defined in Eq. (28) of the paper. The latter includes additional
    prefactors such as the Fresnel reflectivity, transmission coefficients,
    and intrinsic structure factor, whereas r_red contains only the thermal
    roughness contribution.

    Physically, r_red isolates the effect of capillary-wave roughness (including
    bending rigidity kappa) on the ratio between diffuse scattering intensity
    (R*) and specular reflectivity (R), independent of optical factors.

    The function internally computes:
    - Psi_DS(Qz, Qxy0): diffuse roughness factor using
      calc_eCWM_roughness_factor_DS()
    - Psi_R(Qz): specular roughness factor: 
        circular or slit (w/o or with bkg subtraction)
        default: using circular Qxy resolution
        (calc_eCWM_roughness_factor_SP with resolution_mode = 0)
    
    Parameters
    ----------
    beta_space : array-like
        One-dimensional array of exit angles beta in degrees.

    phi : float
        In-plane angular offset (in degrees) defining the diffuse scattering
        position (Qxy0). Must be a single scalar.
    
    -- keyword arguments --    
    alpha : float
        Incident angle in degrees. Must be a single scalar.
            
    energy : float
        X-ray energy in eV.

    DSphi_HWHM : float
        Half-width (HWHM) of the detector acceptance in phi (degrees) for
        diffuse scattering.

    DSbeta_HWHM : float
        Half-width (HWHM) of the detector acceptance in beta (degrees)
        for diffuse scattering.

    tension : float
        Surface tension gamma in N/m.

    temp : float
        Temperature in K.

    kappa : float
        Bending rigidity in units of k_B T.

    amin : float
        Molecular cutoff length in Å, used to define Qmax = pi / amin.
    
    R_resolution_mode : int, optional
        Selects the reflectivity resolution description:
        - 0 : circular Qxy resolution in reciprocal space
        - 1 : rectangular slit resolution in reflectometry detector space
        Default is 0.

    R_resolution : float or array-like
        reflectivity resolution parameter, interpreted according to R_resolution_mode.

        If R_resolution_mode == 0:
            Single float giving the circular Qxy half width
            dQxy_R [1/Angstrom].

        If R_resolution_mode == 1:
            Two-element array-like [slit_v_HWHM, slit_h_HWHM] in mm,
            giving the reflectometry detector slit half widths in the vertical 
            and horizontal directions.
    
    R_energy : float, optional
        reflectometry energy in eV. Required when R_resolution_mode == 1.
        Not used when R_resolution_mode == 0.

    R_sdd : float, optional
        reflectometry sample-to-detector distance in mm. Required when
        resolution_mode == 1. Default is 1000.
    
    R_bkg_mode : None or int, optional
        reflectivity background subtraction mode, only relevant when
        R_resolution_mode == 1.
        - None : no background subtraction
        - 0    : horizontal reflectometry slit offset
        - 1    : vertical reflectometry slit offset
        Default is None.

    R_bkg_off : float, optional
        reflectometry background slit offset in mm when R_bkg_mode is 0 or 1.
        Ignored when bkg_mode is None or when R_resolution_mode == 0.
        Default is 1.

    use_approx : bool, optional
        If True, use the approximate form of the eCWM differential roughness
        factor. If False, use the more accurate expression. Default is False.

    show_plot : bool, optional
        If True, plot the normalized reduced r as a function of Qz.
        Default is True.

    Returns
    -------
    r_red : ndarray
        One-dimensional array containing the reduced roughness-factor ratio
        r_red(Qz, Qxy0) for each beta value.

    eCWM_Psi_DS : ndarray
        Diffuse roughness factor Psi_DS(Qz, Qxy0).

    eCWM_Psi_R : ndarray
        Specular roughness factor Psi_R(Qz).

    Notes
    -----
    - The momentum transfer components are:
          Qz  = k0 * (sin(alpha) + sin(beta))
          Qxy0 = 2*k0*sin(phi/2)
      where k0 = 2*pi/lambda.

    - The reduced r isolates the thermal roughness contribution and removes
      the Fresnel reflectivity and transmission coefficients that appear in
      the full scattering intensity.

    - This quantity is useful for directly comparing diffuse and specular
      scattering within the eCWM framework without additional optical factors.

    References
    ----------
    Chen Shen, Honghu Zhang, Beate Kloesgen, and Benjamin M. Ocko,
    "Extending the capillary wave model to include the effect of
    bending rigidity: X-ray reflectivity and diffuse scattering",
    Phys. Rev. Research 7, 043016 (2025).

    See:
    - Eq. (16): definition of the roughness factor
    - Eq. (17): specular reflectivity
    - Eq. (18): angular integration
    - Eq. (28): definition of r (full expression, not reduced)

    Important
    ---------
    This function returns only the ratio of eCWM roughness factors,
    Psi_DS / Psi_R. It does not include the Fresnel reflectivity,
    transmission coefficients, or other prefactors appearing in the
    full intensity ratio r of Eq. (28).
    
    """
    # ------------------------------------------------------------
    # Validate required inputs for diffuse scattering calculation
    # ------------------------------------------------------------
    required_params = {
        "alpha": alpha,
        "energy": energy,
        "DSphi_HWHM": DSphi_HWHM,
        "DSbeta_HWHM": DSbeta_HWHM,
    }
    
    for name, value in required_params.items():
        if value is None:
            raise ValueError(f"{name} must be provided for diffuse scattering calculation.")
    
        # check scalar (not array/list)
        if np.ndim(value) != 0:
            raise ValueError(f"{name} must be a scalar (float), not an array.")
    
        # try converting to float
        try:
            required_params[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a float (or convertible to float).")
    
    # overwrite with validated values
    alpha = required_params["alpha"]
    energy = required_params["energy"]
    DSphi_HWHM = required_params["DSphi_HWHM"]
    DSbeta_HWHM = required_params["DSbeta_HWHM"]
    
    # ----------------------------
    # validate resolution_mode
    # ----------------------------
    if R_resolution_mode not in (0, 1):
        raise ValueError("resolution_mode must be 0 or 1")
    # ------------------------------------------------------------
    # Resolution model selection
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Mode 0: circular Qxy resolution (Eq. 19)
    # ------------------------------------------------------------
    if R_resolution_mode == 0:
        if np.ndim(R_resolution) != 0:
            raise ValueError(
                "When R_resolution_mode == 0, circular resolution dQxy_R [1/A], resolution must be a single float."
            )
        R_resolution = float(R_resolution)

    # ------------------------------------------------------------
    # Mode 1: rectangular slit resolution (Eq. 18)
    # ------------------------------------------------------------
    elif R_resolution_mode == 1:
        R_res = np.asarray(R_resolution, dtype=float)
        if R_res.shape != (2,):
            raise ValueError(
                "When R_resolution_mode == 1, slit resolution [mm], resolution must be a 2-element array "
                "[sl_v_HWHM, sl_h_HWHM]."
            )
        if R_energy is False or R_energy is None:
            raise ValueError(
                "When R_resolution_mode == 1, reflectivity energy [eV] must be given as a single float."
            )
        if R_sdd is False or R_sdd is None:
            raise ValueError(
                "When R_resolution_mode == 1, reflectivity sdd [mm] must be given as a single float."
            )

        R_energy = float(R_energy)
        R_sdd = float(R_sdd)
        R_resolution = R_res

    # ----------------------------
    # background settings (NEW API)
    # ----------------------------
    # R_bkg_mode:
    #   None → no background
    #   0    → horizontal offset (phi direction)
    #   1    → vertical offset (beta direction)

    if (R_bkg_mode is None) or (R_resolution_mode == 0):
        # background not used
        R_bkg_mode_use = None
        R_bkg_off_use = None

    else:
        if R_bkg_mode not in (0, 1):
            raise ValueError("R_bkg_mode must be None, 0, or 1")

        # must be a single float now
        if np.ndim(R_bkg_off) != 0:
            raise ValueError(
                "When R_bkg_mode is 0 or 1, R_bkg_off must be a single float."
            )
        R_bkg_mode_use = int(R_bkg_mode)
        R_bkg_off_use = float(R_bkg_off)
    
    wavelength = 12400.0 / energy
    wave_number = 2 * pi / wavelength
    qz_space = (np.sin(np.radians(alpha)) + np.sin(np.radians(beta_space))) * wave_number
    qxy0 = 2*wave_number*np.sin(np.radians(phi)/2)
    
    eCWM_Psi_DS = calc_eCWM_roughness_factor_DS(alpha, beta_space, phi, 
                                                energy=energy, DSphi_HWHM=DSphi_HWHM, DSbeta_HWHM=DSbeta_HWHM,
                                                tension=tension, temp=temp, kappa=kappa, amin=amin, 
                                                use_approx=use_approx)
    eCWM_Psi_R = calc_eCWM_roughness_factor_SP(qz_space, 
                                               energy=R_energy, sdd=R_sdd,
                                               resolution_mode=R_resolution_mode, resolution=R_resolution, 
                                               bkg_mode=R_bkg_mode_use, bkg_off=R_bkg_off_use, 
                                               tension=tension, temp=temp, kappa=kappa, amin=amin)
    # reduced r: ratio of roughness factors only
    # does not include Fresnel reflectivity, transmission coefficients,
    # or intrinsic structure-factor terms from the full intensity ratio
    r_red = eCWM_Psi_DS / eCWM_Psi_R
    
    if show_plot:
        label_mode = "Approx" if use_approx else "Accurate"
        plt.figure(figsize=(8, 5))
        plt.plot(qz_space, r_red / r_red[0], label=f"{label_mode} Qxy₀={qxy0:.3f} Å⁻¹", linewidth=1.5)
        plt.xlabel(r"$Q_z$ [$\AA^{-1}$]", fontsize=12)
        plt.ylabel(r"R^{*} / (R/R$_F$)", fontsize=12)
        plt.xlim(0, 1.2)
        plt.grid(True)
        plt.legend(loc="upper left", frameon=False)
        plt.title(f"r ({label_mode})")
        plt.tight_layout()
        plt.show()
    
    return r_red, eCWM_Psi_DS, eCWM_Psi_R


# ---------------------------------------------------------------
# surface scattering optics
# ---------------------------------------------------------------

def calc_fresnel(Qz, Qc):    
    """
    Calculate the Fresnel reflectivity for a given Qz.
    
    This function evaluates the Fresnel reflectivity R_F(Qz) for an ideal,
    flat interface using the standard optical expression for the reflection
    coefficient.
    
    The calculation supports complex values of Qz internally, allowing
    correct handling below the critical angle where total external reflection
    occurs.
    
    Parameters
    ----------
    Qz : array-like
        Momentum transfer perpendicular to the surface [1/Å].
    
    Qc : float
        Critical momentum transfer [1/Å], related to the electron density
        contrast of the interface.
    
    Returns
    -------
    result : ndarray
        Two-column array:
        - column 0: Qz values (real part)
        - column 1: Fresnel reflectivity R_F(Qz)
    
    Notes
    -----
    - The reflectivity is calculated as |r|^2, where r is the Fresnel
      reflection coefficient.
    - For Qz < Qc, the square root becomes complex, corresponding to total
      reflection.
    
    """
    Qz = np.asarray(Qz, dtype=np.complex128)  # allow complex arithmetic
    sqrt_term = np.sqrt(Qz**2 - Qc**2)        # may be complex when Qz < Qc
    r = (Qz - sqrt_term) / (Qz + sqrt_term)   # reflection coefficient
    refl = np.abs(r)**2                       # reflectivity (real-valued)
    return np.column_stack((Qz.real, refl))   # return Qz as real part only


def GIXOS_dQz(Qz, energy_eV, alpha_deg, Ddet_mm, footprint_mm):
    """
    Calculate Qz resolution broadening due to beam footprint.

    This function evaluates the broadening of Qz caused by the finite beam
    footprint on the sample surface in a GIXOS (grazing-incidence x-ray
    off-specular scattering) geometry.

    The footprint leads to an angular spread in the exit angle beta,
    which translates into a spread in Qz.

    Parameters
    ----------
    Qz : ndarray
        Array of shape (n,1) containing Qz values [1/Å].

    energy_eV : float
        X-ray energy in eV.

    alpha_deg : float
        Incident angle in degrees.

    Ddet_mm : float
        Sample-to-detector distance in mm.

    footprint_mm : float
        Beam footprint size on the sample in mm.

    Returns
    -------
    dQz : ndarray
        Array of shape (n,6) with columns:
        - column 0: Qz
        - column 1: central exit angle beta (deg)
        - column 2: maximum exit angle beta_max (deg)
        - column 3: minimum exit angle beta_min (deg)
        - column 4: delta_Qz (half-width of Qz broadening)
        - column 5: relative broadening delta_Qz / Qz

    Notes
    -----
    - The calculation assumes geometrical broadening from footprint effects.
    - The angular spread is converted into Qz spread using standard
      kinematic relations.

    """
    planck = 12400  # eV·A
    wavelength = planck / energy_eV  # Å

    Qz = np.asarray(Qz).reshape(-1, 1)
    # Qz should always be a column vector
    dQz = np.zeros((Qz.shape[0], 6)) # change np.zeros((Qz.shape[0], 5)) to np.zeros((Qz.shape[0], 6)) to match MATLAB output and produce 6 columns
    dQz[:, 0] = Qz[:, 0]
    
    alpha_rad = np.radians(alpha_deg)
    beta_center = np.degrees(np.arcsin(Qz[:, 0] * wavelength / (2 * pi) - np.sin(alpha_rad)))
    beta_max = np.degrees(np.arctan(np.tan(np.radians(beta_center)) * Ddet_mm / (Ddet_mm - footprint_mm)))
    beta_min = np.degrees(np.arctan(np.tan(np.radians(beta_center)) * Ddet_mm / (Ddet_mm + footprint_mm)))

    factor = (2 * pi) / wavelength
    qz_max = (np.sin(np.radians(beta_max)) + np.sin(alpha_rad)) * factor
    qz_min = (np.sin(np.radians(beta_min)) + np.sin(alpha_rad)) * factor
    delta_qz = 0.5 * (qz_max - qz_min)

    dQz[:, 1] = beta_center
    dQz[:, 2] = beta_max
    dQz[:, 3] = beta_min
    dQz[:, 4] = delta_qz
    dQz[:, 5] = dQz[:, 4] / dQz[:, 0] # added to match MATLAB output and create new column

    return dQz


def t_sqr(angle_deg, energy_eV, qc = 0.0218, beta = 1e-9):
    """
    Calculate the transmission coefficient squared |t|^2.

    This function evaluates the squared transmission coefficient for x-rays
    incident on a surface, based on the Fresnel transmission amplitude.

    It accounts for absorption via a small imaginary component beta in the
    refractive index.

    Parameters
    ----------
    angle_deg : float or array-like
        Incident or exit angle in degrees.

    energy_eV : float
        X-ray energy in eV.

    qc : float, optional
        Critical momentum transfer [1/Å]. Default corresponds to water.

    beta : float, optional
        Imaginary part of refractive index (absorption term).
        Default is 1e-9.

    Returns
    -------
    T : ndarray or float
        Transmission coefficient squared |t|^2.

    Notes
    -----
    - For angles below the critical angle, transmission is suppressed.
    - The function supports both scalar and array inputs.

    """
    planck = 12400  # eV·A
    wavelength = planck / energy_eV  # Å
    alpha_c_rad = np.arcsin(qc / (2 * 2 * pi / wavelength))
    angle_rad = np.radians(angle_deg)
    x = angle_rad / alpha_c_rad
    # Handle both scalar and array cases
    T = np.zeros_like(x, dtype=np.float64)
    mask = x > 0
    if np.any(mask):
        T[mask] = np.abs(2 * x[mask] / (x[mask] + np.sqrt(x[mask]**2 - 1 - 2j * beta / alpha_c_rad**2)))**2

    return T


def ave_tbeta_sqr(beta_c_deg, footprint_mm, energy_eV, Ddet_mm, qc=0.0218):
    """
    Calculate averaged transmission coefficient squared over footprint.

    This function evaluates the average transmission coefficient squared
    |t_beta|^2 over the illuminated footprint on the detector, taking into
    account the angular variation caused by finite beam size.

    Parameters
    ----------
    beta_c_deg : float or array-like
        Central exit angle beta in degrees.

    footprint_mm : float
        Beam footprint size in mm.

    energy_eV : float
        X-ray energy in eV.

    Ddet_mm : float
        Sample-to-detector distance in mm.

    qc : float, optional
        Critical momentum transfer [1/Å]. Default corresponds to water.

    Returns
    -------
    result : float or ndarray
        Averaged transmission coefficient squared |t_beta|^2.
        Returns a scalar for scalar input and a 1D array for array input.
    """
    scalar_input = np.ndim(beta_c_deg) == 0
    beta_c = np.asarray(beta_c_deg, dtype=float).ravel()   # shape (n,)

    step = int(np.floor(footprint_mm / 5))
    offsets = np.linspace(-5 * step / 2, 5 * step / 2, step + 1, dtype=float)  # shape (m,)

    # Broadcast to shape (m, n):
    # rows = footprint offsets, columns = beta points
    offset_grid = offsets[:, None]
    beta_grid = beta_c[None, :]

    beta_rad = np.arctan(
        (Ddet_mm * np.tan(np.radians(beta_grid))) / (Ddet_mm - offset_grid)
    )
    beta_deg = np.degrees(beta_rad)

    tbeta_sqr_vals = t_sqr(beta_deg, energy_eV, qc=qc)   # shape (m, n)
    result = np.mean(tbeta_sqr_vals, axis=0)             # shape (n,)

    return float(result[0]) if scalar_input else result


def calc_tbeta_sqr(beta_array, qc, energy_eV, alpha_i_deg, Ddet_mm, footprint_mm):
    """
    Calculate transmission-related quantities as a function of beta.

    This function computes several quantities related to the transmission
    coefficient for a range of exit angles beta.

    The output includes Qz, beta, normalized beta, and the averaged
    transmission coefficient squared.

    Parameters
    ----------
    beta_array : array-like
        One-dimensional array of exit angles beta in degrees.

    qc : float
        Critical momentum transfer [1/Å].

    energy_eV : float
        X-ray energy in eV.

    alpha_i_deg : float
        Incident angle in degrees.

    Ddet_mm : float
        Sample-to-detector distance in mm.

    footprint_mm : float
        Beam footprint size in mm.

    Returns
    -------
    tsqr : ndarray
        Array of shape (n,4) with columns:
        - column 0: Qz [1/Å]
        - column 1: beta (deg)
        - column 2: beta / alpha_c (dimensionless)
        - column 3: averaged |t_beta|^2
    """
    planck = 12400.0
    wavelength = planck / energy_eV  # Å

    # force 1D input so output is always (n, 4)
    beta_array = np.asarray(beta_array, dtype=float).ravel()

    # tsqr has four columns: qz, beta, beta/alpha_c, |t_beta|^2
    tsqr = np.zeros((beta_array.shape[0], 4), dtype=float)

    tsqr[:, 0] = (2 * pi / wavelength) * (
        np.sin(np.radians(alpha_i_deg)) + np.sin(np.radians(beta_array))
    )

    alpha_c = np.degrees(np.arcsin(qc / (2 * 2 * pi / wavelength)))

    tsqr[:, 1] = beta_array
    tsqr[:, 2] = beta_array / alpha_c
    tsqr[:, 3] = ave_tbeta_sqr(beta_array, footprint_mm, energy_eV, Ddet_mm, qc=qc)

    return tsqr