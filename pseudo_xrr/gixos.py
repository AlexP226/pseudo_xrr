# ~~~  Chen Shen Functions: ~~~     # can paste functions into a separate py file and import
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import trapezoid, simpson
from scipy.special import kv as besselk, jv as besselj, gamma
from scipy.constants import pi, Boltzmann as kb
#from numba import njit
#@jit(nopython=True, parallel=True)  # Use Numba for performance optimization
from scipy.integrate import quad # ChatGPT contribution to replace the integration with quad for better performance


def film_correlation_integrand_replacement(r, qxy, eta, Lk, amin): # Changes made
    r = np.asarray(r)  # Ensure r is a numpy array
    rad_term = np.sqrt(r[None, None, :]**2 + amin**2)  # r turned into r[None, None, :] b/c sizing errors when integrating and the [..., None] at the end was removed
    term1 = rad_term**(1 - eta[..., None])
    term2 = np.exp(-eta[..., None] * besselk(0, rad_term / Lk)) - 1
    term3 = besselj(0, rad_term * qxy[..., None])
    return term1 * term2 * term3




def film_integral_delta_beta_delta_phi(beta, phi, kbT_gamma, wave_number, alpha, Lk, amin):   # error arrises here
    beta = np.asarray(beta) # converting to numpy array for performance/vectorization
    phi = np.asarray(phi)


    qmax = pi / amin
    beta_rad = np.radians(beta)
    phi_rad = np.radians(phi)
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

    r_vals = np.linspace(0.001, 8 * Lk, 300)
    integrand_vals = film_correlation_integrand_replacement(r_vals, qxy, eta, Lk, amin)
    integral_vals = trapezoid(integrand_vals, r_vals, axis=-1) # might need axis = 1
    C_prime = 2 * pi * integral_vals

    xi = 2 ** (1 - eta) * gamma(1 - 0.5 * eta) / gamma(0.5 * eta) * (2 * np.pi) / (qz ** 2)    # xi used to = (2 * Lk) ** eta, but in MATLAB looks like: xi = (2.^(1-eta).*gamma(1-0.5*eta)./gamma(0.5*eta)) *2*pi./qz.^2;

    exp_term = np.exp(eta * besselk(0, 1 / (Lk * qmax)))
    result = (xi * qxy ** (eta - 2) + C_prime / qz**2) * (1 / qmax) ** eta * exp_term
    return result


def film_integral_approx_delta_beta_delta_phi(beta, phi, kbT_gamma, wave_number, alpha, Lk, amin):
    qmax = pi / amin
    beta_rad = np.radians(beta)
    phi_rad = np.radians(phi)
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
    #   

    result = kbT_gamma * (1 / qmax)**eta * np.exp(eta * besselk(0, safe_besselk_arg)) * qxy**eta / (qxy**2 + (Lk**2) * qxy**4)
    return result


# -------------------- Main Calculation -------------------- #

def calc_film_DS_RRF_integ(beta_space, qxy0, energy, alpha, Rqxy_HWHM, DSqxy_HWHM, DSbeta_HWHM,
                           tension, temp, kapa, amin, use_approx=False, show_plot=True):
    qmax = pi / amin
    wavelength = 12.4 / energy
    wave_number = 2 * pi / wavelength
    qz_space = (np.sin(np.radians(alpha)) + np.sin(np.radians(beta_space))) * wave_number

    phi = 2 * np.degrees(np.arcsin(qxy0 / (2 * wave_number)))
    phi_HWHM = np.degrees(DSqxy_HWHM * wavelength / (2 * pi))
    phi_upper = phi + phi_HWHM
    phi_lower = phi - phi_HWHM

    beta_upper = beta_space + DSbeta_HWHM
    beta_lower = beta_space - DSbeta_HWHM

    kbT_gamma = kb * temp / tension * 1e20
    Lk = np.sqrt(kapa * kb * temp / tension) * 1e10
    eta = (kbT_gamma / (2 * pi)) * qz_space**2

    xi = (2 ** (1 - eta)) * (gamma(1 - 0.5 * eta) / gamma(0.5 * eta)) * 2 * pi / qz_space**2

    r_vals = np.linspace(0.001, 8 * round(Lk), 1000)
    r_grid = np.sqrt(r_vals**2 + amin**2)
    C_integrand = np.zeros((len(qz_space), len(r_vals)))

    for idx, eta_val in enumerate(eta):
        C_integrand[idx, :] = 2 * pi * r_grid**(1 - eta_val) * (np.exp(-eta_val * besselk(0, r_grid / Lk)) - 1)

    C = trapezoid(C_integrand, r_vals, axis=1)
    RRF_term = ((xi / kbT_gamma) * Rqxy_HWHM**eta + Rqxy_HWHM**2 * C / (4 * pi)) * (1 / qmax)**eta * np.exp(eta * besselk(0, 1 / (Lk * qmax)))

    DS_term = np.zeros(len(beta_space))
    phi_grid = np.linspace(phi_lower, phi_upper, 100)

    for idx, beta in enumerate(beta_space):
        beta_grid = np.linspace(beta_lower[idx], beta_upper[idx], 100)
        beta_mesh, phi_mesh = np.meshgrid(beta_grid, phi_grid, indexing='ij')

        if use_approx:
            vals = film_integral_approx_delta_beta_delta_phi(beta_mesh, phi_mesh, kbT_gamma, wave_number, alpha, Lk, amin)
        else:
            vals = film_integral_delta_beta_delta_phi(beta_mesh, phi_mesh, kbT_gamma, wave_number, alpha, Lk, amin)

        DS_term[idx] = simpson(simpson(vals, phi_grid), beta_grid)

    DS_RRF = DS_term / RRF_term

    if show_plot:
        label_mode = "Approx" if use_approx else "Accurate"
        plt.figure(figsize=(8, 5))
        plt.plot(qz_space, DS_RRF / DS_RRF[0], label=f"{label_mode} Qxy₀={qxy0:.3f} Å⁻¹", linewidth=1.5)
        plt.xlabel(r"$Q_z$ [$\AA^{-1}$]", fontsize=12)
        plt.ylabel(r"R^{*} / (R/R$_F$)", fontsize=12)
        plt.xlim(0, 1.2)
        plt.grid(True)
        plt.legend(loc="upper left", frameon=False)
        plt.title(f"GIXOS factor ({label_mode})")
        plt.tight_layout()
        plt.show()

    return DS_RRF, DS_term, RRF_term

# -------------------- CLI -------------------- #

def GIXOS_fresnel(Qz, Qc):    # apparently does not limit to 1 like MATLAB code does - see AI
    Qz = np.asarray(Qz, dtype=np.complex128)  # allow complex arithmetic
    sqrt_term = np.sqrt(Qz**2 - Qc**2)        # may be complex when Qz < Qc
    r = (Qz - sqrt_term) / (Qz + sqrt_term)   # reflection coefficient
    refl = np.abs(r)**2                       # reflectivity (real-valued)
    return np.column_stack((Qz.real, refl))   # return Qz as real part only


def GIXOS_dQz(Qz, energy_eV, alpha_i_deg, Ddet_mm, footprint_mm):
    """
    footprint induced dQz resolution broadening
    """
    planck = 1240.4  # eV·nm
    wavelength = planck / energy_eV * 10  # Å

    # Qz = np.asarray(Qz).reshape(-1, 1)
    # Qz should always be a column vector
    dQz = np.zeros((Qz.shape[0], 6)) # change np.zeros((Qz.shape[0], 5)) to np.zeros((Qz.shape[0], 6)) to match MATLAB output and produce 6 columns
    dQz[:, 0] = Qz[:, 0]

    alpha_i_rad = np.radians(alpha_i_deg)
    alpha_f_center = np.degrees(np.arcsin(Qz[:, 0] * wavelength / (2 * np.pi) - np.sin(alpha_i_rad)))
    alpha_f_max = np.degrees(np.arctan(np.tan(np.radians(alpha_f_center)) * Ddet_mm / (Ddet_mm - footprint_mm)))
    alpha_f_min = np.degrees(np.arctan(np.tan(np.radians(alpha_f_center)) * Ddet_mm / (Ddet_mm + footprint_mm)))

    factor = (2 * np.pi) / wavelength
    qz_max = (np.sin(np.radians(alpha_f_max)) + np.sin(alpha_i_rad)) * factor
    qz_min = (np.sin(np.radians(alpha_f_min)) + np.sin(alpha_i_rad)) * factor
    delta_qz = 0.5 * (qz_max - qz_min)

    dQz[:, 1] = alpha_f_center
    dQz[:, 2] = alpha_f_max
    dQz[:, 3] = alpha_f_min
    dQz[:, 4] = delta_qz
    dQz[:, 5] = dQz[:, 4] / dQz[:, 0] # added to match MATLAB output and create new column

    return dQz


def vineyard_factor(alpha_f_deg, energy_eV, alpha_i_deg, qc = 0.0218, beta = 1e-9):
    import numpy as np
    """
    modified by Chen
    move qc and beta as an argument, and beta by default using water value
    """
    planck = 1240.4  # eV·nm
    wavelength = planck / energy_eV * 10  # Å
    alpha_c = np.arcsin(qc / (2 * 2 * np.pi / wavelength))

    alpha_i_rad = np.radians(alpha_i_deg)
    alpha_f_rad = np.radians(alpha_f_deg)

    li_term = (alpha_c**2 - alpha_i_rad**2)**2 + (2 * beta)**2
    l_i = 1 / np.sqrt(2) * np.sqrt(alpha_c**2 - alpha_i_rad**2 + np.sqrt(li_term))

    x = alpha_f_deg / np.degrees(alpha_c)

    # Handle both scalar and array cases
    T = np.zeros_like(x, dtype=np.float64)
    l_f = np.zeros_like(x, dtype=np.float64)
    mask = x > 0
    if np.any(mask):
        T[mask] = np.abs(2 * x[mask] / (x[mask] + np.sqrt(x[mask]**2 - 1 - 2j * beta / alpha_c**2)))**2
        lf_term = (alpha_c**2 - alpha_f_rad[mask]**2)**2 + (2 * beta)**2
        l_f[mask] = 1 / np.sqrt(2) * np.sqrt(alpha_c**2 - alpha_f_rad[mask]**2 + np.sqrt(lf_term))

    normalization = wavelength / (2 * np.pi) / l_i
    vf = (wavelength / (2 * np.pi)) * T / (l_f + l_i) / normalization
    return vf


def vf_length_corr(alpha_fc_deg, length_mm, energy_eV, alpha_i_deg, Ddet_mm):
    tan_alpha_fc = np.tan(np.radians(alpha_fc_deg))
    alpha_f_rad = np.arctan((Ddet_mm * tan_alpha_fc) / (Ddet_mm - length_mm))
    alpha_f_deg = np.degrees(alpha_f_rad)
    return vineyard_factor(alpha_f_deg, energy_eV, alpha_i_deg)


def ave_vf(alpha_fc_deg, footprint_mm, energy_eV, alpha_i_deg, Ddet_mm):
    step = int(np.floor(footprint_mm / 5))
    offsets = np.linspace(-5 * step / 2, 5 * step / 2, step + 1)
    offsets = offsets[:, np.newaxis] if np.ndim(alpha_fc_deg) > 0 else offsets
    alpha_fc = np.asarray(alpha_fc_deg)
    alpha_fc = alpha_fc[np.newaxis, :] if alpha_fc.ndim == 1 else alpha_fc

    # Broadcast offsets with alpha_fc
    offset_grid, alpha_grid = np.meshgrid(offsets.squeeze(), alpha_fc.squeeze(), indexing='ij')
    alpha_f_rad = np.arctan((Ddet_mm * np.tan(np.radians(alpha_grid))) / (Ddet_mm - offset_grid))
    alpha_f_deg = np.degrees(alpha_f_rad)
    vf_vals = vineyard_factor(alpha_f_deg, energy_eV, alpha_i_deg)
    return np.mean(vf_vals, axis=0) if vf_vals.ndim > 1 else np.mean(vf_vals)


def GIXOS_Tsqr(Qz_array, Qc, energy_eV, alpha_i_deg, Ddet_mm, footprint_mm):
    planck = 1240.4
    wavelength = planck / energy_eV * 10  # [Å]
    Qz_array = np.atleast_2d(Qz_array)
    Tsqr = np.zeros((Qz_array.shape[0], 4))
    Tsqr[:, 0] = Qz_array[:, 0]

    alpha_f = np.degrees(np.arcsin(Qz_array[:, 0] / (2 * np.pi) * wavelength - np.sin(np.radians(alpha_i_deg))))
    alpha_c = np.degrees(np.arcsin(Qc / (2 * 2 * np.pi / wavelength)))
    Tsqr[:, 1] = alpha_f
    Tsqr[:, 2] = alpha_f / alpha_c
    Tsqr[:, 3] = ave_vf(alpha_f, footprint_mm, energy_eV, alpha_i_deg, Ddet_mm)
    return Tsqr
