"""

0D Energy Balance Climate Model (1900-2025) using Forward Euler Method
---------------------------------------------------
Author: Carden Taylor
Description: A zero-dimensional energy-balance climate model simulating surface 
             temperature trajectories using classical IPCC radiative forcing equations
             and Forward Euler numerical integration.
Validation Data: NASA GISTEMP v4

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================#
# PHYSICAL CONSTANTS & MODEL PARAMETERS     #
# ==========================================#

s_0 = 1361 # TSI (W/m^2)
a = 0.31 # Earth's Albedo
epsilon_0 = 0.61 # Emissivity of the Earth in year 1900
sigma = 5.67e-8 # Stefan-Boltzmann Constant

C_0, M_0, N_0 = 295, 860, 275 # Pre-industrial levels of co2, ch4, and n2o
r_c, r_m, r_n = 0.0030, 0.0063, 0.0018 # Rate of growth for each GHG
T_0 = 287.10 # Initial temperature (in Kelvin) at t = 1900

dt_years = 0.01 # Step size in years
dt_seconds = dt_years * 31536000.0 # Step size in terms of seconds to account for (K/s) measured by the ODE

# ============================================#
# RADIATIVE FORCING & CONCENTRATION EQUATIONS #
# ============================================#

def co2_concentration(t):  # Concentration of Co2 at 't'

    return C_0*(1+r_c)**t 

def ch4_concentration(t):  # Concentration of CH4 at 't'

    return M_0*(1+r_m)**t

def n2o_concentration(t):  # Concentration of N2O at 't'

    return N_0*(1+r_n)**t

def radiative_forcing(M,N):  # Equation f(M,N)

    return 0.47*np.log(1+(2.01e-5)*(M*N)**.75 + 5.31e-15*M*(M*N)**1.52)

def temperature_rate(t, C_t, M_t, N_t, T_k): # Calculate the ROC of Temperature in Kelvin

    solar_flux = (s_0*(1-a))/4
    carbon_tot = 5.35*np.log(C_t/C_0)
    methane_tot = 0.036*(np.sqrt(M_t)-np.sqrt(M_0)) - (radiative_forcing(M_t, N_0) - radiative_forcing(M_0, N_0))
    nitrous_oxide_tot = 0.12*(np.sqrt(N_t) - np.sqrt(N_0)) - (radiative_forcing(M_0, N_t) - radiative_forcing(M_0, N_0))

    f_out = epsilon_0*sigma*(T_k)**4
    f_tot = solar_flux + carbon_tot + methane_tot + nitrous_oxide_tot - f_out


    # Effective ocean mixed-layer heat capacity
    # rho = 1025 kg/m^3
    # c_p  = 3993 J/(kg K)
    # H   = 50 m
    # C = rho*c_p*d
    inverse_C = (1)/(1025*3993*50)

    return inverse_C*f_tot

# ============================================#
# INTEGRATION OF FORWARD EULER METHOD         #
# ============================================#

# Defining arrays for graphing as well as iteration variable T_k_current

t, t_c = [], []
T_k_current = T_0

# Calculating number of steps needed given total years and step size

total_years = 125 # Simulating range: 1900 to 2025
num_steps = int(total_years / dt_years)

# Calculating approximation of global temperature using Forward Euler's Method

for i in range(num_steps):
    t_years = i * dt_years
    
    C_t = co2_concentration(t_years)
    M_t = ch4_concentration(t_years)
    N_t = n2o_concentration(t_years)
    
    g_at = temperature_rate(t_years, C_t, M_t, N_t, T_k_current)
    
    t.append(1900 + t_years)
    t_c.append(T_k_current - 273.15)
    
    T_k_current = T_k_current + dt_seconds * g_at

# ============================================#
# NASA GISTEMP OBSERVATIONAL DATA             #
# ============================================#

actual_years = np.array([
    1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911,
    1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923,
    1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935,
    1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947,
    1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959,
    1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971,
    1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983,
    1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
    1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
    2020, 2021, 2022, 2023, 2024, 2025
])

actual_temperatures = np.array([
    13.95, 13.95, 13.70, 13.64, 13.58, 13.75, 13.85, 13.60, 13.70, 13.69,
    13.79, 13.74, 13.67, 13.72, 13.98, 14.06, 13.80, 13.54, 13.67, 13.91,
    13.85, 13.95, 13.91, 13.84, 13.89, 13.85, 14.04, 13.95, 14.00, 13.78,
    13.97, 14.03, 14.04, 13.89, 14.05, 13.92, 14.01, 14.12, 14.15, 13.98,
    14.14, 14.11, 14.10, 14.06, 14.11, 13.99, 14.01, 14.12, 13.97, 13.91,
    13.83, 13.98, 14.03, 14.12, 13.91, 13.91, 13.82, 14.08, 14.10, 14.05,
    13.98, 14.10, 14.05, 14.03, 13.65, 13.75, 13.93, 13.98, 13.91, 14.00,
    14.04, 13.90, 13.95, 14.18, 13.94, 13.98, 13.79, 14.16, 14.07, 14.13,
    14.27, 14.40, 14.10, 14.34, 14.16, 14.13, 14.19, 14.35, 14.42, 14.28,
    14.49, 14.44, 14.16, 14.18, 14.31, 14.47, 14.36, 14.40, 14.71, 14.44,
    14.41, 14.56, 14.70, 14.64, 14.60, 14.77, 14.63, 14.65, 14.53, 14.65,
    14.72, 14.60, 14.64, 14.67, 14.74, 14.89, 15.01, 14.92, 14.84, 14.97,
    15.01, 14.84, 14.88, 15.17, 15.27, 15.16
])

# ==========================================#
# RESIDUALS                                 #
# ==========================================#

predicted_at_actual_years = np.interp(actual_years, t, t_c)
residuals = actual_temperatures - predicted_at_actual_years

fig, axs = plt.subplots(ncols=2)

# ==========================================#
# VISUALIZATION                             #
# ==========================================#

sns.set_theme(style="darkgrid")
sns.lineplot(x=t,y=t_c, color="blue", label="0D Energy-Balance Climate Model", ax=axs[0])
sns.lineplot(x=actual_years, y=actual_temperatures, color="red", label = "NASA GISTEMP v4", ax=axs[0])
sns.scatterplot(x=predicted_at_actual_years, y=residuals, color = "green", ax=axs[1])

axs[0].set_title("Global Mean Temperature Comparison (1900-2025)")
axs[0].set_xlabel("Year")
axs[0].set_ylabel("Average Global Temperture (°C)")

axs[1].set_title("Model Residuals vs Predicted Temperature")
axs[1].axhline(0, color="black", linestyle="--", alpha=0.7)  # Zero error line
axs[1].set_xlabel("Predicted Average Global Temperture (°C)")
axs[1].set_ylabel("Residuals (°C)")

plt.show()