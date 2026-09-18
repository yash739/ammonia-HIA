### Phase 1: Data Ingestion and State Bookkeeping

1. **Parse Molecular Levels:**
* Extract the first 36 energy levels from `p-nh3@loreau.dat.txt` to cover all rotation-inversion doublets up to $(J,K) = (4,4)$:


* Levels 1–6: $(1,1)$ parity doublet with nuclear quadrupole splitting ($F_1 = 0, 2, 1$ in lower and upper inversion states).


* Levels 7–12: $(2,2)$ doublet ($F_1 = 1, 3, 2$).


* Levels 13–18: $(2,1)$ doublet ($F_1 = 2, 3, 1$).


* Levels 19–24: $(3,2)$ doublet ($F_1 = 2, 3, 4$).


* Levels 25–30: $(3,1)$ doublet ($F_1 = 3, 4, 2$).


* Levels 31–36: $(4,4)$ doublet ($F_1 = 3, 5, 4$).




* Retain energies $E_i$ (converted from $\text{cm}^{-1}$ to $\text{erg}$ or $\text{K}$) and statistical weights $g_i$ directly from the data file.




2. **Ingest Radiative Transitions:**
* Read the radiative transitions connecting states $u \le 36$ and $l \le 36$ (yielding the 77 allowed transitions in the 36-level system).


* For each transition $t = (u, l)$, record:
* Rest frequency $\nu_t$.


* Einstein spontaneous emission coefficient $A_{ul}$.


* Compute stimulated coefficients:



$$B_{ul} = \frac{c^2}{2 h \nu_t^3} A_{ul}, \quad B_{lu} = \frac{g_u}{g_l} B_{ul}$$






3. **Ingest and Interpolate Collisional Rates:**
* Read state-to-state collisional de-excitation rate coefficients $C_{ul}(T)$ between states $u > l$ for the 11 tabulated kinetic temperatures $T \in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]\text{ K}$.


* Implement cubic spline interpolation over $\log(T)$ to evaluate $C_{ul}(T_k)$ continuously across the target model range $T_k \in [18, 40]\text{ K}$.


* Calculate upward collisional rates by detailed balance:



$$C_{lu}(T_k) = \frac{g_u}{g_l} C_{ul}(T_k) \exp\left(-\frac{E_u - E_l}{k_B T_k}\right)$$





---

### Phase 2: Hyperfine Frequency Grouping and Geometry Setup

1. **Velocity Dispersion and Line Overlap Criteria:**
* Adopt the thermal intrinsic clump linewidth $\Delta v_{\text{clump}} = 0.3\text{ km s}^{-1}$.


* For any pair of transitions $t, t'$, compute their rest velocity separation:



$$\delta v = c \, \frac{\vert{}\nu_t - \nu_{t'}\vert{}}{\nu_0}$$


* If $\delta v < \Delta v_{\text{clump}}$, assign transitions $t$ and $t'$ to the same overlapping line group $G$.


* For the far-IR $(2,1) \to (1,1)$ transition ($\sim 1168\text{ GHz}$ and $1215\text{ GHz}$):


* Group the inner quadrupole satellites and the main component together into a single overlapping frequency group $G_{\text{FIR}}$.


* Leave the outer quadrupole satellites ($F_1 = 1 \to 0$ and $F_1 = 0 \to 1$) as separate, uncoupled single lines.




* This yields Stutzki's structure of 12 overlapping groups and 28 isolated single lines.




2. **Optical Depth Along Clump Radius:**
* For a given input parameter pair $(n_{\text{H}_2}, N_{\text{NH}_3}/\Delta v)$, define the clump radius $R$ through the column density:

$$N_{\text{NH}_3} = \left(\frac{N_{\text{NH}_3}}{\Delta v}\right) \Delta v_{\text{clump}}$$


* For each transition $t$ connecting $l \to u$, compute the line-center absorption coefficient per unit length:



$$k_t = \frac{h \nu_t}{4\pi \Delta \nu_{\text{clump}}} (n_l B_{lu} - n_u B_{ul})$$



where $\Delta \nu_{\text{clump}} = \nu_t (\Delta v_{\text{clump}} / c)$.


* Sum over all transitions $g$ belonging to group $G$ to obtain the radial group optical depth:



$$\tau_G = \sum_{g \in G} k_g R$$




3. **Precomputed Escape Probability Kernel:**
* Precompute a 1D spline table for the center-of-sphere Gaussian escape probability $\beta(\tau)$ over $\log_{10} \tau \in [-4, 4]$:



$$\beta(\tau) = \frac{1}{\sqrt{\pi}} \int_{-\infty}^{+\infty} e^{-z^2} \exp\left(-\tau e^{-z^2}\right) dz$$


* For each group $G$, look up $\beta_G = \beta(\tau_G)$.





---

### Phase 3: Statistical Equilibrium and Non-LTE Solver

1. **Mean Radiation Field Coupling:**
* Evaluate the cosmic microwave background intensity at frequency $\nu_G$:



$$F_{\nu_G} = \frac{2 h \nu_G^3}{c^2} \frac{1}{\exp(h\nu_G / k_B T_{\text{bg}}) - 1}, \quad T_{\text{bg}} = 2.7\text{ K}$$


* Define the pooled source function of group $G$:



$$S_G = \frac{\sum_{g \in G} k_g S_g}{\sum_{g \in G} k_g}, \quad S_g = \frac{2 h \nu_g^3}{c^2} \left(\frac{g_u n_l}{g_l n_u} - 1\right)^{-1}$$


* Compute the effective mean radiation field seen by transition $t \in G$:



$$J_t = F_{\nu_t} \beta_G + S_G (1 - \beta_G)$$




2. **Formulate the Nonlinear Rate Equations:**
* Set up the steady-state statistical equilibrium system for populations $n_i$ ($i = 1, \dots, 36$):



$$\frac{dn_i}{dt} = \sum_{j \ne i} \left[ n_j \left(n_{\text{H}_2} C_{ji} + R_{ji}\right) - n_i \left(n_{\text{H}_2} C_{ij} + R_{ij}\right) \right] = 0$$



where radiative rates $R_{ij}$ are defined as:



$$R_{ji} = \begin{cases} A_{ji} + B_{ji} J_{ji}, & E_j > E_i \\ B_{ji} J_{ij}, & E_j < E_i \end{cases}$$


* Enforce particle conservation $\sum_{i=1}^{36} n_i = n_{\text{tot}}$ (or normalize to $\sum n_i = 1$).




3. **Newton-Raphson Iteration Engine:**
* **Initialization:** Initialize level populations $n_i^{(0)}$ with a thermal Boltzmann distribution at kinetic temperature $T_k$:



$$n_i^{(0)} \propto g_i \exp\left(-\frac{E_i}{k_B T_k}\right)$$


* **Iteration Loop:**
* Evaluate optical depths $\tau_G^{(k)}$, escape probabilities $\beta_G^{(k)}$, and mean fields $J_t^{(k)}$.


* Construct the $36 \times 36$ rate matrix $\mathbf{M}$ and residual vector $\mathbf{F}(\mathbf{n}^{(k)})$.


* Form the Jacobian matrix $\mathbf{\mathcal{J}} = \partial \mathbf{F} / \partial \mathbf{n}$, accounting for the explicit dependence of $J_t$ on populations via $\tau_G$ and $S_G$.


* Replace row 36 with the normalization condition $\sum_i n_i = 1$.
* Solve the linear update system:

$$\mathbf{\mathcal{J}} \cdot \Delta \mathbf{n} = -\mathbf{F}$$


$$n_i^{(k+1)} = n_i^{(k)} + \alpha \, \Delta n_i$$



using a damping parameter $\alpha \in (0.5, 1.0]$ if high optical depths induce oscillatory behavior.


* **Convergence Criterion:** Terminate when the fractional population changes satisfy $\max_i \vert{}\Delta n_i / n_i\vert{} < 10^{-4}$ across three consecutive iterations.





---

### Phase 4: Emergent Clump Brightness and Observable Synthesis

1. **Chord-Averaged Spherical Escape Factor:**
* Compute Stutzki’s spherical cloud geometry factor $e(2\tau_G)$ for the emergent surface intensity:



$$e(2\tau_G) = 2 \, \frac{1 - e^{-2\tau_G}(1 + 2\tau_G)}{(2\tau_G)^2}$$



(using the series expansion $e(x) \approx 1 - \frac{2}{3}x + \frac{1}{4}x^2$ for $x < 10^{-3}$ to prevent division by zero).


2. **Emergent Line-Center Brightness Temperatures:**
* For each $(1,1)$, $(2,2)$, and $(2,1)$ hyperfine component $t$, calculate the clump-surface brightness temperature:



$$T_{B,t} = \frac{c^2}{2 k_B \nu_t^2} (S_G - F_{\nu_t}) \left[1 - e(2\tau_G)\right]$$




3. **Compute Hyperfine Observable Diagnostics:**
* Define the $(1,1)$ main component as the reference brightness $T_B(\Delta F_1 = 0) = T_B(1,1:\text{main})$.


* Calculate the four quadrupole satellite intensity ratios:



$$R(1 \to 0) = \frac{T_B(F_1 = 1 \to 0)}{T_B(\Delta F_1 = 0)} \quad (\text{outer blue satellite})$$


$$R(0 \to 1) = \frac{T_B(F_1 = 0 \to 1)}{T_B(\Delta F_1 = 0)} \quad (\text{outer red satellite})$$


$$R(1 \to 2) = \frac{T_B(F_1 = 1 \to 2)}{T_B(\Delta F_1 = 0)} \quad (\text{inner red satellite})$$


$$R(2 \to 1) = \frac{T_B(F_1 = 2 \to 1)}{T_B(\Delta F_1 = 0)} \quad (\text{inner blue satellite})$$


* Calculate the inversion line ratios:



$$R_{22/11} = \frac{T_B(2,2:\Delta F_1 = 0)}{T_B(1,1:\Delta F_1 = 0)}, \quad R_{21/11} = \frac{T_B(2,1)}{T_B(1,1:\Delta F_1 = 0)}$$





---

### Phase 5: Grid Generation, Observational Fitting, and Clump Physics

1. **Model Grid Architecture:**
* Compute a 3D forward-model grid over the parameter space:


* $T_k \in [18, 40]\text{ K}$ (e.g., slices at $18, 24, 30, 36\text{ K}$).


* $\log_{10}(n_{\text{H}_2}\ [\text{cm}^{-3}]) \in [3.5, 7.0]$ in steps of $0.05$.


* $\log_{10}(N_{\text{NH}_3}/\Delta v\ [\text{cm}^{-2}\text{ s km}^{-1}]) \in [13.7, 15.6]$ in steps of $0.05$.




* Save outputs to an HDF5 table storing $(T_k, n_{\text{H}_2}, N_{\text{NH}_3}/\Delta v) \to (R(1\to0), R(0\to1), R(1\to2), R(2\to1), R_{22/11}, R_{21/11}, T_{B,\text{main}})$.


2. **Spectral Fitting Module:**
* For an observed source (e.g., from Stutzki et al. 1984 Table 2), construct the chi-squared metric across observed satellite ratios $R_{\text{obs}, k}$ and uncertainties $\sigma_k$:



$$\chi^2 = \sum_{k=1}^{5} \left(\frac{R_{\text{theor}, k}(T_k, n_{\text{H}_2}, N_{\text{NH}_3}/\Delta v) - R_{\text{obs}, k}}{\sigma_k}\right)^2$$


* Determine best-fit parameters $(T_k, n_{\text{H}_2}, N_{\text{NH}_3})$ by minimizing $\chi^2$ via Levenberg-Marquardt or grid search.


* Evaluate the beam filling factor $\eta_f$ using the observed line-center brightness $T_{B,\text{obs}}$:



$$\eta_f = \frac{T_{B,\text{obs}}}{T_{B,\text{theor}}}$$



(Reject solutions where $\eta_f > 1$ as unphysical low-density artifacts).




3. **Derive Clump Stability and Mass Parameters:**
* For the high-density best-fit point, calculate the local clump properties:


* Jeans length:

$$\lambda_J = 0.776 \times 10^{-3}\text{ pc} \left(\frac{T_k / 10\text{ K}}{n_{\text{H}_2} / 10^7\text{ cm}^{-3}}\right)^{1/2}$$


* Jeans mass:

$$M_J = 0.103\text{ M}_\odot \left(\frac{(T_k / 10\text{ K})^3}{n_{\text{H}_2} / 10^7\text{ cm}^{-3}}\right)^{1/2}$$


* Maximum single-clump mass at source distance $r$:

$$M_c(\max) = 347.4\text{ M}_\odot \left(\frac{n_{\text{H}_2}}{10^7\text{ cm}^{-3}}\right) \left(\frac{r}{0.5\text{ kpc}}\right)^3 \eta_f^{3/2}$$


* Total number of clumps per beam $K$ (assuming $M_c = M_J$):



$$K = 225 \left(\frac{\eta_f \, (n_{\text{H}_2}/10^7\text{ cm}^{-3})}{T_k / 10\text{ K}}\right) \left(\frac{r}{0.5\text{ kpc}}\right)^2 \left(\frac{\Delta v_{\text{obs}}}{\Delta v_{\text{clump}}}\right)$$


* Volume number density of clumps $\nu$ and smeared-out mean density $\bar{n}_{\text{H}_2}$:



$$\bar{n}_{\text{H}_2} = 2 \times 10^5\text{ cm}^{-3} \left(\frac{T_k}{10\text{ K}} \frac{n_{\text{H}_2}}{10^7\text{ cm}^{-3}}\right)^{1/2} \frac{\eta_f}{(r/0.5\text{ kpc})(\theta_s / 1')} \frac{\Delta v_{\text{obs}}}{\Delta v_{\text{clump}}}$$







---

### Phase 6: Model Validation and Benchmarking

1. **Topology Test Against Stutzki & Winnewisser (1985) Contours:**
* Slice the computed grid at $T_k = 24\text{ K}$ and plot the isolines in the $\log(n_{\text{H}_2})\text{--}\log(N_{\text{NH}_3}/\Delta v)$ plane:


* Verify that $R(0 \to 1)$ peaks above $1.0\text{--}1.2$ in the density range $n_{\text{H}_2} \sim 10^{4.5}\text{--}10^{5.5}\text{ cm}^{-3}$.


* Verify that $R(1 \to 0)$ exhibits a minimum ($< 0.25\text{--}0.20$) in the density range $n_{\text{H}_2} \sim 10^{5.0}\text{--}10^{6.5}\text{ cm}^{-3}$.


* Confirm that the averaged inner satellite ratio remains close to the standard LTE curve:



$$\frac{R(1 \to 2) + R(2 \to 1)}{2} \approx \frac{1 - e(2\tau_s)}{1 - e(2\tau_m)}$$






2. **Empirical Benchmarks:**
* Fit S106 $(0,0)$ using observed ratios from Table 2 ($R(1\to0)=0.428$, $R(0\to1)=0.503$, $R_{22/11}=0.52$).


* Verify convergence to the high-density branch $n_{\text{H}_2} \approx 10^{6.25}\text{ cm}^{-3}$ with $\chi^2 < 1.0$.


* Predict the $(2,1)/(1,1)$ ratio and confirm it reproduces $T_B(2,1)/T_B(1,1) \approx 0.018$, validating against the measured value of $0.014 \pm 0.003$.