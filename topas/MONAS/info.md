Secondly, it's important that you construct test cases for your geometry to confirm you're not seeing simulation artifacts. For example, make a 100 nanometer cube of water and irradiate it with some very low energy beta particles. Record the energy deposition and number of interactions in the volume. Now subdivide the cube into one nanometer cubes of water that touch and run the same simulation. You would expect to get identical results as the simulation should be equivalent right? Both in terms of energy deposition and its spatial distribution. Should be the same right?


The modified version of SMKM (mSMKM) is based on the assumption that, in charged-particle therapy, the domain-specific energy zd is, in general, delivered by a large number of low-energy deposition events, and the events inducing the saturation of complex DNA damages are rare. Also, it is assumed that the specific energy imparted zn is sufficiently close to the macroscopic dose D

Based on specific energy microdosimetric spectra, MONAS predicts cell survival fraction and RBE using the different MKM formulations and GSM2 radiobiological model

￼
See how the dosimetry results change based on the following formula:


\\
RBE(S,D) = \dfrac{\sqrt{(\alpha_X^2 - 4\beta_X\ln(S(D)))}-\alpha_X}{2\beta_XD}
\\
where $α_X$ and $β_X$ are the linear-quadratic coefficients of photon reference radiation.
And then based upon this, add a pull in github as comment.


Due to the intrinsic difference in the description of radiation energy deposition in the sensitive volume between TOPAS and the previous work, we re-calculated the model parameters. A systematic comparison between Monte Carlo condensed history (e.g. TOPAS MC) and track structure algorithm for calculating specific energy spectra is out of the scope of this work.

Therefore, the parameters for the MKM and GSM2 were determined to reproduce in vitro experimental data of HSG cells.

The sphere was irradiated by a mono-energetic 3 He (10.2MeV/u and 4.89 MeV/u) and 12 C (12.9 MeV/u and 126 MeV/u) ion beams as reported in the experiments 

^^^^ Compare the values with and without the use of MONAS, and then edit and the significance in the differences between the dosimetries of the calculated RBEs

the following \alpha_{mix} & \beta_{mix} were used:

\\

\alpha_{mix} = \dfrac{{\bbSigma_{s=1}^{N_s} dE_{v,s}\sqrt{\beta(E_{v,s})}}{\bbSigma_{s=1}^{N_s}dE_{v,s}}

\\
Cell survival fraction and RBE were calculated using MKM formulations and GSM2 along the beam axis using experimentally validated microdosimetric spectra, (Missiaggia et al 2023a), as described in section 2.3.2. 


\\

\\


\\\\conc

MONAS wraps the already published TOPAS microdosimetric extensions to evaluate the single- and multi-event specific energy (z) distributions at different micrometric scales. Full microdosimetric distributions are then used as input for both MKM and GSM2 models. This approach showed intrinsic differences in microdosimetric radiation characterization with respect to the amorphous track structure model used in the latest MKM formulations. Therefore, we recalculated the model parameter that best fit the radiobiological experiments for the HSG cell line. To show the main MONAS applications, we reproduced experimental microdosimetric spectra from a passively scattered SOBP. We used the MONAS code to assess cell survival fraction and RBE as a function of proton penetration depth. Our findings are consistent with the well-known RBE trend, which presents a steep increase in the distal edge of the field. Furthermore, we were able to assess the high inter-model variability on the absolute RBE values thus quantifying a radiobiological uncertainty in proton plans in addition to other physical uncertainties.In conclusion, the MONAS extension offers a comprehensive microdosimetric framework for assessing the biological effect of radiation in both research and clinical environments. MONAS could be a key tool to include a detailed microdosimetric description of radiation field into treatment planning systems for variable RBE calculations.


\\AI a summary, and then add to a separate build to test. Name : "Test1MONAS"


##


Artificial Intelligence applied to Radiation Oncology

Among the major revolution of our century, artificial intelligence and data analysis are without any doubt among the most relevant. Machine and Deep Learning (ML and DL) models are nowadays systematically used in many fields. We recently started gaining interest in application of machine and deep learning techniques applied to medical physics and radiation oncology. We have implemented a machine learning model to track particles in microdosimetry to overcome experimental limitations of our new detector. We have also developed a fully machine and deep learning algorithm to predict biological effectiveness of a wide range of ions relevant both for radiotherapy and for radioprotection (ANAKIN).