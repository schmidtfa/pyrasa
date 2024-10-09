---
title: 'PyRASA - Spectral parameterization in python based on IRASA'
tags:
    - Python
    - spectral parametrization
    - 1/f
    - aperiodic
    - oscillations
    - electrophysiology
    - time frequency analysis
    - electroencephalography
    - EEG
    - magnetoencephalography
    - MEG
authors:
    - name: Fabian Schmidt
      corresponding: true
      orcid: 0000-0002-9839-1614
      affiliation: 1
    - name: Thomas Hartmann
      orcid: 0000-0002-8298-8125
      affiliation: 1
    - name: Nathan Weisz
      orcid: 0000-0001-7816-0037
      affiliation: "1, 2"
affiliations:
    - name: Paris-Lodron-University of Salzburg, Department of Psychology, Centre for Cognitive Neuroscience, Salzburg, Austria
      index: 1
    - name: Neuroscience Institute, Christian Doppler University Hospital, Paracelsus Medical University, Salzburg, Austria
      index: 2
date: XX August 2024
bibliography: paper.bib
---

# Summary
The electric signals generated by physiological activity exhibit both activity patterns that are regularly repeating over time (i.e. periodic) and activity patterns that are temporally irregular (i.e. aperiodic). In recent years several algorithms have been proposed to separate the periodic from the aperiodic parts of the signal, such as the irregular-resampling auto-spectral  analysis (IRASA; `@wen2016separating`). IRASA separates periodic and aperiodic components by up-/downsampling time domain signals and computing their respective auto-power spectra. Finally, the aperiodic component is isolated by averaging over the resampled auto-power spectra removing any frequency specific activity. The aperiodic component can then be subtracted from the original power spectrum yielding the residual periodic component. 
`PyRASA` is a package that is built upon and extends the IRASA algorithm `[@wen2016separating]`. The package allows its users not only to separate power spectra, but also contains functionality to further parametrize the periodic and aperiodic spectra, by means of peak detection and several slope fitting options (eg. spectral knees). Furthermore, we extended the IRASA algorithm to the time-frequency domain allowing for a time-resolved spectral parameterization using IRASA.

# Statement of Need
`PyRASA` is an open-source Python package for the parametrization of (neural) power spectra. `PyRASA` has a lightweight architecture that allows users to directly apply the respective functions to separate power spectra to numpy arrays containing time series data `[@harris2020array]`. However, `PyRASA` can also be optionally extended with functionality to be used in conjunction with MNE Python (a popular beginner-friendly tool for the analysis of electrophysiological data, `@gramfort2014mne`). Thus offering both beginners in (neural) time series analysis and more advanced users a tool to easily analyze their data. The IRASA algorithm per se has been implemented in a couple other software packages `[@cole2019neurodsp; @vallat2021open; @oostenveld2011fieldtrip]`, but these implementations of IRASA largely lack functionality to further parametrize periodic and aperiodic spectra in their respective components. We close this gap by offering such functionality both for periodic and aperiodic spectra. For periodic spectra users can extract peak height, bandwidth and center frequency of putative oscillations. Aperiodic spectra can be further analyzed by means of several slope fitting options that allow not only for the assessment of Goodness of fit by several metrics (R2, mean squared error), but also allow for model comparison using information criteria (BIC/AIC). Furthermore, we extended the IRASA algorithm to the time-frequency domain, by computing IRASA over up/downsampled versions of spectrograms instead of power spectra thereby also allowing for a time-resolved spectral parametrization of (neural) time series data. 

# Related Projects
`PyRASA’s` functionality is inspired by specparam (formerly FOOOF, `@donoghue2020parameterizing`) a popular tool spectral parametrization built upon a different algorithm that seperates powers spectra by first flattening the spectrum and then sequentially modelling peaks as gaussians which is followed a final fit of the aperiodic component. Each algorithm (IRASA vs. Specparam) comes with their specific advantages and disadvantages that are in part also discussed herein `[@gerster2022separating]` and further eluded to in our `Documentation`.

The IRASA algorithm has also been implemented as part of other software packages NeuroDSP `[@cole2019neurodsp]`, YASA `[@vallat2021open]` and FieldTrip `[@oostenveld2011fieldtrip]`.

# Acknowledgements
We want to thank Gianpaolo Demarchi, Patrick Reisinger and Mohammed Ameen for beta testing PyRASA and helpful comments that improved its development.

# References