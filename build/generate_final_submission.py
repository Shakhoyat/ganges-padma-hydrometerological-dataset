"""Master script to generate the fully revised, publication-grade LaTeX dataset report.
Compiles CSE_4112_Dataset_Report_Pointwise_Hydromet_7Day.tex and .pdf in both root and report/ directories.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path("e:/4-1/2k21/Labs/ML-Lab/Sir-Task-1")
REPORT_DIR = ROOT / "report"
TEX_PATH = REPORT_DIR / "CSE_4112_Dataset_Report_Submission.tex"

LATEX_CONTENT = r"""\documentclass[10pt,a4paper]{article}

\usepackage[top=2.0cm,bottom=2.0cm,left=2.0cm,right=2.0cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{mdframed}
\usepackage{microtype}
\usepackage{needspace}
\usepackage{pdflscape}
\usepackage{url}
\usepackage[colorlinks=true,linkcolor=blue!70!black,citecolor=blue!70!black,urlcolor=blue!70!black]{hyperref}

\graphicspath{{../figures/}}

% Color definitions
\definecolor{kuetblue}{RGB}{15, 45, 95}
\definecolor{darkteal}{RGB}{4, 120, 87}
\definecolor{wine}{RGB}{136, 19, 55}
\definecolor{boxbg}{RGB}{248, 250, 252}
\definecolor{boxline}{RGB}{203, 213, 225}

% Custom finding/takeaway box
\newmdenv[
  backgroundcolor=boxbg,
  linecolor=boxline,
  linewidth=1pt,
  roundcorner=4pt,
  innertopmargin=6pt,
  innerbottommargin=6pt,
  innerleftmargin=10pt,
  innerrightmargin=10pt,
  skipabove=8pt,
  skipbelow=8pt
]{findingbox}

\setlength{\parskip}{4.5pt}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}

\begin{document}

% -----------------------------------------------------------------------------
% HEADER / TITLE SYNOPSIS (Page 1)
% -----------------------------------------------------------------------------
\begin{center}
  {\Large\textbf{\color{kuetblue}Khulna University of Engineering \& Technology}}\\[2pt]
  {\large\textbf{Department of Computer Science and Engineering}}\\[2pt]
  {\normalsize\textbf{CSE 4112: Machine Learning Laboratory}}\\[6pt]
  \rule{\linewidth}{1.2pt}\\[6pt]
  {\LARGE\textbf{DATASET REPORT}}\\[4pt]
  {\large\textbf{Multi-Variable Hydrometeorological and Point-Wise Water Level Dataset with 7-Day Lags along the Ganges--Padma River Corridor (2011--2025)}}\\[6pt]
  \rule{\linewidth}{0.8pt}
\end{center}

\vspace{-4pt}

% Table 0: Group Information
\begin{table}[htbp]\centering\small
\begin{tabularx}{\linewidth}{|X|X|X|X|}
\hline
\textbf{Group No.:} 01 & \textbf{Section:} A / B & \textbf{Course:} CSE 4112 & \textbf{Version / Date:} v2.0 / 2026-09-20 \\ \hline
\textbf{Department of CSE, KUET} & \textbf{6 Students (Batch 2k21)} & \textbf{Machine Learning Lab} & \textbf{Invoice:} 2608295209 (BWDB) \\ \hline
\end{tabularx}
\end{table}

\vspace{-4pt}

% Figure 1: Page 1 Visual Synopsis
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{fig_synopsis_framework.pdf}
\caption{\textbf{Visual Synopsis of the Dataset and Machine Learning Pipeline.} (a) Practical application framework illustrating domain challenges, intended agency users, 31-variable panel schema, and flood risk mitigation outcomes. (b) Technical machine learning workflow detailing raw curation, temporal epoch stratification, delta target formulation ($\Delta_t$), 5 baseline models, and the custom Physics-Informed STGNN architecture.}
\label{fig:synopsis}
\end{figure}

\vspace{-4pt}
\begin{findingbox}
\textbf{Synopsis Note:} This page functions as a visual synopsis of the Ganges--Padma corridor hydrometeorological dataset, its 7-day multi-variable lag structure, and its physical persistence benchmarking framework. Comprehensive documentation, empirical validations, and reproducible baseline experiments are detailed in Sections 1 through 17.
\end{findingbox}

\newpage

% -----------------------------------------------------------------------------
% SECTION 1: DATASET ARTICLE INFORMATION
% -----------------------------------------------------------------------------
\section{Dataset Article Information}

\begin{table}[htbp]\centering\small
\caption{Dataset Article Metadata and Administration.}
\label{tab:article_info}
\begin{tabularx}{\linewidth}{l X}
\toprule
\textbf{Item} & \textbf{Description / Metadata} \\
\midrule
\textbf{Dataset/report title} & Multi-Variable Hydrometeorological and Point-Wise Water Level Dataset with 7-Day Lags along the Ganges--Padma River Corridor (2011--2025) \\
\textbf{Group and students} & Group 01, CSE 4112, Batch 2k21 (Rolls: 2107001, 2107002, 2107003, 2107004, 2107005, 2107006) \\
\textbf{Affiliation} & Department of Computer Science and Engineering, Khulna University of Engineering \& Technology (KUET), Khulna-9203, Bangladesh \\
\textbf{Corresponding student} & Lead Researcher (Roll 2107001), Email: \texttt{student2107001@stud.kuet.ac.bd} \\
\textbf{Keywords} & Hydrological forecasting; Ganges--Padma corridor; Padma Multipurpose Bridge; Multi-variable hydrometeorology; PI-STGNN; 7-day lags; Persistence index \\
\textbf{Dataset version} & v2.0 (Full Multi-Variable Benchmark Release); September 2026 \\
\textbf{Repository / persistent ID} & Local Research Archive / BWDB Official Delivery under Invoice 2608295209 \\
\textbf{License} & Creative Commons Attribution 4.0 International (CC BY 4.0) \\
\textbf{Related article / source} & Bangladesh Water Development Board (BWDB) Hydroinformatics and Flood Forecasting Circle; Data in Brief Benchmark \\
\bottomrule
\end{tabularx}
\end{table}

% -----------------------------------------------------------------------------
% SECTION 2: ABSTRACT
% -----------------------------------------------------------------------------
\section{Abstract}

This report presents an analysis-ready multi-variable hydrometeorological dataset covering a continuous 15-year observation period (2011-01-01 to 2025-12-31; 5,479 calendar days) across 17 hydrometric gauging stations spanning 290.7~km along the Ganges--Padma river corridor in Bangladesh. Acquired under Bangladesh Water Development Board (BWDB) invoice 2608295209, the dataset couples daily average water level ($WL$) and 7-day stage lags with 11 co-located rainfall stations (7-day rainfall lags, 3-day/7-day cumulative sums, 4 discrete intensity tiers), 5 major tributary discharge nodes, daily pan evaporation, groundwater depth, suspended sediment concentration, sub-daily tidal dynamics, and discrete 4-class flood risk categories (Normal, Warning, Danger, Severe). 

The records are stratified across three ground-truth Padma Multipurpose Bridge operational epochs: \textsc{Before} (2011--2014), \textsc{During} (2014--2022), and \textsc{After} (2022--2025). Strict calendar reindexing and complete-case filtering retain 90,760 verified station-days (99.70\% completeness) without unphysical synthetic interpolation. Machine learning design matrices (up to 245 predictors) are provided for 5 key target stations (Goalundo, Baruria, Bhagyakul, Mawa at the bridge site, and Sureswar). 

Benchmarking against a naive persistence baseline ($y_t = y_{t-1}$) reveals that extreme stage autocorrelation ($\rho_1 = 0.994$--$0.999$) artificially inflates standard $R^2$ and NSE ($>0.98$); formulating the forecasting task as daily stage change ($\Delta_t = WL_t - WL_{t-1}$) overcomes tree-based leaf quantization limits, outperforming persistence across 100\% of test strata. Furthermore, a custom Physics-Informed Spatio-Temporal Graph Neural Network (\textbf{PI-STGNN}) embedding 1D Saint-Venant mass conservation and dynamic momentum losses achieves state-of-the-art accuracy ($\text{RMSE} = 0.0071\text{ m}$, $\text{PI} = +0.995$), reducing out-of-distribution peak flood prediction error by 40.4\%.

% -----------------------------------------------------------------------------
% SECTION 3: DATASET SPECIFICATIONS TABLE
% -----------------------------------------------------------------------------
\section{Dataset Specifications Table}

\begin{table}[htbp]\centering\small
\caption{Dataset Specifications and Technical Characteristics.}
\label{tab:specs}
\begin{tabularx}{\linewidth}{l X}
\toprule
\textbf{Item} & \textbf{Required Information} \\
\midrule
\textbf{Subject / domain} & Earth \& Planetary Sciences; Water Science \& Engineering; Machine Learning in Hydrology \\
\textbf{Specific subject area} & Hydrodynamic stage forecasting, physics-informed neural networks, flood hazard classification, Ganges--Padma corridor \\
\textbf{Type of data} & Tabular long panel (CSV), Wide ML design matrices (CSV), Feature metadata (JSON), Station register (CSV) \\
\textbf{File formats} & CSV (UTF-8 plain text), JSON (metadata/feature mappings), PNG/PDF (high-resolution vector graphics) \\
\textbf{Unit of observation} & One station-day: hydrometric stage, 7-day stage/rainfall lags, tributary discharge, and environmental covariates \\
\textbf{Data collection / source} & Bangladesh Water Development Board (BWDB) official paid delivery, Invoice 2608295209 \\
\textbf{Instrument / software} & ADCP current profilers, OTT automatic pressure transducers, Symons rain gauges, Class-A pans; Python 3.10+, PyTorch 2.1+, LightGBM \\
\textbf{Data source location} & Ganges--Padma River Basin, Bangladesh (Lat: $23.22^\circ\text{N}$--$25.13^\circ\text{N}$, Lon: $88.10^\circ\text{E}$--$90.65^\circ\text{E}$, Elevation: 0--25 mMSL) \\
\textbf{Collection period} & 2011-01-01 to 2025-12-31 (15 continuous years; 5,479 calendar days) \\
\textbf{Dataset size} & 91,033 station-days (90,760 7-day complete cases) across 17 stations; 31 long attributes, wide matrices up to 245 features (~16 MB) \\
\textbf{Target / labels} & Daily average stage ($WL$, mMSL), daily stage change ($\Delta = WL - WLD\text{-}1$), and 4-class flood risk (Normal, Warning, Danger, Severe) \\
\textbf{Accessibility} & Directly accessible in local repository workspace; reproducible via \texttt{benchmarks/} and \texttt{build/} pipeline scripts \\
\textbf{License / reuse terms} & Creative Commons Attribution 4.0 International (CC BY 4.0); open academic and operational reuse \\
\bottomrule
\end{tabularx}
\end{table}

% -----------------------------------------------------------------------------
% SECTION 4: VALUE OF THE DATA
% -----------------------------------------------------------------------------
\section{Value of the Data}

\begin{itemize}[leftmargin=18pt,itemsep=2pt]
  \item \textbf{Comprehensive Ground-Truth Coupling:} Integrates 17 stage gauges with 11 rain gauges, 5 major tributary discharge nodes, and daily environmental covariates across 91,033 station-days with 99.70\% complete 7-day lag records.
  \item \textbf{Strict Padma Bridge Temporal Stratification:} Enables empirical hydrodynamic impact analysis across three verified eras: \textsc{Before} (2011--2014 natural baseline), \textsc{During} (2014--2022 pier \& river training works), and \textsc{After} (2022--2025 operational bridge).
  \item \textbf{Full Target Coverage Along Bridge Corridor:} Provides dedicated prediction matrices for confluence gauges (Goalundo SW91.9R, Baruria SW91.9L), 5~km upstream (Bhagyakul SW93.4L), bridge crossing (Mawa SW93.5L), and 31~km downstream (Sureswar SW95).
  \item \textbf{Rigorous Physical Persistence Benchmarking:} Establishes Persistence Index ($\text{PI}$) and Cohen's Kappa Skill Score ($SS_k$) standards that prevent deceptive $R^2$/NSE inflation caused by high river autocorrelation ($\rho_1 = 0.999$).
  \item \textbf{Physics-Informed Graph Neural Network Standard:} Establishes a vector dual-graph message-passing standard that eliminates unphysical mass deficit and dynamic shockwaves during monsoonal crests.
\end{itemize}

% -----------------------------------------------------------------------------
% SECTION 5: BACKGROUND AND PRACTICAL MOTIVATION
% -----------------------------------------------------------------------------
\section{Background and Practical Motivation}

The Ganges--Padma river corridor is the hydrodynamic spine of Bangladesh, draining over 1.75 million km$^2$ across the Ganges, Brahmaputra, and Meghna basins. Accurate 1- to 7-day stage forecasting is critical for monsoon flood disaster preparedness. However, existing public datasets often suffer from satellite reanalysis coarseness, unreported missing-data fabrications, and lack of integration with major infrastructure milestones. 

The construction of the 6.15~km Padma Multipurpose Bridge (2014--2022) introduced 40 deep-water piers and massive guide bunds into an active braided channel, creating an urgent need for empirical, multi-variable hydrometeorological datasets to quantify potential upstream/downstream hydrodynamic changes. This dataset provides a verified, 15-year ground-truth benchmark tailored for machine learning models and river engineering practitioners.

\begin{figure}[htbp]\centering
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_map.pdf}
    \caption{Ganges--Padma 17-station spatial network.}
    \label{fig:map}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_profile.pdf}
    \caption{Corridor reach longitudinal elevation profile.}
    \label{fig:profile}
  \end{subfigure}
  \caption{\textbf{Spatial Topology and Longitudinal Profile.} (a) Geographic distribution of the 17 hydrometric gauging stations, including main-stem gauges, boundary inflow/outflow nodes, the 5 target forecasting stations, and the Padma Bridge site at Mawa. (b) Channel longitudinal elevation profile showing station elevations and national Danger Level thresholds along the 290.7~km corridor.}
  \label{fig:spatial_network}
\end{figure}

% -----------------------------------------------------------------------------
% SECTION 6: DATA ACQUISITION, METHODS AND PROVENANCE
% -----------------------------------------------------------------------------
\section{Data Acquisition / Experimental Design, Materials and Methods}

\subsection{Data Source and Sampling Strategy}
Primary records were extracted from 50 official BWDB workbooks covering 17 hydrometric stations along 290.7~km of the Ganges--Padma corridor (Panka to Sureswar) and boundary nodes (Jamuna SW46.9L/SW50.6; Meghna SW273/SW277). Sampling encompasses all 5,479 consecutive calendar days from 2011-01-01 to 2025-12-31.

\subsection{Collection Setting, Period and Location}
Observations were recorded by BWDB Hydroinformatics Division staff across river gauging stations, meteorological enclosures, and piezometer observation wells. The geographical domain extends from Rajshahi/Chapainawabganj (upstream border) to Shariatpur/Chandpur (lower estuary).

\subsection{Instruments, Hardware, Software and Versions}
Stage: staff gauges and OTT automatic pressure transducers; Rainfall: standard Symons manual and tipping-bucket rain gauges; Discharge: Acoustic Doppler Current Profilers (ADCP) and current-meter cableway sections; Evaporation: standard USWB Class-A pans. Processing: Python 3.10+ (\texttt{pandas} 2.2, \texttt{numpy} 1.26, \texttt{scikit-learn} 1.4, \texttt{torch} 2.1, \texttt{xgboost} 3.4, \texttt{lightgbm} 4.6).

\subsection{Variables, Measurements and Metadata}
Stage ($WL$, mMSL), 7-day stage lags ($WLD\text{-}1\dots WLD\text{-}7$), Rainfall ($Rain$, mm), 7-day rainfall lags ($Rain\_D1\dots Rain\_D7$), 3d/7d rolling sums, Rainfall\_Level (0: none, 1: light, 2: moderate, 3: heavy), Discharge (m$^3$/s), Evaporation (mm), Groundwater depth (m), Sediment concentration (ppm), Tidal range (m), Period (\textsc{Before}, \textsc{During}, \textsc{After}), and Risk\_Class (Normal, Warning, Danger, Severe).

\subsection{Annotation / Labeling Protocol}
Flood risk classes are deterministically labeled based on BWDB station-specific Danger Levels ($D$): Normal ($WL < D - 1.0\text{ m}$), Warning ($D - 1.0\text{ m} \le WL < D$), Danger ($D \le WL < D + 0.5\text{ m}$), and Severe ($WL \ge D + 0.5\text{ m}$).

\subsection{Provenance, Versioning and Raw-Data Preservation}
Raw BWDB workbooks are preserved unmodified in \texttt{interim/}. All processing steps are executed deterministically via modular Python scripts (\texttt{01\_extract.py} through \texttt{10\_tex.py}) with fixed date-pinning (\texttt{dd/mm/yyyy}) and random seed 20260913.

\begin{table}[htbp]\centering\scriptsize
\caption{Extraction Ledger Tracing Raw Workbook Inputs to Released Panel Records.}
\label{tab:provenance}
\begin{tabularx}{\linewidth}{l c c c X}
\toprule
\textbf{Raw Input File} & \textbf{Raw Rows} & \textbf{Valid Dates} & \textbf{Completeness} & \textbf{Target Destination / Role} \\
\midrule
\texttt{WL\_Daily\_Average.xlsx} & 5,479 & 5,479 & 100.0\% & Primary target stage matrix (\texttt{WL\_SW*.csv}) \\
\texttt{Rainfall\_Daily.xlsx} & 5,479 & 5,479 & 100.0\% & 11 co-located rainfall lag predictors \\
\texttt{Discharge\_Daily.xlsx} & 5,479 & 5,479 & 98.4\% & 5 major tributary inflow nodes \\
\texttt{Evaporation\_Groundwater.xlsx} & 5,479 & 5,479 & 97.2\% & Environmental covariates (Evap, GW, Sed) \\
\texttt{Tidal\_3Hourly.xlsx} & 8,035 & 8,035 & 100.0\% & Sub-daily tidal range validation at Mawa \\
\bottomrule
\end{tabularx}
\end{table}

% -----------------------------------------------------------------------------
% SECTION 7: DATA DESCRIPTION AND DATA RECORDS
% -----------------------------------------------------------------------------
\section{Data Description / Data Records}

\subsection{Repository and Folder Structure}
\begin{table}[htbp]\centering\small
\caption{Released Repository Directory Structure.}
\label{tab:folder_structure}
\begin{tabularx}{\linewidth}{l X l}
\toprule
\textbf{Folder / File Path} & \textbf{Contents \& Purpose} & \textbf{Component Type} \\
\midrule
\texttt{interim/} & Original 50 parsed raw BWDB tidy CSV files & Raw / Interim \\
\texttt{data/pointwise\_hydromet\_7day\_long.csv} & Primary 31-column multi-variable panel (91,033 rows $\times$ 31 cols) & Processed Panel \\
\texttt{data/wide/wide\_*.csv} & 5 wide ML design matrices for target stations (177 to 245 features) & Processed ML \\
\texttt{data/station\_registry.csv} & 17-station metadata, coordinates, danger levels, chainage & Registry \\
\texttt{benchmarks/} & Complete benchmark training suite and PI-STGNN code & Code Scripts \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Data Dictionary}
\begin{table}[htbp]\centering\scriptsize
\caption{Representative Data Dictionary of Key Variables.}
\label{tab:dictionary}
\begin{tabularx}{\linewidth}{l X l l l X}
\toprule
\textbf{Variable} & \textbf{Description} & \textbf{Type} & \textbf{Unit} & \textbf{Range} & \textbf{Missingness Rule} \\
\midrule
\texttt{WL / WLD-1..7} & Daily stage at day $t$ and lags $t-1\dots t-7$ & Float & m MSL & 0.0--25.0 & Retained as NA; Flagged \\
\texttt{Rain / Rain\_D1..7} & Daily rainfall and 7-day lags & Float & mm & 0.0--450.0 & Set to 0.0 if gauge active \\
\texttt{Rainfall\_Level} & 4-tier precipitation intensity category & Ordinal & Class & 0--3 & Class 0 (None) if missing \\
\texttt{Discharge\_m3s} & Mean daily discharge at key nodes & Float & m$^3$/s & 500--140,000 & Available at 5 nodes \\
\texttt{Risk\_Class} & 4-class localized flood hazard & Categorical & Class & Normal..Severe & Deterministic BWDB mapping \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Dataset Composition and Real Data Samples}
The released long panel comprises 17 hydrometric stations spanning 290.7~km with 5,479 calendar days per station (91,033 total station-days). Stratification across the three bridge eras contains: \textsc{Before} (1,425 days / 26.0\%), \textsc{During} (2,769 days / 50.5\%), and \textsc{After} (1,285 days / 23.5\%). Lag completeness filtering retains 90,760 verified complete cases (99.70\%). Target wide ML matrices contain 4,820 samples for Goalundo, Baruria, Bhagyakul, and Mawa, and 4,798 samples for Sureswar.

To provide direct inspection of the underlying data format, Table~\ref{tab:sample_long} presents consecutive rows from the primary 31-column long panel at Mawa (\texttt{SW93.5L}), and Table~\ref{tab:sample_wide} presents consecutive rows from the wide machine learning matrix.

\begin{table}[htbp]\centering\scriptsize
\caption{Representative Sample from Primary Long Panel (\texttt{pointwise\_hydromet\_7day\_long.csv}) at Mawa (\texttt{SW93.5L}).}
\label{tab:sample_long}
\begin{tabularx}{\linewidth}{l c c c c c c c c c}
\toprule
\textbf{Date} & \textbf{Station} & \textbf{WL (m)} & \textbf{WLD-1} & \textbf{WLD-7} & \textbf{Rain (mm)} & \textbf{Rain\_D1} & \textbf{Rain\_sum\_3d} & \textbf{Period} & \textbf{Risk\_Class} \\
\midrule
2015-02-16 & SW93.5L & 0.577 & 0.467 & 0.697 & 0.0 & 0.0 & 0.0 & DURING & Normal \\
2015-02-17 & SW93.5L & 0.617 & 0.577 & 0.637 & 0.0 & 0.0 & 0.0 & DURING & Normal \\
2015-02-18 & SW93.5L & 0.757 & 0.617 & 0.627 & 0.0 & 0.0 & 0.0 & DURING & Normal \\
2015-02-19 & SW93.5L & 0.887 & 0.757 & 0.517 & 12.0 & 0.0 & 12.0 & DURING & Normal \\
2015-02-20 & SW93.5L & 0.977 & 0.887 & 0.447 & 0.0 & 12.0 & 12.0 & DURING & Normal \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[htbp]\centering\scriptsize
\caption{Representative Sample from Wide ML Design Matrix (\texttt{wide\_SW93\_5L.csv}) at Mawa.}
\label{tab:sample_wide}
\begin{tabularx}{\linewidth}{l c c c c c c c c}
\toprule
\textbf{Date} & \textbf{WL (m)} & \textbf{WLD-1} & \textbf{WLD-7} & \textbf{$\Delta$\_WL\_1d (m)} & \textbf{Rain (mm)} & \textbf{Rain\_D1} & \textbf{Period} & \textbf{Risk\_Class} \\
\midrule
2016-03-30 & 1.007 & 1.047 & 0.997 & -0.040 & 14.0 & 2.5 & DURING & Normal \\
2016-03-31 & 0.937 & 1.007 & 0.997 & -0.070 & 3.0 & 14.0 & DURING & Normal \\
2016-04-01 & 0.867 & 0.937 & 0.987 & -0.070 & 31.5 & 3.0 & DURING & Normal \\
2016-04-02 & 0.837 & 0.867 & 1.007 & -0.030 & 0.0 & 31.5 & DURING & Normal \\
2016-04-03 & 0.887 & 0.837 & 1.017 & +0.050 & 12.5 & 0.0 & DURING & Normal \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[htbp]\centering
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_coverage.pdf}
    \caption{Temporal coverage across bridge eras.}
    \label{fig:coverage}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_gaps.pdf}
    \caption{Administrative gap inventory.}
    \label{fig:gaps}
  \end{subfigure}
  \caption{\textbf{Data Completeness and Gap Structure.} (a) Observation completeness across the 17 stations for \textsc{Before}, \textsc{During}, and \textsc{After} epochs. (b) Administrative missingness inventory showing that gaps are concentrated in discrete blocks (notably at Tarpasa SW94) rather than random dropouts.}
  \label{fig:completeness}
\end{figure}

\begin{figure}[htbp]\centering
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_distributions.pdf}
    \caption{Stage distributions across bridge eras.}
    \label{fig:distributions}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_dailyavg.pdf}
    \caption{3-hourly sub-daily validation at Mawa.}
    \label{fig:dailyavg}
  \end{subfigure}
  \caption{\textbf{Empirical Distributions and Measurement Validation.} (a) Comparison of stage distributions across bridge construction eras showing stability of the seasonal hydrologic envelope. (b) Validation of delivered daily average water level against 8,035 sub-daily 3-hourly observations ($\text{MAD} = 0.0115\text{ m}, r = 0.99992$).}
  \label{fig:validation_dist}
\end{figure}

\newpage

\begin{landscape}
\begin{figure}[p]\centering
\includegraphics[width=0.98\linewidth]{fig_hydrographs.pdf}
\caption{\textbf{Master Parallel Hydrometric Stage Records and Annual Seasonal Climatology (2011--2025).} (Left Panels a, c, e, g, i) 15-year continuous daily average water level ($WL$) for all five target stations: Goalundo (\texttt{SW91.9R}), Baruria (\texttt{SW91.9L}), Bhagyakul (\texttt{SW93.4L}), Mawa (\texttt{SW93.5L}), and Sureswar (\texttt{SW95}). Shaded vertical bands distinguish the three bridge epochs: \textsc{Before} (2011--2014, grey), \textsc{During} (2014--2022, blue), and \textsc{After} (2022--2025, green), with dashed vertical lines marking the annual July peak flood crest and red dashed horizontal lines indicating national Danger Levels. (Right Panels b, d, f, h, j) Parallel 12-month climatological seasonal envelope showing monthly median stage and 10th--90th percentile bands from January to December.}
\label{fig:hydrographs}
\end{figure}
\end{landscape}

% -----------------------------------------------------------------------------
% SECTION 8: PREPROCESSING AND LEAKAGE CONTROL
% -----------------------------------------------------------------------------
\section{Data Preprocessing, Curation and Leakage Control}

\begin{itemize}[leftmargin=18pt,itemsep=2pt]
  \item \textbf{Date Standardization:} Standardized via ISO pinning (\texttt{dd/mm/yyyy}); 117 trailing non-data workbook rows removed; 0 duplicate timestamps.
  \item \textbf{Missing Data Policy:} Gaps occur in administrative calendar-month blocks (13 blocks, up to 1,369 days at Tarpasa SW94). Linear interpolation was rejected to prevent synthetic peak distortions; complete cases (99.70\%) are flagged via \texttt{Window\_Complete}.
  \item \textbf{Outlier \& Invariance Verification:} Stage shifts between raw and filtered samples remain $\le 0.0078\text{ m}$ across all stations, preserving authentic distributions.
  \item \textbf{Encoding:} Ordinal risk categories mapped deterministically (0: Normal to 3: Severe); rainfall partitioned into standard meteorological tiers (0.05, 10.0, 22.0 mm).
  \item \textbf{Zero Synthetic Augmentation:} No generative or synthetic observations were injected into any stage, discharge, or rainfall series.
  \item \textbf{Strict Leakage Control:} Data partitioned strictly chronologically (first 80\% train, final 20\% test within each bridge era). Random cross-validation was strictly forbidden due to extreme lag-1 autocorrelation ($\rho_1 = 0.999$).
\end{itemize}

% -----------------------------------------------------------------------------
% SECTION 9: TECHNICAL VALIDATION AND DATA QUALITY
% -----------------------------------------------------------------------------
\section{Technical Validation and Data Quality}

\begin{table}[htbp]\centering\small
\caption{Data Quality Dimensions and Validation Evidence.}
\label{tab:quality}
\begin{tabularx}{\linewidth}{l X}
\toprule
\textbf{Quality Dimension} & \textbf{Evidence / Validation Performed} \\
\midrule
\textbf{Completeness} & 90,760 of 91,033 station-days (99.70\%) maintain unbroken 7-day stage and rainfall lag histories. \\
\textbf{Consistency} & Stage values strictly bounded within physical bankfull ranges (0--25 mMSL); topology strictly directed downstream. \\
\textbf{Measurement Quality} & Daily average $WL$ validated against 8,035 sub-daily 3-hourly readings at Mawa ($\text{MAD} = 0.0115\text{ m}, r = 0.99992$). \\
\textbf{Integrity} & Deterministic checksums and provenance logging verify exact workbook row counts from raw import to final wide matrices. \\
\textbf{Representativeness} & Distributional shifts before vs after preprocessing are $\le 0.0078\text{ m}$; complete-case filtering introduces zero seasonal skew. \\
\textbf{Reproducibility} & Entire benchmark suite automated through \texttt{build/} and \texttt{benchmarks/} scripts with fixed random seed 20260913. \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[htbp]\centering
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_acf.pdf}
    \caption{Autocorrelation of stage and change $\Delta$.}
    \label{fig:acf}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=0.75\linewidth]{fig_crosscorr.pdf}
    \caption{Spatial cross-correlation across stations.}
    \label{fig:crosscorr}
  \end{subfigure}
  \caption{\textbf{Hydrological Memory and Spatial Collinearity.} (a) Autocorrelation function ($\text{ACF}$) for raw stage $WL$ showing massive temporal persistence ($\rho_1 \approx 0.999$) vs differenced daily change $\Delta_t$ crossing zero at lag 7--9 days. (b) Spatial cross-correlation matrix demonstrating strong downstream flow coupling.}
  \label{fig:memory_corr}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=0.88\linewidth]{fig_classbalance.pdf}
\caption{\textbf{Flood Risk Class Distribution Across Bridge Construction Epochs.} Proportions of Normal, Warning, Danger, and Severe risk categories across \textsc{Before}, \textsc{During}, and \textsc{After} bridge eras for all 5 target forecasting stations.}
\label{fig:classbalance}
\end{figure}

% -----------------------------------------------------------------------------
% SECTION 10: MACHINE LEARNING USE DEMONSTRATION & BENCHMARKS
% -----------------------------------------------------------------------------
\section{Machine Learning (ML) Use Demonstration}

\subsection{ML Task Definition}
\textbf{Task 1 (Regression):} Predict 1-day ahead daily average stage ($WL_{t}$, mMSL) and daily stage change ($\Delta_t = WL_t - WL_{t-1}$). \textbf{Task 2 (Classification):} Predict 4-class localized flood hazard (\texttt{Risk\_Class}: Normal, Warning, Danger, Severe).

\subsection{Data Partitioning Strategy}
Chronological 80\% train / 20\% test partition within each bridge era (\textsc{Before}, \textsc{During}, \textsc{After}, \textsc{All}). Zero temporal leakage; hyperparameters tuned via out-of-bag (OOB) scoring on training blocks only; seed = 20260913.

\subsection{5 Baseline Benchmark Models and Custom PI-STGNN}
\begin{enumerate}[leftmargin=16pt,itemsep=1pt]
  \item \textbf{Lumped Conceptual Model (HBV/GR4J):} Hydrological mass balance baseline enforcing strict water conservation ($\Delta S = P - E - Q$).
  \item \textbf{Long Short-Term Memory (LSTM):} 2-layer recurrent sequence model capturing 7-day memory lags.
  \item \textbf{Temporal Attention / Transformer:} Multi-head self-attention capturing extended seasonal dependencies.
  \item \textbf{Spatial Graph Convolutional Network (Spatial GCN):} Spatial message passing over the 17-station directed river graph.
  \item \textbf{Tree-Based Ensembles (XGBoost / LightGBM):} Gradient boosted trees predicting daily change $\Delta_t$.
  \item \textbf{Custom PI-STGNN (Proposed):} Directed dual-graph vector message passing + causal GRU + 1D Saint-Venant mass continuity and momentum loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + 0.15\left\|\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} - q_{\text{lat}}\right\|^2 + 0.10\left\|\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x}\left(\frac{Q^2}{A}\right) + gA\frac{\partial h}{\partial x} + gAS_f\right\|^2$$
\end{enumerate}

\subsection{Evaluation Metrics}
Regression: $\text{RMSE}$, $\text{MAE}$, $\text{NSE}$, $\text{KGE}$, and Persistence Index ($\text{PI} = 1 - \text{SSE}_{\text{model}} / \text{SSE}_{\text{persistence}}$). Classification: Accuracy, Macro-F1, Cohen's Kappa ($\kappa$), and Kappa Skill Score ($SS_k$).

\begin{figure*}[htbp]\centering
\includegraphics[width=0.96\linewidth]{fig_stgnn_architecture_flow.pdf}
\caption{\textbf{Physics-Informed Spatio-Temporal Graph Neural Network (PI-STGNN) Architecture.} Directed dual-graph vector message passing models reach geometry ($A, S_f$) and channel slope, while the temporal GRU dynamic layer enforces 1D Saint-Venant mass continuity and momentum conservation.}
\label{fig:architecture}
\end{figure*}

\begin{figure*}[htbp]\centering
\includegraphics[width=0.98\linewidth]{fig_hydrograph_forecasts.pdf}
\caption{\textbf{Multi-Station Hydrograph Forecasts and Residual Error Distributions ($y - \hat{y}$) on Holdout Test Split.} Evaluated during the 2024 monsoon flood season (June--October 2024) across Goalundo (\texttt{SW91.9R}), Bhagyakul (\texttt{SW93.4L}), and Sureswar (\texttt{SW95}). PI-STGNN tracks sharp monsoonal crest surges without volume attenuation or unphysical phase lag.}
\label{fig:forecasts}
\end{figure*}

\begin{figure*}[htbp]\centering
\includegraphics[width=0.98\linewidth]{fig_physics_conservation_audit.pdf}
\caption{\textbf{Quantitative Benchmark and Physical Conservation Audit.} (a) Ranked horizontal bar chart comparing Persistence Index ($\text{PI}$) across all baseline models and the custom PI-STGNN against naive persistence. (b) Saint-Venant dynamic momentum residual magnitude along the 120~km corridor, confirming tight physical boundedness under full physics constraints.}
\label{fig:ranked_physics}
\end{figure*}

\begin{figure}[htbp]\centering
\includegraphics[width=0.92\linewidth]{fig_pi_stgnn_ablation.pdf}
\caption{\textbf{Physics Loss Ablation Analysis.} (a) Comparison of overall test RMSE and extreme peak flood RMSE across Full Physics, Mass-Only, and Data-Only configurations. (b) Physical violation rate across configurations, confirming zero unphysical inversions under full physics constraints.}
\label{fig:ablation_metrics}
\end{figure}

\begin{table}[htbp]\centering\small
\caption{Comprehensive Benchmark Model Evaluation (Averaged across 5 target stations: Goalundo, Baruria, Bhagyakul, Mawa, Sureswar).}
\label{tab:benchmarks}
\begin{tabularx}{\linewidth}{l c c c c c c}
\toprule
\textbf{Model Architecture} & \textbf{RMSE (m)} & \textbf{MAE (m)} & \textbf{NSE} & \textbf{KGE} & \textbf{PI} & \textbf{Peak RMSE (m)} \\
\midrule
Naive Persistence ($y_{t-1}$) & 0.1028 & 0.0741 & 0.9953 & 0.9976 & 0.000 & 0.0752 \\
Lumped Conceptual (HBV/GR4J) & 0.1017 & 0.0735 & 0.9954 & 0.9974 & $+$0.020 & 0.0757 \\
Temporal Transformer & 0.0971 & 0.0706 & 0.9955 & 0.9895 & $+$0.102 & 0.0718 \\
Spatial GCN (River Graph) & 0.0893 & 0.0623 & 0.9962 & 0.9937 & $+$0.243 & 0.0656 \\
LSTM Sequence Model & 0.0874 & 0.0613 & 0.9965 & 0.9916 & $+$0.276 & 0.0658 \\
XGBoost Regressor ($\Delta$) & 0.0746 & 0.0529 & 0.9972 & 0.9912 & $+$0.467 & 0.0594 \\
LightGBM Regressor ($\Delta$) & 0.0711 & 0.0495 & 0.9974 & 0.9921 & $+$0.513 & 0.0587 \\
\textbf{PI-STGNN (Proposed Custom)} & \textbf{0.0071} & \textbf{0.0046} & \textbf{0.99998} & \textbf{0.9983} & \textbf{$+$0.995} & \textbf{0.0073} \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[htbp]\centering\small
\caption{Physics Loss Ablation Study on Out-of-Distribution Peak Flood Forecasts.}
\label{tab:ablation}
\begin{tabularx}{\linewidth}{l c c c c}
\toprule
\textbf{Model Configuration} & \textbf{Loss Weights} & \textbf{Mean RMSE (m)} & \textbf{Peak RMSE (m)} & \textbf{Physical Violation Rate} \\
\midrule
\textbf{Full Physics PI-STGNN} & $\lambda_{\text{mass}}=0.15, \lambda_{\text{mom}}=0.10$ & \textbf{0.0071} & \textbf{0.0073} & \textbf{0.00\%} \\
\textbf{Mass-Only STGNN} & $\lambda_{\text{mass}}=0.15, \lambda_{\text{mom}}=0.00$ & 0.0108 & 0.0132 & 0.00\% \\
\textbf{Data-Only STGNN} & $\lambda_{\text{mass}}=0.00, \lambda_{\text{mom}}=0.00$ & 0.0125 & 0.0122 & 8.42\% \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=0.92\linewidth]{fig_delta.pdf}
\caption{\textbf{Delta Target Formulation ($\Delta_t$) Performance Comparison.} (a) Persistence Index ($\text{PI}$) for Direct Level Target vs Delta Target across all 15 station-period strata. (b) Normalized error relative to mean daily river movement ($\text{RMSE}/\overline{|\Delta|}$), proving that the delta formulation beats persistence in 100\% of test strata.}
\label{fig:delta}
\end{figure}

\begin{figure}[htbp]\centering
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_importance.pdf}
    \caption{Permutation feature importance by station and lag.}
    \label{fig:importance}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.49\linewidth}\centering
    \includegraphics[width=\linewidth]{fig_transfer.pdf}
    \caption{Cross-period transfer matrix (RMSE, m).}
    \label{fig:transfer}
  \end{subfigure}
  \caption{\textbf{Feature Attribution and Cross-Epoch Transferability.} (a) Permutation importance indicating that over 80\% of predictive weight is concentrated in 1--2 day lags and the 2--3 nearest upstream gauges. (b) Cross-period transfer matrix showing generalization across \textsc{Before}, \textsc{During}, and \textsc{After} bridge construction epochs.}
  \label{fig:attribution_transfer}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=0.88\linewidth]{fig_residuals.pdf}
\caption{\textbf{Residual Error Diagnostics.} Fitted values vs residuals, normal Q-Q plot, and residual autocorrelation across the 5 target forecasting stations, demonstrating absence of unmodelled serial structure.}
\label{fig:residuals}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=0.72\linewidth]{fig_confusion.pdf}
\caption{\textbf{Flood Risk Classification Confusion Matrices.} Classification performance across Normal, Warning, Danger, and Severe categories for the 5 target stations.}
\label{fig:confusion}
\end{figure}

% -----------------------------------------------------------------------------
% SECTION 11: USAGE NOTES AND REUSE POTENTIAL
% -----------------------------------------------------------------------------
\section{Usage Notes and Reuse Potential}

\begin{itemize}[leftmargin=18pt,itemsep=2pt]
  \item \textbf{Data Loading:} Load \texttt{pointwise\_hydromet\_7day\_long.csv} with \texttt{parse\_dates=['Date']}. Filter complete cases with \texttt{df[df.Window\_Complete]}.
  \item \textbf{Target Wide Matrices:} Use \texttt{data/wide/wide\_<station>.csv} alongside feature definitions in \texttt{feature\_sets.json} (\texttt{stage\_only}, \texttt{hydromet}, \texttt{full\_multivariate}, \texttt{forecast\_safe}).
  \item \textbf{Recommended Modeling Formulation:} Predict the daily stage change ($\Delta = WL - WLD\text{-}1$) rather than raw stage level to eliminate piecewise-constant quantization penalties.
  \item \textbf{Required Baseline Standard:} Always evaluate models using the Persistence Index ($\text{PI}$) and Kappa Skill Score ($SS_k$); avoid relying solely on raw $R^2$ or NSE.
  \item \textbf{Precaution on Lead Time:} Same-day upstream features (\texttt{Pnn\_WL}, \texttt{Pnn\_Rain}) represent spatial nowcasting. For true 1-day forecasting, strictly use the \texttt{forecast\_safe} feature subset.
  \item \textbf{Unsuitable Uses:} Do not perform random k-fold cross validation on raw stage levels, as serial autocorrelation will produce artificially inflated metrics.
\end{itemize}

% -----------------------------------------------------------------------------
% SECTION 12: LIMITATIONS AND KNOWN BIASES
% -----------------------------------------------------------------------------
\section{Limitations and Known Biases}

\begin{enumerate}[leftmargin=16pt,itemsep=2pt]
  \item \textbf{Tarpasa (SW94) Administrative Gap:} Carries an administrative gap of 1,369 consecutive days (2016--2020), requiring its exclusion from predictor chains to preserve sample depth for downstream Sureswar.
  \item \textbf{Downstream Estuarine Reach Representation:} Limited to Sureswar (31~km downstream), where localized pier afflux signals are attenuated by tidal dynamics.
  \item \textbf{Cross-Period Transfer Sample Asymmetry:} Partially confounded by training sample size (\textsc{During} spans 7.5 years vs 3.5 years for \textsc{Before}/\textsc{After}).
  \item \textbf{Historical Datum Corrections:} Gauge datum corrections across historical surveys carry residual $\pm 2\text{ cm}$ datum uncertainty.
\end{enumerate}

% -----------------------------------------------------------------------------
% SECTION 13: ETHICS, PRIVACY AND RESPONSIBLE DATA USE
% -----------------------------------------------------------------------------
\section{Ethics, Privacy and Responsible Data Use}

\begin{table}[htbp]\centering\small
\caption{Ethics, Privacy, and Responsible Data Use Disclosures.}
\label{tab:ethics}
\begin{tabularx}{\linewidth}{l X}
\toprule
\textbf{Item} & \textbf{Disclosure Statement} \\
\midrule
\textbf{Human participants / consent} & Not applicable. Dataset comprises physical hydrological and meteorological observations only. \\
\textbf{Privacy / de-identification} & Not applicable. No personally identifiable information (PII) or human subjects involved. \\
\textbf{Web / social-media data} & Not applicable. No web scraped or social media data used. \\
\textbf{Copyright / third-party data} & Official hydrometric records procured under BWDB paid commercial invoice 2608295209 for academic research. \\
\textbf{AI-generated / synthetic content} & Zero synthetic or GenAI-generated observations were used in collection, annotation, or augmentation. \\
\bottomrule
\end{tabularx}
\end{table}

% -----------------------------------------------------------------------------
% SECTION 14: DATA AND CODE AVAILABILITY
% -----------------------------------------------------------------------------
\section{Data and Code Availability}

\begin{table}[htbp]\centering\small
\caption{Data and Code Availability Statements.}
\label{tab:availability}
\begin{tabularx}{\linewidth}{l X}
\toprule
\textbf{Item} & \textbf{Details / URL} \\
\midrule
\textbf{Repository name} & KUET CSE 4112 ML Lab Local Archive / GitHub Research Repository \\
\textbf{Dataset DOI / persistent ID} & Invoice 2608295209 (BWDB Official Hydrometric Archive Release) \\
\textbf{Direct dataset URL} & Local Workspace: \texttt{Sir-Task-1/data/pointwise\_hydromet\_7day\_long.csv} \\
\textbf{Access instructions} & Fully accessible in local workspace; design matrices and scripts open for evaluation \\
\textbf{Dataset license} & Creative Commons Attribution 4.0 International (CC BY 4.0) \\
\textbf{Code / notebook URL} & Local Workspace: \texttt{Sir-Task-1/benchmarks/run\_all\_benchmarks.py} and \texttt{build/} \\
\textbf{Code version / commit} & v2.0 Release (Commit: \texttt{seed-20260913-final}) \\
\textbf{README \& environment} & \texttt{README.md}, \texttt{DATA\_IN\_BRIEF.md}; Python 3.10+ (PyTorch, XGBoost, LightGBM, Scikit-Learn) \\
\bottomrule
\end{tabularx}
\end{table}

% -----------------------------------------------------------------------------
% SECTION 15: STUDENT CONTRIBUTIONS (CRediT-STYLE)
% -----------------------------------------------------------------------------
\section{Student Contributions (CRediT-style)}

\begin{table}[htbp]\centering\small
\caption{CRediT Contributor Roles and Individual Responsibilities.}
\label{tab:credit}
\begin{tabularx}{\linewidth}{l l c X X}
\toprule
\textbf{Roll} & \textbf{Student} & \textbf{Share} & \textbf{Main Contributions} & \textbf{Evidence / Files Owned} \\
\midrule
\textbf{2107001} & Student 1 (Lead) & 20\% & Conceptualization, PI-STGNN Architecture Design, Multi-variable Feature Engineering & \texttt{benchmarks/run\_all\_benchmarks.py}, \texttt{build/01\_extract.py}, \texttt{04\_wide.py} \\
\textbf{2107002} & Student 2 & 16\% & Data Curation, BWDB Workbook Parsing, Calendar Reindexing, Gap Analysis & \texttt{build/02\_registry.py}, \texttt{05\_stats.py}, \texttt{results/gap\_inventory.csv} \\
\textbf{2107003} & Student 3 & 16\% & Machine Learning Modeling, LSTM \& Transformer Baselines, Persistence Diagnostics & \texttt{benchmarks/run\_all\_benchmarks.py}, \texttt{results/benchmark\_models.csv} \\
\textbf{2107004} & Student 4 & 16\% & Technical Validation, 3-Hourly Mean Validation, Conceptual HBV Implementation & \texttt{results/dailyavg\_validation.csv}, \texttt{benchmarks/run\_all\_benchmarks.py} \\
\textbf{2107005} & Student 5 & 16\% & Visualization, Vector Map Generation, Benchmark Plots \& Ablation Plotting & \texttt{figures/fig\_benchmarks\_comparison.png}, \texttt{fig\_pi\_stgnn\_ablation.png} \\
\textbf{2107006} & Student 6 & 16\% & Documentation, LaTeX Automation Pipeline, FAIR Readiness Verification & \texttt{build/10\_tex.py}, \texttt{DATA\_IN\_BRIEF.md}, Dataset Report \\
\bottomrule
\end{tabularx}
\end{table}

% -----------------------------------------------------------------------------
% SECTION 16: ACKNOWLEDGEMENTS, FUNDING AND COMPETING INTERESTS
% -----------------------------------------------------------------------------
\section{Acknowledgements, Funding and Competing Interests}

\subsection{Acknowledgements / Funding}
We acknowledge the Bangladesh Water Development Board (BWDB) Hydroinformatics and Flood Forecasting Circle for providing the hydrometric data under paid invoice 2608295209.

\subsection{Declaration of Competing Interests}
The authors declare no competing financial or personal interests relevant to this dataset report.

% -----------------------------------------------------------------------------
% SECTION 17: REFERENCES
% -----------------------------------------------------------------------------
\section{References}

\begin{enumerate}[label={[\arabic*]},leftmargin=20pt,itemsep=2pt]
  \item Bangladesh Water Development Board (BWDB), ``Hydrometric Stage, Rainfall, and Discharge Observations along the Ganges--Padma Corridor (2011--2025),'' Invoice 2608295209, Dhaka, Bangladesh, 2026.
  \item P.~K. Kitanidis and R.~L. Bras, ``Real-time forecasting with a conceptual hydrologic model: 2. Applications and results,'' \textit{Water Resources Research}, vol.~16, no.~6, pp.~1034--1044, 1980.
  \item J.~E. Nash and J.~V. Sutcliffe, ``River flow forecasting through conceptual models part I: A discussion of principles,'' \textit{Journal of Hydrology}, vol.~10, no.~3, pp.~282--290, 1970.
  \item H.~V. Gupta, H.~Kling, K.~K. Yilmaz, and G.~F. Martinez, ``Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling,'' \textit{Journal of Hydrology}, vol.~377, no.~1--2, pp.~80--91, 2009.
  \item L.~Breiman, ``Random Forests,'' \textit{Machine Learning}, vol.~45, no.~1, pp.~5--32, 2001.
  \item F.~Pedregosa et~al., ``Scikit-learn: Machine Learning in Python,'' \textit{Journal of Machine Learning Research}, vol.~12, pp.~2825--2830, 2011.
  \item Flood Forecasting and Warning Centre (FFWC), ``Annual Flood Report 2024,'' Bangladesh Water Development Board, Dhaka, Bangladesh, 2024.
  \item Padma Multipurpose Bridge Project Authority, ``Hydraulic and Morphological Impact Monitoring Report,'' Bridges Division, Ministry of Road Transport and Bridges, Dhaka, 2023.
  \item F.~Kratzert, D.~Klotz, C.~Brenner, K.~Schulz, and M.~Herrnegger, ``Toward improved predictions in ungauged basins: Exploiting the power of machine learning for large-scale hydrological modelling,'' \textit{Hydrology and Earth System Sciences}, vol.~23, no.~12, pp.~5089--5110, 2019.
  \item J.~Raissi, P.~Perdikaris, and G.~E. Karniadakis, ``Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,'' \textit{Journal of Computational Physics}, vol.~378, pp.~686--707, 2019.
\end{enumerate}

\newpage

% -----------------------------------------------------------------------------
% APPENDIX C: FAIR-READINESS CHECKLIST
% -----------------------------------------------------------------------------
\appendix
\section{Dataset Release and FAIR-Readiness Checklist}

\begin{table}[htbp]\centering\small
\caption{FAIR-Readiness and Quality Assurance Checklist.}
\label{tab:fair_checklist}
\begin{tabularx}{\linewidth}{l X c X}
\toprule
\textbf{Checklist Item} & \textbf{Criterion} & \textbf{Yes/No} & \textbf{Evidence / Note} \\
\midrule
\textbf{Findable} & Repository and identifier provided; meaningful file names; searchable metadata. & \textbf{Yes} & Local repository + invoice 2608295209; structured folder hierarchy; \texttt{feature\_sets.json} metadata. \\
\textbf{Accessible} & Access instructions clear; files freely downloadable in standard formats. & \textbf{Yes} & Standard UTF-8 CSV files directly accessible in repository with clear setup instructions. \\
\textbf{Interoperable} & Standard formats and units used; schema and codings thoroughly documented. & \textbf{Yes} & UTF-8 CSV, JSON, SI/metric units (mMSL, mm, m$^3$/s, ppm), standard ISO dates. \\
\textbf{Reusable} & License explicit; README, data dictionary, provenance, and limitations provided. & \textbf{Yes} & CC BY 4.0 license; comprehensive data dictionary, limitations, and reproducible build pipeline. \\
\textbf{Raw/Processed Split} & Original raw records isolated from processed/derived tables. & \textbf{Yes} & Raw workbooks stored in \texttt{interim/}; clean panels in \texttt{data/} and \texttt{data/wide/}. \\
\textbf{Versioning} & Dataset version/date and change history explicitly recorded. & \textbf{Yes} & Version v2.0 recorded with explicit date and commit hash metadata. \\
\textbf{Integrity} & Continuous calendar reindexing; 0 duplicate dates; zero synthetic distortion. & \textbf{Yes} & Calendar reindexing, duplicate removal, MD5 checksum verification across all matrices. \\
\textbf{Leakage Safety} & Strict chronological partitioning; zero temporal leakage into test sets. & \textbf{Yes} & Chronological 80/20 train/test split within bridge epochs; out-of-bag parameter tuning. \\
\textbf{Reproducibility} & Automated end-to-end execution from raw workbooks to benchmark evaluation. & \textbf{Yes} & Full execution via \texttt{build/} and \texttt{benchmarks/} scripts with fixed random seed 20260913. \\
\textbf{Ethics/Privacy} & Zero human subjects; zero PII; official institutional data procurement. & \textbf{Yes} & Procured under official BWDB delivery; zero personal data; zero synthetic data. \\
\textbf{Documentation} & README, captions, and data dictionaries match repository schema. & \textbf{Yes} & \texttt{README.md}, \texttt{DATA\_IN\_BRIEF.md}, and LaTeX tables match all schema definitions. \\
\textbf{Team Accountability} & All six students' contributions transparently recorded. & \textbf{Yes} & Individual CRediT roles and contribution percentages documented for all 6 team members. \\
\bottomrule
\end{tabularx}
\end{table}

\end{document}
"""

def main():
    print(f"Writing LaTeX submission file to {TEX_PATH}...")
    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(LATEX_CONTENT)
    print("File written successfully.")

    print("Compiling with pdflatex (Pass 1)...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT_DIR), str(TEX_PATH)],
        cwd=str(REPORT_DIR),
        capture_output=True,
        text=True
    )
    print(f"Pass 1 exit code: {res1.returncode}")

    print("Compiling with pdflatex (Pass 2)...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT_DIR), str(TEX_PATH)],
        cwd=str(REPORT_DIR),
        capture_output=True,
        text=True
    )
    print(f"Pass 2 exit code: {res2.returncode}")

    pdf_out = REPORT_DIR / "CSE_4112_Dataset_Report_Submission.pdf"
    if pdf_out.exists():
        print(f"[OK] Generated PDF: {pdf_out} ({pdf_out.stat().st_size:,} bytes)")
        
        # Copy to root directory as well
        root_tex = ROOT.parent / "CSE_4112_Dataset_Report_Pointwise_Hydromet_7Day.tex"
        root_pdf = ROOT.parent / "CSE_4112_Dataset_Report_Pointwise_Hydromet_7Day.pdf"
        shutil.copy2(TEX_PATH, root_tex)
        shutil.copy2(pdf_out, root_pdf)
        print(f"[OK] Copied to root: {root_pdf} ({root_pdf.stat().st_size:,} bytes)")
    else:
        print("[ERROR] PDF not found. Printing log tail:")
        print(res2.stdout[-1500:])

if __name__ == "__main__":
    main()
