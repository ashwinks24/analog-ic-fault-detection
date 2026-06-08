# Analog IC Fault Detection System
**IIT Kanpur | Ashwin Kumar Singh | EE Department**

## Overview
ML-based automated fault diagnosis for Fully 
Differential OTA circuits using ngspice simulations.

## Results
| Model | Accuracy |
|-------|----------|
| Random Forest | 100% |
| XGBoost | 99.17% |
| MLP (PyTorch) | 98.89% |
| Decision Tree | 98.33% |
| SVM (RBF) | 97.50% |
| Naive Bayes | 95.00% |

## Circuit Specs
- Technology: TSMC 180nm CMOS
- GBW: 10 MHz | PM: 90° | THD: 0.136%
- Tool: ngspice

## Pipeline
FDA OTA Design → Fault Injection (600 sims) → 
Feature Extraction (23 features) → 
ML Classification → Streamlit Dashboard

## Tech Stack
ngspice · Python · PyTorch · XGBoost · Streamlit

## Live Demo
Coming soon
