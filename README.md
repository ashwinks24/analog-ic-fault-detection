# ⚡ Analog IC Fault Detection System
**IIT Kanpur | Ashwin Kumar Singh | Electrical Engineering | Y24**

## Live Demo
🚀 [ic-fault-classifier.streamlit.app](https://ic-fault-classifier.streamlit.app)

---

## What This Project Does
Analog IC faults are hard to detect manually. This project builds an end-to-end ML pipeline that automatically diagnoses faults in a Fully Differential OTA circuit from waveform data.

---

## Circuit Specifications
| Parameter | Value |
|-----------|-------|
| Technology | TSMC 180nm CMOS |
| Supply VDD | 1.8V |
| GBW | 10 MHz |
| Phase Margin | 90° |
| DC Gain | 34 dB |
| THD | 0.136% |
| Tool | ngspice |

---

## Fault Types
| Fault | Description |
|-------|-------------|
| Healthy | Normal baseline |
| M1 Width Fault | Diff pair W reduced 15-25% |
| M2 Width Fault | Opposite asymmetry to M1 |
| M3 Mirror Mismatch | PMOS mirror W +10-20% |
| High Temperature | 100 to 125 degrees C |
| VDD Overvoltage | Supply +5-12% above nominal |

---

## ML Results
| Model | Accuracy |
|-------|----------|
| Random Forest | 100.00% |
| XGBoost | 99.17% |
| MLP PyTorch | 98.89% |
| Decision Tree | 98.33% |
| SVM RBF | 97.50% |
| Naive Bayes | 95.00% |

Production model: XGBoost 99.17% — better generalization across validation splits than Random Forest.

---

## Tech Stack
ngspice, Python, PyTorch, XGBoost, scikit-learn, Streamlit

---

## How to Run Locally
git clone https://github.com/ashwinks24/analog-ic-fault-detection
cd analog-ic-fault-detection
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py

---

## Author
Ashwin Kumar Singh
B.Tech Electrical Engineering, IIT Kanpur, Y24
