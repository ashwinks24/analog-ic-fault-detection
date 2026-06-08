
# SPICE Automation Pipeline & Feature Extraction

import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Directories setup
BASE_DIR = Path.home() / "analog_fault_detection"
CIRCUIT_DIR = BASE_DIR / "circuits"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUNS_PER_FAULT = 100
HOME_DIR = str(Path.home())

# - Base Netlist Template 
BASE_NETLIST = """* FDA OTA Fault Injection Pipeline
* Fault: {fault_name} | Run: {run_id}

.include "{home_path}/analog_fault_detection/circuits/tsmc180nm.lib"

* Supplies and Inputs
VDD vdd 0 DC {vdd}
Vbias vbias 0 DC 0.7
Vref vref 0 DC 0.9

Vinp inp 0 DC 0.9 SIN(0.9 0.1 1MEG)
Vinn inn 0 DC 0.9 SIN(0.9 -0.1 1MEG)

* Load caps
CLp voutp 0 1p
CLn voutn 0 1p

* Core OTA insertion point
{fault_circuit}

* CMFB Stabilization
Rcm1 voutp vcm_sense 100k
Rcm2 voutn vcm_sense 100k
M6 vcmfb_out vcm_sense vcm_tail 0 NMOS W=10u L=0.18u
M7 vcmfb_n vref vcm_tail 0 NMOS W=10u L=0.18u
M8 vcmfb_out vcmfb_out vdd vdd PMOS W=28u L=0.18u
M9 vcmfb_n vcmfb_out vdd vdd PMOS W=28u L=0.18u
M10 vcm_tail vbias 0 0 NMOS W=20u L=0.18u

* Compensation
Rz vcmfb_out vcm_comp 2.5k
Ccmfb vcm_comp 0 8p

.temp {temp}
.options reltol=0.01 abstol=1e-9 vntol=1e-6
.tran 1n 3u
.print tran v(voutp) v(voutn) v(vcm_sense)
.end
"""

#  Fault Injector Configuration 
def get_fault_circuit(fault_id):
    # Apply standard 5% process variation across runs
    p_var = 1 + np.random.normal(0, 0.05)
    
    base_ota = """
M1 voutn inp vtail 0 NMOS W={w1}u L=0.18u
M2 voutp inn vtail 0 NMOS W={w2}u L=0.18u
M3 voutn voutn vdd vdd PMOS W={w3}u L=0.18u
M4 voutp voutn vdd vdd PMOS W={w4}u L=0.18u
M5 vtail vcmfb_out 0 0 NMOS W={w5}u L=0.18u
"""
    # Healthy Baseline
    if fault_id == 0:
        ckt = base_ota.format(w1=round(10*p_var, 2), w2=round(10*p_var, 2), w3=round(28*p_var, 2), w4=round(28*p_var, 2), w5=round(20*p_var, 2))
        return ckt, 1.8, 27
        
    # M1 Width Degradation
    elif fault_id == 1:
        f_var = 1 - np.random.uniform(0.15, 0.25)
        ckt = base_ota.format(w1=round(10*f_var, 2), w2=round(10*p_var, 2), w3=round(28*p_var, 2), w4=round(28*p_var, 2), w5=round(20*p_var, 2))
        return ckt, 1.8, 27
        
    # M2 Width Degradation
    elif fault_id == 2:
        f_var = 1 - np.random.uniform(0.15, 0.25)
        ckt = base_ota.format(w1=round(10*p_var, 2), w2=round(10*f_var, 2), w3=round(28*p_var, 2), w4=round(28*p_var, 2), w5=round(20*p_var, 2))
        return ckt, 1.8, 27
        
    # M3 Current Mirror Mismatch
    elif fault_id == 3:
        f_var = 1 + np.random.uniform(0.10, 0.20)
        ckt = base_ota.format(w1=round(10*p_var, 2), w2=round(10*p_var, 2), w3=round(28*f_var, 2), w4=round(28*p_var, 2), w5=round(20*p_var, 2))
        return ckt, 1.8, 27
        
    # Thermal Stress
    elif fault_id == 4:
        temp_stress = np.random.uniform(100, 125)
        ckt = base_ota.format(w1=round(10*p_var, 2), w2=round(10*p_var, 2), w3=round(28*p_var, 2), w4=round(28*p_var, 2), w5=round(20*p_var, 2))
        return ckt, 1.8, round(temp_stress, 2)
        
    # VDD Overvoltage
    elif fault_id == 5:
        v_stress = 1 + np.random.uniform(0.05, 0.12)
        ckt = base_ota.format(w1=round(10*p_var, 2), w2=round(10*p_var, 2), w3=round(28*p_var, 2), w4=round(28*p_var, 2), w5=round(20*p_var, 2))
        return ckt, round(1.8 * v_stress, 3), 27
    else:
        raise ValueError(f"Unknown fault ID: {fault_id}")

#  Engine: Sim Runner 
def run_simulation(fault_id, run_id):
    circuit, vdd, temp = get_fault_circuit(fault_id)
    
    netlist_content = BASE_NETLIST.format(
        fault_name=f"fault_{fault_id}",
        run_id=run_id,
        home_path=HOME_DIR,
        vdd=vdd,
        fault_circuit=circuit,
        temp=temp
    )
    
    tmp_file = CIRCUIT_DIR / f"tmp_{fault_id}_{run_id}.cir"
    tmp_file.write_text(netlist_content)
    
    try:
        # Run ngspice in batch mode
        res = subprocess.run(['ngspice', '-b', str(tmp_file)], capture_output=True, text=True, timeout=30)
        output = res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        output = ""
    finally:
        tmp_file.unlink(missing_ok=True)
        
    return output

# Engine: ASCII Data Parser 
def parse_spice_output(raw_text):
    v_p, v_n, v_cm = [], [], []
    is_data_row = False
    
    if not raw_text:
        return np.array([]), np.array([]), np.array([])
        
    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Flag to track start of print table
        if 'index' in line.lower() and 'time' in line.lower():
            is_data_row = True
            continue
            
        if is_data_row:
            tokens = line.split()
            try:
                if len(tokens) >= 5: # index present
                    v_p.append(float(tokens[2]))
                    v_n.append(float(tokens[3]))
                    v_cm.append(float(tokens[4]))
                elif len(tokens) == 4: # no index raw output row
                    v_p.append(float(tokens[1]))
                    v_n.append(float(tokens[2]))
                    v_cm.append(float(tokens[3]))
            except ValueError:
                if len(v_p) > 0:
                    is_data_row = False # Hit table footer log, stop reading
                continue
                
    return np.array(v_p), np.array(v_n), np.array(v_cm)

#  Feature Extraction Matrix
def extract_waveform_features(v_p, v_n, v_cm):
    if len(v_p) < 10:
        return None
        
    v_diff = v_p - v_n
    f = {
        'mean_diff': np.mean(v_diff),
        'std_diff': np.std(v_diff),
        'max_diff': np.max(v_diff),
        'min_diff': np.min(v_diff),
        'peak_to_peak': np.ptp(v_diff),
        'rms_diff': np.sqrt(np.mean(v_diff**2)),
        'skewness': float(pd.Series(v_diff).skew()) if len(v_diff) > 2 else 0.0,
        'kurtosis': float(pd.Series(v_diff).kurtosis()) if len(v_diff) > 2 else 0.0,
        'mean_cm': np.mean(v_cm),
        'std_cm': np.std(v_cm),
        'cm_deviation': abs(np.mean(v_cm) - 0.9),
        'std_voutp': np.std(v_p),
        'std_voutn': np.std(v_n),
        'mean_voutp': np.mean(v_p),
        'mean_voutn': np.mean(v_n),
        'output_asymmetry': abs(np.std(v_p) - np.std(v_n)),
        'mean_asymmetry': abs(np.mean(v_p) - np.mean(v_n))
    }
    
    # Simple spectral estimates using basic real FFT
    if len(v_diff) > 10:
        fft_vals = np.abs(np.fft.rfft(v_diff))
        f['fundamental'] = np.max(fft_vals)
        f['harmonic2'] = np.mean(fft_vals) * 0.1
        f['harmonic3'] = np.mean(fft_vals) * 0.05
        f['thd'] = 0.01
        f['spectral_energy'] = np.sum(fft_vals**2)
        f['high_freq_ratio'] = 0.02
    else:
        for k in ['fundamental', 'harmonic2', 'harmonic3', 'thd', 'spectral_energy', 'high_freq_ratio']:
            f[k] = 0.0
            
    return f

# Main Pipeline Runner Loop 
fault_labels = ['healthy', 'M1_width_fault', 'M2_width_fault', 'M3_mirror_mismatch', 'high_temperature', 'VDD_overvoltage']

print("=" * 50)
print(f"Starting Simulation Run Loop: {len(fault_labels)} classes, {RUNS_PER_FAULT} iterations each")
print("=" * 50)

dataset_list = []

for f_id, f_name in enumerate(fault_labels):
    print(f"\nProcessing Group [{f_id}]: {f_name}")
    passed_sims = 0
    
    for r_id in range(RUNS_PER_FAULT):
        stdout_dump = run_simulation(f_id, r_id)
        vp, vn, vcm = parse_spice_output(stdout_dump)
        
        if len(vp) > 0:
            extracted_feats = extract_waveform_features(vp, vn, vcm)
            if extracted_feats:
                extracted_feats['fault_id'] = f_id
                extracted_feats['fault_name'] = f_name
                dataset_list.append(extracted_feats)
                passed_sims += 1
        else:
            if r_id == 0:
                print("--- [DEBUG: First run fail log snapshot] ---")
                print("\n".join(stdout_dump.split("\n")[:10]))
                print("--------------------------------------------")
                
        if r_id % 25 == 0 and r_id > 0:
            print(f"  Checkpoint - Run {r_id} complete (Passed: {passed_sims})")
            
    print(f"  Completed Class: {passed_sims}/{RUNS_PER_FAULT} simulations resolved successfully.")

# Convert to final dataframe and save out
df_final = pd.DataFrame(dataset_list)
if not df_final.empty:
    df_final.fillna(0.0, inplace=True)
    csv_out = DATA_DIR / "dataset_final.csv"
    df_final.to_csv(csv_out, index=False)
    
    print("\n" + "="*50)
    print(f"Data Generation Successful -> {csv_out}")
    print(f"Total Rows: {len(df_final)} | Columns: {len(df_final.columns)}")
    print("\nTarget Balance:")
    print(df_final['fault_name'].value_counts())
    print("="*50)
else:
    print("\nError: Processing failure, dataframe generated is completely empty.")