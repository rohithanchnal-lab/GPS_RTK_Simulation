# Automotive GNSS Simulation: GPS vs. RTK
**Developed by Neha Adhikari and Rohit Hanchnal**   
*Part of Coursework for ESE 525: Modern Sensors in AI Applications*

## Project Context
This project explores the implementation of **Carrier-Phase RTK (Real-Time Kinematic)** sensors to provide the high-precision required for autonomous decision-making algorithms.

## Technical Implementation

Unlike standard code-phase sensors that rely on the coarse C/A code, this simulation models the **Carrier Phase** of the $L1$ signal. 
* **Precision:** By utilizing the ~19cm wavelength of the carrier signal, we move from meter-level to centimeter-level granularity.
* **Double-Differencing:** We simulate the cancellation of ionospheric and tropospheric delays by differencing signals between the rover, a base station, and multiple satellites.

## 📊 Sensor Performance Analysis

Standard GNSS yields a ~5.00m error due to atmospheric delays and code-phase limitations, failing to maintain vehicle lanes. RTK+Kalman reduces this to ~0.02m, providing the centimeter-level precision required for safe autonomous navigation.