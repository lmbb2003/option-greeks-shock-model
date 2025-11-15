# option-greeks-shock-model

This is a small project I built to practice option pricing and risk under Black–Scholes.  
The goal was just to understand how Delta/Gamma/Vega react when spot or volatility move.  
Nothing too “quant library” here — just a student project that works and helped me understand things.

Some parts are not perfect (especially in the notebook), but the model runs properly end-to-end.

---

## 1. What the project does

- Black–Scholes pricing for call/put  
- Delta, Gamma, Vega  
- Spot shocks: +5%, -5%, +10%, -10%  
- Vol shocks: +5 vols, -5 vols, +10 vols, -10 vols  
- PnL impact for each scenario  
- Sensitivity table exported to CSV  
- Three plots:
  - PnL vs Spot
  - PnL vs Vol
  - Scenario comparison barplot

The implementation is voluntarily straightforward so I could follow every step.

---


I keep the notebook on a separate branch (`ipynb-version`) because it contains tests, comments, and some trial-and-error.

---

## 2. How to run it

### Run the Python script: python src/option_greeks_shock_model.py

All plots and the CSV are saved automatically in the `outputs/` folder.

---

## 3. Notes

- Not everything is fully optimized — the main goal was learning, not writing a perfect module.
- Some variable names and comments (especially in the notebook) are inconsistent because I tried things as I went.
- The script version (`.py`) is much cleaner than the notebook version.
- This is basic Black–Scholes only (no vol smile, no stochastic vol, etc.).

---

## 4. Why I built it

I wanted to practice:

- Black–Scholes  
- Greeks behaviour  
- Scenario analysis  
- PnL interpretation  
- Plotting and reporting  
- Structuring a small project properly

---

## 5. Conclusion

It’s a simple but complete student project on option risk under market shocks.  
Everything runs correctly, the Greeks behave as expected, and the plots help visualize the risk.

I may add Rho, Theta or a more complete scenario grid later, but for now the project does what I wanted.




