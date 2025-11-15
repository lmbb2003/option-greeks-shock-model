#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
from scipy.stats import norm  


# In[42]:


# Black-Scholes + Delta, Gamma, Vega
def bs_price_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    # S : spot
    # K : strike
    # T : maturity (y)
    # r : risk-free rate
    # sigma : vol implicite
    # q : dividende
    # option_type : "call" ou "put"
    
    # d1 et d2 formule 
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)) 
    # old try:(np.log(S/K)+(r-q-0.3*sigma**2)*T)/(sigma*np.sqrt(T))  # jsp pourquoi j'avais mis -0.3
    d2 = d1 - sigma * np.sqrt(T) 


    
    # pdf et cdf de la normale
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)
    
    # Prix + Delta en fonction type d'option
    if option_type == "call":
        price = S * np.exp(-q * T) * Nd1 - K * np.exp(-r * T) * Nd2
        delta = np.exp(-q * T) * Nd1
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = np.exp(-q * T) * (Nd1 - 1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'") 
    
    # gamma formule de base pas cherché plus loin
    gamma = (np.exp(-q * T) * nd1) / (S * sigma * np.sqrt(T))
    
    # Vega (sensibilité au sigma) ne pas oublier sqrt(T)!!
    vega = S * np.exp(-q * T) * nd1 * np.sqrt(T)
    
    return price, delta, gamma, vega


# In[43]:


# Paramètres
S0 = 100   #120 too much
K = 100
T = 0.5    # 6 mois 
r = 0.02
sigma = 0.20
q = 0.0
option_type = "call" 

price, delta, gamma, vega = bs_price_greeks(S0, K, T, r, sigma, option_type, q)


print("=== Black-Scholes ===")
print(f"Price : {price:.4f}")
print(f"Delta : {delta:.4f}")
print(f"Gamma : {gamma:.6f}")
print(f"Vega  : {vega:.4f}")


# In[44]:


import pandas as pd  #visu

# Scénarios stress spot
price_shocks = {
    "+5%": 1.05,
    "-5%": 0.95,
    "+10%": 1.10,
    "-10%": 0.90
}

rows = []

try:
    base_price
except:
    # peut vraiment mieux faire honnetement 
    base_price, base_delta, base_gamma, base_vega = bs_price_greeks(
        S0, K, T, r, sigma, option_type, q
    )
    print("(j'avais oublié base_price donc je le recalcule comme un bourrin)")

# stress test vilain comme tout
for nomScenario in price_shocks.keys():    
    
    facteur = price_shocks[nomScenario]     
    Sstress = S0 * facteur                  
    
    
    pr, dlt, gmm, vg = bs_price_greeks(Sstress, K, T, r, sigma, option_type, q) #hasardeux

    pnl_stress = pr - base_price    

    print("Scénario ==> ", nomScenario)
    print("   spot_stress :", Sstress)
    print("   price :", pr, "| pnl:", pnl_stress)
    print("   delta/gamma/vega :", dlt, gmm, vg)
    print("------------------------------")

    rows.append({
        "scenarioName": nomScenario,
        "stressSpot": Sstress,
        "prix": pr,
        "Delta": dlt,
        "Gamma": gmm,
        "Vega": vg,
        "PNL": pnl_stress
    })

# comment ca marche? je n'en ai pas la moindre idée. (lol)

df_price = pd.DataFrame(rows)
print("\n=== Tableau des stress spot ===")
print(df_price)


# In[45]:


# Scénarios stress volatilité 
vol_shocks = {
    "+5pts": 0.05,   # 20% 25%
    "-5pts": -0.05,  # 20% 15%
    "+10pts": 0.10,  # 20% 30%
    "-10pts": -0.10  # 20% 10%
}

rows_vol = []


try:
    base_price
except:
    base_price, base_delta, base_gamma, base_vega = bs_price_greeks(S0, K, T, r, sigma, option_type, q)
    print("(j'avais oublié base_price donc je le recalcule ici aussi...)")

for nom_scen_vol in vol_shocks:    # tjr pas de .item j'ai des outils mais je prefere utiliser mes pieds
    decalage_vol = vol_shocks[nom_scen_vol]  

    sigma_stressé = sigma + decalage_vol   # marrant nan?
    
    if sigma_stressé <= 0:    # sinn BS explose, merci chat gpt
        sigma_stressé = 0.0001   

    try:   
        p_st, d_st, g_st, v_st = bs_price_greeks(S0, K, T, r, sigma_stressé, option_type, q)
    except:
        print("problème bizarre avec sigma =", sigma_stressé)
        p_st, d_st, g_st, v_st = None, None, None, None   # technique de barbare ca

    pnl_st = p_st - base_price

    print("VOL ->", nom_scen_vol)
    print("   nouvelleVol =", sigma_stressé)
    print("   price =", p_st, "  pnl =", pnl_st)
    print("   greeks :", d_st, g_st, v_st)
    print("----------------------------------------")

    rows_vol.append({
        "nomScenarioVol" : nom_scen_vol,    
        "volStressée" : sigma_stressé, # RIP CSV
        "spotRef" : S0,                    
        "prix_après" : p_st,  # incohérent 
        "Delta" : d_st,
        "Gamma" : g_st,
        "Vega" : v_st,
        "PNL_stress" : pnl_st               
    })


df_vol = pd.DataFrame(rows_vol)
print("\n=== Tableau des stress vol ===")
print(df_vol)


# In[50]:


# Fusion price + vol pour avoir le tableay
df_all = pd.concat([df_price, df_vol], ignore_index=True)

# Export CSV
df_all.to_csv("option_greeks_stress_results.csv", index=False) 
print("\n>>> CSV exporté : option_greeks_stress_results.csv")


# In[59]:


import matplotlib.pyplot as plt

# 1 PnL en fonction de S*
plt.figure(figsize=(7,4)) # 7x4 totalement au hasard 
plt.plot(df_price["stressSpot"], df_price["PNL"], marker="o")


for i, row in df_price.iterrows():
    plt.annotate(row["scenarioName"], (row["stressSpot"], row["PNL"]))

plt.axhline(0, linestyle="--")
plt.xlabel("stressed spot")
plt.ylabel("PNL")
plt.title("PnL vs Spot ")
plt.tight_layout()
plt.savefig("pnl_vs_spot.png")

print(">> Graph sauvegardé")


# In[62]:


# 2 PnL en fonction de sigma*
plt.figure(figsize=(7,4))

# X = vol stressée
# Y = PNL stress
plt.plot(df_vol["volStressée"], df_vol["PNL_stress"], marker="o")

# annotations
for i, row in df_vol.iterrows():
    plt.annotate(row["nomScenarioVol"], (row["volStressée"], row["PNL_stress"]))

plt.axhline(0, linestyle="--")
plt.xlabel("Vol stress")
plt.ylabel("PNL stress")
plt.title("PnL vs Vol — Volatility Shocks")
plt.tight_layout()
plt.savefig("pnl_vs_vol.png")

print(">> Graphique pnl_vs_vol.png sauvegardé")


# In[65]:


# 3 comparatif des scénarios
labels = list(df_price["scenarioName"]) + list(df_vol["nomScenarioVol"])

pnl_values = list(df_price["PNL"]) + list(df_vol["PNL_stress"])

types = ["Price"] * len(df_price) + ["Vol"] * len(df_vol)

plt.figure(figsize=(10,5))
x = range(len(labels))

plt.bar(x, pnl_values)

plt.xticks(x, [f"{lab} ({typ})" for lab, typ in zip(labels, types)], rotation=45)
plt.axhline(0, linestyle="--")
plt.ylabel("PnL")
plt.title(" Price vs Vol stress comparison")
plt.tight_layout()
plt.savefig("stress_pnl_barplot.png")

print(">> Graph stress_pnl_barplot.png saved")


# In[48]:


get_ipython().system('jupyter nbconvert --to script option_greeks_shock_model.ipynb')


# In[ ]:




