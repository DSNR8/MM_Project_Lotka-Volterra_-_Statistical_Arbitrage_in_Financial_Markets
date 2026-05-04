#Project for MM 
#pct_change()
# Title: Μοντέλα δυναμικών συστημάτων με βάση το Lotka-Voltera 

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore') # Κρύβει τα ενοχλητικά κόκκινα μηνύματα
from statsmodels.tsa.stattools import coint

# ==========================================
# 0. ΣΥΝΑΡΤΗΣΕΙΣ ΓΙΑ ΤΑ METHODOLOGY SLIDES
# ==========================================
def plot_predator_prey_slide():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off') 
    plt.text(0.5, 0.95, "Μεθοδολογία: Μοντέλο Θηρευτή - Λείας (SPY vs VIX)", fontsize=16, fontweight='bold', ha='center', va='top', color='darkred')
    
    eq_text = (
        r"$\mathbf{Εξισώσεις\ Θηρευτή-Λείας:}$" + "\n\n"
        r"Λεία (SPY):   $\frac{1}{SPY} \frac{d(SPY)}{dt} = \alpha - \beta(VIX)$" + "\n\n"
        r"Θηρευτής (VIX): $\frac{1}{VIX} \frac{d(VIX)}{dt} = -\gamma + \delta(SPY)$"
    )
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#fff3e0", ec="orange", lw=1.5)
    plt.text(0.05, 0.80, eq_text, fontsize=14, ha='left', va='top', bbox=bbox_props)
    
    vars_text = (
        "Επεξήγηση Μεταβλητών (Οικολογικά vs Χρηματοοικονομικά):\n\n"
        "• α (Ανάπτυξη Λείας): Φυσικός ρυθμός ανάπτυξης (S&P 500).\n"
        "• β (Ζημιά από Θηρευτή): Αρνητική επίδραση του φόβου (VIX).\n"
        "• γ (Μείωση Θηρευτή): Ο ρυθμός που υποχωρεί η μεταβλητότητα.\n"
        "• δ (Τροφοδότηση): Η άνοδος της αγοράς που 'ταΐζει' τον φόβο."
    )
    plt.text(0.05, 0.50, vars_text, fontsize=12, ha='left', va='top', color='black')
    
    method_text = (
        "Τρόπος Επίλυσης:\n\n"
        "Απλή Γραμμική Παλινδρόμηση \n"
        "(Linear Regression):\n\n"
        "1. Η χθεσινή τιμή του VIX \n"
        "   προβλέπει τη σημερινή \n"
        "   απόδοση του SPY.\n\n"
        "2. Η χθεσινή τιμή του SPY \n"
        "   επηρεάζει τη σημερινή \n"
        "   αλλαγή στον φόβο."
    )
    bbox_method = dict(boxstyle="square,pad=0.8", fc="#e8f5e9", ec="green", lw=1.5)
    plt.text(0.65, 0.80, method_text, fontsize=12, ha='left', va='top', bbox=bbox_method)
    plt.tight_layout()
    plt.show()

def plot_methodology_slide():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off') 
    plt.text(0.5, 0.95, "Μεθοδολογία: Ανταγωνιστικό Μοντέλο Lotka-Volterra στις Αγορές", fontsize=16, fontweight='bold', ha='center', va='top', color='darkblue')
    
    eq_text = (
        r"$\mathbf{Γενικές\ Εξισώσεις\ Ανταγωνισμού:}$" + "\n\n"
        r"$\frac{1}{KO} \frac{d(KO)}{dt} = r_1 - a_{11}(KO) - a_{12}(PEP)$" + "\n"
        r"$\frac{1}{PEP} \frac{d(PEP)}{dt} = r_2 - a_{22}(PEP) - a_{21}(KO)$"
    )
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="gray", lw=1)
    plt.text(0.05, 0.80, eq_text, fontsize=14, ha='left', va='top', bbox=bbox_props)
    
    vars_text = (
        "Επεξήγηση Μεταβλητών:\n\n"
        "• Αριστερό Μέλος: Απόδοση (Return) μετοχής.\n"
        "• r1, r2 (Εγγενής Ανάπτυξη): Ρυθμός χωρίς ανταγωνισμό.\n"
        "• a11, a22 (Αυτο-περιορισμός): 'Ταβάνι' αγοράς (κορεσμός).\n"
        "• a12, a21 (Ανταγωνιστική Πίεση): Πόσο η μία κόβει πωλήσεις από την άλλη."
    )
    plt.text(0.05, 0.50, vars_text, fontsize=12, ha='left', va='top', color='black')
    
    method_text = (
        "Τρόπος Επίλυσης:\n\n"
        "Προσεγγίζουμε τις \n"
        "διαφορικές εξισώσεις μέσω \n"
        "Πολλαπλής Γραμμικής \n"
        "Παλινδρόμησης (Multiple \n"
        "Linear Regression).\n\n"
        "Η απόδοση (Y) εξαρτάται \n"
        "από τις καθυστερημένες \n"
        "τιμές των δύο μετοχών \n"
        "(X1, X2)."
    )
    bbox_method = dict(boxstyle="square,pad=0.8", fc="#e3f2fd", ec="#1e88e5", lw=1.5)
    plt.text(0.65, 0.80, method_text, fontsize=12, ha='left', va='top', bbox=bbox_method)
    plt.tight_layout()
    plt.show()

# ==========================================
# 1. ΚΑΤΕΒΑΣΜΑ ΔΕΔΟΜΕΝΩΝ
# ==========================================
print("Κατέβασμα Δεδομένων...")
tickers = ["SPY", "^VIX", "KO", "PEP"]
data = yf.download(tickers, start="2019-01-01", end="2024-01-01")['Close'].dropna()

scaler = StandardScaler()
window_smooth = 21 

# ==========================================
# 2. ΘΗΡΕΥΤΗΣ vs ΛΕΙΑ (SPY vs VIX)
# ==========================================
plot_predator_prey_slide()

spy_vix = data[['SPY', '^VIX']].copy()
spy_vix_smoothed = spy_vix.rolling(window=window_smooth).mean().dropna()

log_spy_vix_smoothed = np.log(spy_vix_smoothed)

returns_spy_vix = log_spy_vix_smoothed.diff()
X_spy_vix = log_spy_vix_smoothed.shift(1)

# Σωστή ευθυγράμμιση και Naming
df_pv = pd.concat([returns_spy_vix, X_spy_vix], axis=1).dropna()
df_pv.columns = ['SPY_ret', 'VIX_ret', 'SPY_lag', 'VIX_lag']

Y_spy = df_pv['SPY_ret']
Y_vix = df_pv['VIX_ret']
X_spy = df_pv[['SPY_lag']] # Διορθώθηκε για να ταιριάζει σωστά στο fit
X_vix = df_pv[['VIX_lag']] # Διορθώθηκε για να ταιριάζει σωστά στο fit

# Fit με τα aligned δεδομένα
reg_prey = LinearRegression().fit(X_vix, Y_spy)
alpha, beta = reg_prey.intercept_, -reg_prey.coef_[0]

reg_predator = LinearRegression().fit(X_spy, Y_vix)
gamma, delta = -reg_predator.intercept_, reg_predator.coef_[0]

text_spy_vix = '\n'.join((
    r'$\alpha$ (Ανάπτυξη SPY): %.5f' % alpha,
    r'$\beta$ (Ζημιά από VIX): %.5f' % beta,
    r'$\gamma$ (Μείωση VIX): %.5f' % gamma,
    r'$\delta$ (Τροφοδότηση VIX): %.5f' % delta
))

spy_vix_norm = pd.DataFrame(scaler.fit_transform(spy_vix_smoothed), columns=spy_vix_smoothed.columns, index=spy_vix_smoothed.index)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(spy_vix_norm['SPY'], spy_vix_norm['^VIX'], c=range(len(spy_vix_norm)), cmap='viridis', s=10, alpha=0.8)
plt.plot(spy_vix_norm['SPY'], spy_vix_norm['^VIX'], color='gray', alpha=0.3)
plt.title('Market Phase Portrait: SPY (Prey) vs VIX (Predator)')
plt.xlabel('SPY (Normalized)')
plt.ylabel('VIX (Normalized)')
plt.colorbar(scatter, label='Χρόνος (Ημέρες από την αρχή)')
plt.grid(True)
props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.gca().text(0.95, 0.95, text_spy_vix, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props1)
plt.show()


# ==========================================
# 3. ΑΝΤΑΓΩΝΙΣΤΙΚΟ ΜΟΝΤΕΛΟ (KO vs PEP)
# ==========================================
plot_methodology_slide()

ko_pep = data[['KO', 'PEP']].copy()
ko_pep_smoothed = ko_pep.rolling(window=window_smooth).mean().dropna()

log_ko_pep = np.log(ko_pep_smoothed)

returns_ko_pep = log_ko_pep.diff() 
X_ko_pep = log_ko_pep.shift(1)

# Σωστή ευθυγράμμιση και Naming
df_lv = pd.concat([returns_ko_pep, X_ko_pep], axis=1).dropna()
df_lv.columns = ['KO_ret', 'PEP_ret', 'KO_lag', 'PEP_lag']

Y_ko = df_lv['KO_ret']
Y_pep = df_lv['PEP_ret']
X = df_lv[['KO_lag', 'PEP_lag']]

# Fit με τα aligned δεδομένα
reg_ko = LinearRegression().fit(X, Y_ko)
r1, a11, a12 = reg_ko.intercept_, -reg_ko.coef_[0], -reg_ko.coef_[1]

reg_pep = LinearRegression().fit(X, Y_pep)
r2, a21, a22 = reg_pep.intercept_, -reg_pep.coef_[0], -reg_pep.coef_[1]

text_combined = ('Coke (KO):\n' + r'$r_1$: %.5f, $a_{11}$: %.5f, $a_{12}$: %.5f' % (r1, a11, a12) + 
                 '\n\n' + 
                 'Pepsi (PEP):\n' + r'$r_2$: %.5f, $a_{22}$: %.5f, $a_{21}$: %.5f' % (r2, a22, a21))

ko_pep_norm = pd.DataFrame(scaler.fit_transform(ko_pep_smoothed), columns=ko_pep_smoothed.columns, index=ko_pep_smoothed.index)

plt.figure(figsize=(8, 6))
plt.plot(ko_pep_norm['KO'], ko_pep_norm['PEP'], color='purple', alpha=0.5)
plt.title('Ανταγωνιστικό Lotka-Volterra\n(KO vs PEP)')
plt.xlabel('Coca-Cola (Normalized)')
plt.ylabel('Pepsi (Normalized)')
plt.grid(True)
props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.gca().text(0.05, 0.95, text_combined, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props2)
plt.tight_layout()
plt.show()


# ==========================================
# 4. BACKTESTING: ΑΠΛΟ vs FACTOR-NEUTRAL
# ==========================================
print("Υπολογισμός Αλγορίθμου Trading...")

window_bt = 120 # Παράθυρο 120 ημερών για αποφυγή θορύβου

pos_old = np.zeros(len(data))
pos_new = np.zeros(len(data))
beta_pep_old_arr = np.zeros(len(data))
beta_pep_new_arr = np.zeros(len(data))
beta_spy_new_arr = np.zeros(len(data))

curr_old = 0
curr_new = 0

log_ko = np.log(data['KO']).values
log_pep = np.log(data['PEP']).values
log_spy = np.log(data['SPY']).values

for i in range(window_bt, len(data)):
    # --- ΙΣΤΟΡΙΚΑ ΔΕΔΟΜΕΝΑ (120 μέρες) ---
    y_train = log_ko[i-window_bt:i]
    x_old_train = log_pep[i-window_bt:i].reshape(-1, 1)
    x_new_train = np.column_stack((log_pep[i-window_bt:i], log_spy[i-window_bt:i]))
    
    # --- ΤΟ ΣΗΜΕΡΑ (Μόνο η 1 μέρα) ---
    y_today = log_ko[i]
    x_pep_today = log_pep[i]
    x_spy_today = log_spy[i]
    
    # --- ΜΟΝΤΕΛΟ 1: ΑΠΛΟ PAIRS TRADING (KO vs PEP) ---
    reg_old = LinearRegression().fit(x_old_train, y_train)
    res_old_train = y_train - reg_old.predict(x_old_train)
    std_old = np.std(res_old_train)
    
    pred_old_today = reg_old.predict(np.array([[x_pep_today]]))[0]
    z_old = (y_today - pred_old_today) / std_old if std_old > 0 else 0
    beta_pep_old_arr[i] = reg_old.coef_[0]
    
    if z_old < -2.0: curr_old = 1
    elif z_old > 2.0: curr_old = -1
    elif abs(z_old) < 0.5: curr_old = 0
    pos_old[i] = curr_old

    # --- ΜΟΝΤΕΛΟ 2: FACTOR NEUTRAL (KO vs PEP + SPY) ---
    reg_new = LinearRegression().fit(x_new_train, y_train)
    res_new_train = y_train - reg_new.predict(x_new_train)
    std_new = np.std(res_new_train)
    
    pred_new_today = reg_new.predict(np.array([[x_pep_today, x_spy_today]]))[0]
    z_new = (y_today - pred_new_today) / std_new if std_new > 0 else 0
    
    beta_pep_new_arr[i] = reg_new.coef_[0]
    beta_spy_new_arr[i] = reg_new.coef_[1]
    
    if z_new < -2.0: curr_new = 1
    elif z_new > 2.0: curr_new = -1
    elif abs(z_new) < 0.5: curr_new = 0
    pos_new[i] = curr_new

# ==========================================
# 5. ΑΠΟΔΟΣΕΙΣ ΚΑΙ ΓΡΑΦΗΜΑ
# ==========================================
data['Pos_Old'] = pos_old
data['Pos_New'] = pos_new

ret_ko = np.log(data['KO']).diff() 
ret_pep = np.log(data['PEP']).diff()
ret_spy = np.log(data['SPY']).diff()

data['Strat_Old'] = data['Pos_Old'].shift(1) * (ret_ko - pd.Series(beta_pep_old_arr, index=data.index).shift(1) * ret_pep)
data['Strat_New'] = data['Pos_New'].shift(1) * (ret_ko - pd.Series(beta_pep_new_arr, index=data.index).shift(1) * ret_pep - pd.Series(beta_spy_new_arr, index=data.index).shift(1) * ret_spy)

data['Cum_Old'] = 100 * (1 + data['Strat_Old'].fillna(0)).cumprod()
data['Cum_New'] = 100 * (1 + data['Strat_New'].fillna(0)).cumprod()
data['Cum_SPY'] = 100 * (1 + ret_spy.fillna(0)).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Cum_SPY'], label='S&P 500 (Buy & Hold)', color='blue', alpha=0.3, linestyle='--')
plt.plot(data.index, data['Cum_Old'], label='Παλιό Μοντέλο (Απλό Pairs Trading)', color='red', alpha=0.7)
plt.plot(data.index, data['Cum_New'], label='Νέο Μοντέλο (Factor Neutral)', color='green', linewidth=2.5)

plt.title('Σύγκριση Pairs Trading: Απλό vs Ουδέτερο προς την Αγορά', fontsize=15, fontweight='bold')
plt.ylabel('Αξία Χαρτοφυλακίου (Αρχικό: 100€)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

print("\n" + "="*45)
print("ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΡΟΣΟΜΟΙΩΣΗΣ (Αρχικό: 100€)")
print("="*45)
print(f"S&P 500 (Αγορά):               {data['Cum_SPY'].iloc[-1]:.2f} €")
print(f"Παλιό Μοντέλο (Απλό):          {data['Cum_Old'].iloc[-1]:.2f} €")
print(f"Νέο Μοντέλο (Factor Neutral):  {data['Cum_New'].iloc[-1]:.2f} €")
print("="*45)

print("\n--- Έλεγχος Cointegration ---")
score, pvalue, _ = coint(np.log(data['KO']), np.log(data['PEP']))
print(f"Cointegration p-value: {pvalue:.4f}")
if pvalue < 0.05:
    print("✅ Pair likely cointegrated → Strategy likely reliable at < 0.05")
elif pvalue >= 0.05 and pvalue < 0.1:
    print("⚠️ Pair NOT cointegrated at (0.05) but likely cointegrated at (0.1) → strategy unreliable (0.05) but likely reliable at (0.1)")
elif pvalue >= 0.1:
    print("⚠️ Pair NOT cointegrated → strategy unreliable (0.1)")

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ΠΑΡΑΜΕΤΡΟΙ EULER-LAGRANGE (OPTIMAL CONTROL)
# ==========================================
Q0 = 10000      # Συνολικές μετοχές που πρέπει να αγοράσουμε (Το σήμα από το Z-Score)
T = 1.0         # Διαθέσιμος χρόνος (π.χ. 1 ημέρα trading = 6.5 ώρες)
N = 100         # Βήματα χρόνου (π.χ. κάθε 4 λεπτά)
t = np.linspace(0, T, N)

# Παράμετροι Αγοράς
kappa = 0.01    # Συντελεστής Market Impact (Πόσο "σκληρή" είναι η αγορά)
phi = 0.5       # Συντελεστής Ρίσκου (Πόσο φοβόμαστε την αβεβαιότητα)

# Υπολογισμός του γάμμα (Η λύση της διαφορικής εξίσωσης)
gamma = np.sqrt(phi / kappa)

# ==========================================
# ΥΠΟΛΟΓΙΣΜΟΣ ΔΙΑΔΡΟΜΩΝ
# ==========================================

# 1. Naive Execution (Ο "Άσχετος"): Αγοράζει τα πάντα στο λεπτό 0.
q_naive = np.zeros_like(t)
q_naive[0] = Q0 

# 2. TWAP / Γραμμική (Ο "Απλός" Αλγόριθμος): Αγοράζει σταθερή ποσότητα κάθε λεπτό
q_twap = Q0 * (1 - t/T)

# 3. Euler-Lagrange Optimal Execution (Το "Hedge Fund"):
# Η αναλυτική λύση της διαφορικής εξίσωσης q''(t) - gamma^2 * q(t) = 0 με συνοριακές συνθήκες
q_optimal = Q0 * np.sinh(gamma * (T - t)) / np.sinh(gamma * T)

# Υπολογισμός Ταχύτητας Συναλλαγών (Πόσες αγοράζουμε σε κάθε βήμα: -dq/dt)
trades_twap = -np.gradient(q_twap)
trades_optimal = -np.gradient(q_optimal)

# ==========================================
# ΟΠΤΙΚΟΠΟΙΗΣΗ
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Γράφημα 1: Υπόλοιπο Μετοχών q(t) ---
ax1.plot(t, q_naive, label='Αφελής Εκτέλεση (Όλα στο 0)', color='red', linestyle=':')
ax1.plot(t, q_twap, label='Γραμμική TWAP', color='blue', linestyle='--')
ax1.plot(t, q_optimal, label='Euler-Lagrange Optimal', color='green', linewidth=3)
ax1.set_title('Φθίνουσα Πορεία Αποθέματος q(t)')
ax1.set_xlabel('Χρόνος t (Μέσα στη μέρα)')
ax1.set_ylabel('Υπολειπόμενες Μετοχές για Αγορά')
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- Γράφημα 2: Ταχύτητα Συναλλαγών (Η Δράση) ---
ax2.plot(t, trades_twap, label='Ταχύτητα TWAP (Σταθερή)', color='blue', linestyle='--')
ax2.plot(t, trades_optimal, label='Ταχύτητα Euler-Lagrange', color='green', linewidth=3)
ax2.set_title('Ρυθμός Συναλλαγών / Ταχύτητα $\dot{q}(t)$')
ax2.set_xlabel('Χρόνος t (Μέσα στη μέρα)')
ax2.set_ylabel('Ποσότητα αγοράς ανά βήμα')
ax2.fill_between(t, 0, trades_optimal, color='green', alpha=0.1)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('Βέλτιστη Εκτέλεση μέσω Αρχής Ελάχιστης Δράσης (Euler-Lagrange)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

