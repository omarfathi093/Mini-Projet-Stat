#%%
#   Mini Projet 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# chargement de donnees
df = pd.read_excel('pcb_dataset_1.xlsx')
n, p = df.shape
print(f"Dataset : {n} observations, {p} variables")

# variables quantitatives
var_quant = ['resistance_ohm', 'capacitance_uF', 'temperature_test_C',
             'voltage_drop_V', 'current_leak_mA', 'signal_noise_dB',
             'power_consumption_W', 'response_time_ms']

# variable groupe pour l'ADL
col_group = 'defect'


#%%

# description 
print("\nTypes des variables")
print(df.dtypes)

# on convertit les variables qualitatives en category
df['component_type'] = df['component_type'].astype('category')
df['production_line'] = df['production_line'].astype('category')
df['supplier']        = df['supplier'].astype('category')
df['defect']          = df['defect'].astype('category')
df['defect_type']     = df['defect_type'].astype('category')

print("\nSTATISTIQUES DESCRIPTIVES (variables quant.)")
print(df[var_quant].describe())
# Ecart-type de 453ohm pour une moyenne de 422ohm et 103uF pour une moyenne de 69uF 
# Il faudra faire une analyse par type de composant car l'écart-type très élevé pour resistance_ohm et capacitance_uF (populations hétérogènes)

print("\nValeurs manquantes")
print(df.isnull().sum())


#%%

# ANALYSE UNIVARIEE


# Distribution des variables qualitatives
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ['component_type', 'production_line', 'supplier']):
    df[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Distribution : {col}')
    ax.set_xlabel('')
plt.suptitle("Variables qualitatives")
plt.tight_layout()
plt.show()
# CRITIQUE : Les 4 fournisseurs sont bien équilibrés (~3700 chacun) → pas de biais de
# sous-représentation. Line_C a 38% de moins d'observations que Line_A : à surveiller
# si on compare les taux de défauts par ligne.

# Taux de défauts global
print("\n===== TAUX DE DEFAUTS =====")
print(df['defect'].value_counts())
print(f"Taux de défauts global : {(df['defect']=='FAIL').mean()*100:.1f}%")
# CRITIQUE : 48% de défauts, c'est anormalement élevé pour un vrai processus industriel
# (on attend <5%). Ce dataset semble conçu pour l'apprentissage supervisé avec
# classes équilibrées → bon pour l'ADL, mais ne reflète pas une réalité industrielle.

# Boxplots des variables quantitatives par statut de défaut
df[var_quant + ['defect']].boxplot(column=var_quant, by='defect', figsize=(16, 8))
plt.suptitle("Distributions par statut (OK vs FAIL)")
plt.tight_layout()
plt.show()
# CRITIQUE : Si les distributions OK et FAIL se chevauchent beaucoup pour une variable,
# celle-ci aura peu de pouvoir discriminant seule. L'ADL combinera toutes les variables.


#%%
# ==============================================================
# 3/ ANALYSE PAR GROUPE (type de composant et fournisseur)
# ==============================================================

print("\n===== MOYENNES PAR TYPE DE COMPOSANT =====")
print(df.groupby('component_type')[var_quant].mean().round(2))
# CRITIQUE : Les moyennes sont très différentes selon le type → confirme qu'une
# analyse globale mélange des populations hétérogènes.

print("\n===== TAUX DE DEFAUTS PAR LIGNE DE PRODUCTION =====")
taux = df.groupby('production_line')['defect'].apply(lambda x: (x=='FAIL').mean()*100)
print(taux.round(1))
# CRITIQUE : Si Line_C a un taux significativement plus élevé, cela pointe vers
# un problème de processus sur cette ligne (calibration, opérateur, matériel...).

print("\n===== TAUX DE DEFAUTS PAR FOURNISSEUR =====")
taux_sup = df.groupby('supplier')['defect'].apply(lambda x: (x=='FAIL').mean()*100)
print(taux_sup.round(1))
# CRITIQUE : Des taux différents entre fournisseurs révèle une qualité inégale.
# C'est une information clé pour la sélection/qualification des fournisseurs.


#%%
# ==============================================================
# 4/ ANALYSE BIVARIEE - CORRELATIONS
# ==============================================================

C = df[var_quant].corr(method='pearson')
print("\n===== MATRICE DE CORRELATION =====")
print(C.round(2))

plt.figure(figsize=(9, 7))
sns.heatmap(C, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Matrice de corrélation - variables quantitatives")
plt.tight_layout()
plt.show()
# CRITIQUE : Des corrélations fortes (|r| > 0.7) indiquent des redondances.
# En ADL l'homoscédasticité est une hypothèse importante : vérifier que les
# structures de variance-covariance sont similaires entre groupes OK et FAIL.


#%%
# ==============================================================
# 5/ INTERVALLE DE CONFIANCE SUR current_leak_mA
# ==============================================================
# On estime la moyenne de current_leak_mA pour les composants FAIL

X_fail = df[df['defect'] == 'FAIL']['current_leak_mA'].dropna()
n_f    = len(X_fail)
mu_est = X_fail.mean()
sigma  = X_fail.std()  # on utilise std empirique (sigma inconnu)

ic = norm.interval(0.95, loc=mu_est, scale=sigma / np.sqrt(n_f))
print(f"\n===== IC 95% - current_leak_mA (composants FAIL) =====")
print(f"n = {n_f}, µ_estimé = {mu_est:.4f} mA")
print(f"IC = [{ic[0]:.4f} ; {ic[1]:.4f}] mA")
# CRITIQUE : Avec n=7221 observations, l'IC est très étroit → la moyenne est
# estimée avec grande précision. Un IC étroit n'implique pas que la distribution
# est normale : vérifier visuellement avec l'histogramme ci-dessous.

plt.figure()
plt.hist(X_fail, bins=40, density=True, alpha=0.7, label='FAIL')
plt.axvline(mu_est, color='red', label=f'Moyenne = {mu_est:.3f}')
plt.axvline(ic[0],  color='black', linestyle='--', label='IC 95%')
plt.axvline(ic[1],  color='black', linestyle='--')
plt.xlabel("current_leak_mA")
plt.ylabel("Densité")
plt.title("Distribution current_leak_mA (FAIL) + IC 95%")
plt.legend()
plt.show()


#%%
# ==============================================================
# 6/ ADL - ANALYSE DISCRIMINANTE LINEAIRE (OK vs FAIL)
# ==============================================================
# Objectif : trouver les axes qui séparent le mieux OK et FAIL
# On a K=2 classes → au maximum K-1 = 1 axe discriminant

Xdf = df[var_quant + [col_group]].dropna().reset_index(drop=True)

lda   = LinearDiscriminantAnalysis()
Xlda  = lda.fit_transform(Xdf[var_quant], Xdf[col_group])

# Avec K=2 classes, on n'a qu'1 seul axe discriminant
plt.figure()
for g, color in zip(['OK', 'FAIL'], ['blue', 'red']):
    idx = Xdf[col_group] == g
    plt.hist(Xlda[idx, 0], bins=50, alpha=0.5, label=g, color=color, density=True)
plt.xlabel("Composante discriminante 1")
plt.title("Distribution sur l'axe discriminant (OK vs FAIL)")
plt.legend()
plt.show()
# CRITIQUE : Si les deux distributions se chevauchent peu → bonne séparation.
# Si elles se chevauchent beaucoup → les variables mesurées ne suffisent pas
# à discriminer clairement OK et FAIL (erreur de classement élevée).

# Taux de bonne classification (sur les données d'apprentissage)
score = lda.score(Xdf[var_quant], Xdf[col_group])
print(f"\n===== SCORE ADL (données apprentissage) =====")
print(f"Taux de bonne classification : {score*100:.1f}%")
# CRITIQUE IMPORTANTE : Ce score est calculé sur les mêmes données que
# l'apprentissage → il est optimiste (biais). Pour une évaluation honnête,
# il faudrait utiliser une validation croisée ou un jeu de test séparé.


#%%
# ==============================================================
# 7/ ADL - PAR TYPE DE COMPOSANT (4 classes)
# ==============================================================
# Ici on discrimine les 4 types : Resistor, Capacitor, IC, Inductor
# K=4 → au maximum 3 axes discriminants

Xdf2 = df[var_quant + ['component_type']].dropna().reset_index(drop=True)

lda2  = LinearDiscriminantAnalysis()
Xlda2 = lda2.fit_transform(Xdf2[var_quant], Xdf2['component_type'])

labels2 = Xdf2['component_type'].cat.categories.tolist()
colors2 = ['red', 'blue', 'green', 'orange']

plt.figure(figsize=(8, 6))
for g, label, color in zip(labels2, labels2, colors2):
    idx = Xdf2['component_type'] == g
    plt.scatter(Xlda2[idx, 0], Xlda2[idx, 1], label=label, alpha=0.3, s=5, color=color)

# Centres de gravité
centres      = Xdf2.groupby('component_type')[var_quant].mean()
centres_lda2 = lda2.transform(centres)
plt.scatter(centres_lda2[:, 0], centres_lda2[:, 1],
            color='black', marker='X', s=150, label='Centres', zorder=5)

plt.xlabel("Composante discriminante 1")
plt.ylabel("Composante discriminante 2")
plt.title("ADL - Séparation des types de composants")
plt.legend()
plt.show()
# CRITIQUE : Les types de composants devraient être très bien séparés car leurs
# plages de valeurs physiques (résistance, capacitance...) sont par nature différentes.
# Une séparation parfaite confirmerait que les variables mesurent bien ce qu'elles
# sont censées mesurer. Si les clusters se chevauchent → problème de mesure ou
# de nommage dans le dataset.

# Cercle des corrélations
Xlda2_df     = pd.DataFrame(Xlda2[:, :2], columns=['CD1', 'CD2'])
correlations = pd.DataFrame(index=var_quant)
for col in var_quant:
    correlations.loc[col, 'COR_1'] = np.corrcoef(Xdf2[col], Xlda2_df['CD1'])[0, 1]
    correlations.loc[col, 'COR_2'] = np.corrcoef(Xdf2[col], Xlda2_df['CD2'])[0, 1]

print("\n===== CORRELATIONS VARIABLES / AXES DISCRIMINANTS =====")
print(correlations.round(3))
# CRITIQUE : Les variables avec |COR_1| proche de 1 sont les plus importantes
# pour l'axe 1 (séparation principale). Si resistance_ohm et capacitance_uF
# dominent → normal, ce sont les grandeurs physiques définissant le type.

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='black'))
for col in var_quant:
    ax.plot(correlations.loc[col, 'COR_1'], correlations.loc[col, 'COR_2'], 'o')
    ax.text(correlations.loc[col, 'COR_1'], correlations.loc[col, 'COR_2'], col, fontsize=7)
ax.axhline(0, color='grey')
ax.axvline(0, color='grey')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
plt.title("Cercle des corrélations - ADL type de composant")
plt.show()

print("\n===== SCORE ADL types de composant =====")
print(f"Taux de bonne classification : {lda2.score(Xdf2[var_quant], Xdf2['component_type'])*100:.1f}%")