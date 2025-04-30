import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

raw_df = pd.read_csv('patients_dec.csv', keep_default_na=False)

# Compute CPTs from raw data --------------------------------------------
# Smoking CPD
sm_counts = raw_df['Smoking'].value_counts(normalize=True).reindex(['No','Yes'])
cpd_smoking = TabularCPD('Smoking', 2,
                         [[sm_counts['No']], [sm_counts['Yes']]],
                         state_names={'Smoking':['No','Yes']})

# AgeGroup CPD
ag_counts = raw_df['AgeGroup'].value_counts(normalize=True).reindex(['Child','Adult','Senior'])
cpd_age = TabularCPD('AgeGroup', 3,
                     [[ag_counts['Child']], [ag_counts['Adult']], [ag_counts['Senior']]],
                     state_names={'AgeGroup':['Child','Adult','Senior']})

# Immunity CPD
im_counts = raw_df['Immunity'].value_counts(normalize=True).reindex(['Good','Weak'])
cpd_immunity = TabularCPD('Immunity', 2,
                          [[im_counts['Good']], [im_counts['Weak']]],
                          state_names={'Immunity':['Good','Weak']})

# Pneumonia CPD
p_counts = raw_df.groupby(['Smoking','AgeGroup','Immunity','Pneumonia']).size().unstack(fill_value=0)
p_probs  = p_counts.div(p_counts.sum(axis=1), axis=0).reindex(columns=['No','Yes'])
values_p = [p_probs['No'].values, p_probs['Yes'].values]
cpd_pneumonia = TabularCPD('Pneumonia', 2, values_p,
                           evidence=['Smoking','AgeGroup','Immunity'],
                           evidence_card=[2,3,2],
                           state_names={
                               'Pneumonia':['No','Yes'],
                               'Smoking':['No','Yes'],
                               'AgeGroup':['Child','Adult','Senior'],
                               'Immunity':['Good','Weak']
                           })

# Fever CPD
f_counts = raw_df.groupby(['Pneumonia','Fever']).size().unstack(fill_value=0)
f_probs  = f_counts.div(f_counts.sum(axis=1), axis=0)
values_f = [f_probs['High'].values, f_probs['Low'].values, f_probs['None'].values]
cpd_fever = TabularCPD('Fever', 3, values_f,
                       evidence=['Pneumonia'], evidence_card=[2],
                       state_names={'Fever':['High','Low','None'],
                                    'Pneumonia':['No','Yes']})

# Cough CPD
c_counts = raw_df.groupby(['Pneumonia','Cough']).size().unstack(fill_value=0)
c_probs  = c_counts.div(c_counts.sum(axis=1), axis=0)
values_c = [c_probs['Severe'].values, c_probs['Mild'].values, c_probs['None'].values]
cpd_cough = TabularCPD('Cough', 3, values_c,
                       evidence=['Pneumonia'], evidence_card=[2],
                       state_names={'Cough':['Severe','Mild','None'],
                                    'Pneumonia':['No','Yes']})

# XRay CPD
x_counts = raw_df.groupby(['Fever','Cough','XRay']).size().unstack(fill_value=0)
x_probs  = x_counts.div(x_counts.sum(axis=1), axis=0)
values_x = [x_probs['Abnormal'].values, x_probs['Normal'].values]
cpd_xray = TabularCPD('XRay', 2, values_x,
                      evidence=['Fever','Cough'], evidence_card=[3,3],
                      state_names={'XRay':['Abnormal','Normal'],
                                   'Fever':['High','Low','None'],
                                   'Cough':['Severe','Mild','None']})

# Build model & add CPDs -----------------------------------------------
model = BayesianNetwork([
    ('Smoking','Pneumonia'), ('AgeGroup','Pneumonia'), ('Immunity','Pneumonia'),
    ('Pneumonia','Fever'), ('Pneumonia','Cough'), ('Fever','XRay'), ('Cough','XRay')
])
model.add_cpds(cpd_smoking, cpd_age, cpd_immunity,
               cpd_pneumonia, cpd_fever, cpd_cough, cpd_xray)
model.check_model()

# Inference -------------------------------------------------------------
infer = VariableElimination(model)
posterior = infer.query(['Pneumonia'],
                        evidence={'Smoking':'Yes','AgeGroup':'Senior',
                                  'Immunity':'Weak','Fever':'High',
                                  'Cough':'Mild','XRay':'Abnormal'})
labels = posterior.state_names['Pneumonia']
probs  = posterior.values

# Print detailed results ------------------------------------------------
print("=== Pneumonia Risk Assessment ===")
for lbl, pr in zip(labels, probs):
    print(f"P(Pneumonia={lbl} | evidence) = {pr:.3f}")

# Plot BN structure -----------------------------------------------------
plt.figure(figsize=(8,6))
pos = nx.spring_layout(model)
nx.draw_networkx_nodes(model, pos, node_color='skyblue', node_size=2000)
nx.draw_networkx_labels(model, pos, font_size=10)
nx.draw_networkx_edges(model, pos, arrows=False)
plt.title("Bayesian Network for Pneumonia Diagnosis")
plt.axis('off')
plt.show()

# Plot posterior as bar chart -------------------------------------------
plt.figure(figsize=(6,4))
bars = plt.bar(labels, probs, color=['lightgreen','salmon'])
plt.ylabel("Probability")
plt.title("Posterior Probability of Pneumonia")
plt.ylim(0,1)
for bar, pr in zip(bars, probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{pr:.1%}", ha='center')
plt.show()

# Friendly explanation --------------------------------------------------
print(f"\nKey takeaway: with these symptoms, there's a {probs[labels.index('Yes')]:.1%} chance of pneumonia.")