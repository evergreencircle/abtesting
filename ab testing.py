#%% md
# ### Confidence interal
# #### https://t.me/c/2360473936/100
#%%

import scipy.stats as stats
import numpy as np

control_conversions = 500
control_total = 10000
test_conversions = 555
test_total = 10000 

control_rate = control_conversions / control_total
test_rate = test_conversions / test_total

effect = test_rate - control_rate

se = np.sqrt((control_rate * (1 - control_rate)) / control_total +
             (test_rate * (1 - test_rate)) / test_total)

z_score = effect / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

ci_low, ci_high = effect - 1.96 * se, effect + 1.96 * se

print('Разница средних, %')
print(np.round(effect, 3) * 100)

print('Доверительный интервал, %')
print([np.round(ci_low, 3) * 100, np.round(ci_high, 3) * 100])

print('p-value')
print(np.round(p_value, 2))

# Разница средних, %
# 0.5
# Доверительный интервал, %
# [-0.1, 1.2]
# p-value
# 0.08

#%% md
# ### Confince interval for business
#%%

arppu = 1000
low_effect_arppu = ci_low * control_total *  arppu 
high_effect_arppu = ci_high * control_total *  arppu 
print([low_effect_arppu, high_effect_arppu])

# [-6955.767415148702, 116955.76741514866]

#%%
