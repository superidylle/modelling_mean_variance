# -*- coding: utf-8  -*-
# @Author: Xingqi Ye
# @Time: 2019-05-14-19


import pandas as pd
import numpy as np
import config
from functions import *
from math import *
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)


pd.set_option('expand_frame_repr', False)

#Question 1
daily_data = pd.read_csv(config.input_data_path + '/HW3-data.csv')

daily_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

daily_data['Date'] = pd.to_datetime(daily_data['Date'])

daily_data.set_index('Date', inplace=True)

monthly_data = daily_data.resample(rule='M', base=0, label='right', closed='right').last()


#Question 2
Average_Return_trans = []

Return = pd.DataFrame()

for code in monthly_data.columns:


    monthly_data[code + 'return'] = monthly_data[code].pct_change(1)

    Average_Return_trans.append(monthly_data[code + 'return'].mean())

    Return = Return.append(monthly_data[code + 'return'].dropna())

Average_Return_trans = np.matrix(Average_Return_trans)

Return_matrix = Return.values

Covariance_matrix = np.cov(Return_matrix)

Covariance_matrix_inverse = np.linalg.inv(Covariance_matrix)

Average_Return = np.transpose(Average_Return_trans)

Term_I = np.ones([7, 1])

Term_I_inverse = np.ones([1, 7])

Term_A = Average_Return_trans @ Covariance_matrix_inverse @ Average_Return

Term_B = Average_Return_trans @ Covariance_matrix_inverse @ Term_I

Term_C = Term_I_inverse @ Covariance_matrix_inverse @ Term_I


# Question 3
result_1 = np.zeros((2, 20))

for i in range(1, 21, 1):

    mu_p = 0.005 * i / 12

    lambda_p = (mu_p * Term_C - Term_B) / (Term_A * Term_C - Term_B ** 2)

    gamma_p  = (Term_A - Term_B * mu_p) / (Term_A * Term_C - Term_B ** 2)

    alpha_weight  = Covariance_matrix_inverse @ Average_Return * lambda_p + Covariance_matrix_inverse @ Term_I * gamma_p

    alpha_weight_trans = np.transpose(alpha_weight)

    variance_p  = alpha_weight_trans @ Covariance_matrix @ alpha_weight

    volatility_p = sqrt(variance_p)

    result_1[0, i-1] = volatility_p *sqrt(12)
    result_1[1, i-1] = mu_p * 12

    # print("Portfolio weights are:")
    # print(alpha_weight_trans)
    #
    # print("Portfolio mean equals to:")
    # print(mu_p)
    #
    # print("Portfolio volatility equals to ")
    # print(volatility_p)
    #
    # print("====================================")


fig, ax = plt.subplots()
ax.plot(result_1[0, :], result_1[1, :], 'o', color ='red')
ax.plot(result_1[0, :], result_1[1, :], '-', color ='red')


result_2 = np.zeros((2, 20))

for i in range(1, 21, 1):

    mu_p = 0.005 * i / 12

    rf = 0.02 / 12

    delta = (mu_p - rf) / (np.transpose(Average_Return - Term_I * rf) @ Covariance_matrix_inverse @ (Average_Return - Term_I * rf))

    omega_p = Covariance_matrix_inverse @ (Average_Return - Term_I * rf) * delta

    variance_p = np.transpose(omega_p) @ Covariance_matrix @ omega_p

    volatility_p = sqrt(variance_p)

    result_2[0, i-1] = volatility_p * sqrt(12)
    result_2[1, i-1] = mu_p * 12


ax.plot(result_2[0, :], result_2[1, :], 'o', color='blue')
ax.plot(result_2[0, :], result_2[1, :], '-', color='blue')
plt.title('Efficient Frontier')
plt.xlabel('annualised volatility')
plt.ylabel('annualised returns')
plt.show()

# Question 5

# MDR
MDR_res = maximum_diversification(Return_matrix)

print(MDR_res)



# GMV
GMV_res = global_minimum_variance(Return_matrix)

print(GMV_res)


# MSR
MSR_res = maximum_sharpe_ratio(Return_matrix)
print(MSR_res)

