# -*- coding: utf-8; -*-
"""GPmerit.py: Gaussian Process surrogate model of NMEP output from M. Ratcliff
2018 study on HOV and S in knock-limited engines.

--------------------------------------------------------------------------------
Developed by the NREL Computational Science Center
and LBNL Center for Computational Science and Engineering
Contact: Ryan King <ryan.king@nrel.gov>

Authors: Ryan King, Ray Grout and Juliane Mueller
--------------------------------------------------------------------------------


This file is part of the Co-optimizer, developed as part of the Co-Optimization
of Fuels & Engines (Co-Optima) project sponsored by the U.S. Department of
Energy (DOE) Office of Energy Efficiency and Renewable Energy (EERE), Bioenergy
Technologies and Vehicle Technologies Offices. (Optional): Co-Optima is a
collaborative project of multiple national laboratories initiated to
simultaneously accelerate the introduction of affordable, scalable, and
sustainable biofuels and high-efficiency, low-emission vehicle engines.

"""
import numpy as np
import scipy as scp
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn import preprocessing
import matplotlib.pyplot as plt


def run_GP():
    """loads Ratcliff data, returns fully trained GP."""

    dfLoad = pd.read_excel('Figures 1 & 2 for SAE PFL 2018 Paper_RK.xlsx',
                           sheetname="Load Sweep Data Reformatted")
    dfIAT = pd.read_excel('Figures 1 & 2 for SAE PFL 2018 Paper_RK.xlsx',
                          sheetname="IAT Sweep Data Reformatted")
    dfAll = pd.concat([dfLoad, dfIAT], ignore_index=True)

    dfAll = dfAll.dropna()
    dfAll = dfAll[dfAll.Injection != 'UI']
    inputs = dfAll.columns[[3, 4, 5, 9, 10, 11]]
    output = dfAll.columns[1]
    dfOutputs = dfAll[output]
    dfInputs = dfAll[inputs]

    X = dfInputs.values
    # print(X.shape)
    # print(np.amin(X, axis = 0), np.amax(X, axis = 0))
    y = dfOutputs.values
    scaler = preprocessing.StandardScaler().fit(X)
    return train_GP(X, y, scaler)


def run_GP_fuels():
    """Removes one fuel at a time from Ratcliff data, trains GP, and tries
    to predict performance of that fuel."""

    dfLoad = pd.read_excel('Figures 1 & 2 for SAE PFL 2018 Paper_RK.xlsx',
                           sheetname="Load Sweep Data Reformatted")
    dfIAT = pd.read_excel('Figures 1 & 2 for SAE PFL 2018 Paper_RK.xlsx',
                          sheetname="IAT Sweep Data Reformatted")
    dfAll = pd.concat([dfLoad, dfIAT], ignore_index=True)

    dfAll = dfAll.dropna()
    dfAll = dfAll[dfAll.Injection != 'UI']
    inputs = dfAll.columns[[3, 4, 5, 9, 10, 11]]  # CA50, IAT, KI, RON, S,HOV
    output = dfAll.columns[1]
    dfOutputs = dfAll[output]
    dfInputs = dfAll[inputs]

    X = dfInputs.values
    y = dfOutputs.values
    scaler = preprocessing.StandardScaler().fit(X)
    font = {'size': 20}
    for fuel in dfAll.Fuel.unique():
        y_mean, y_std, y_true = predict_fuel(dfAll, inputs, output, fuel=fuel)
        # plt.figure()
        fig, ax = plt.subplots()
        plt.plot(range(len(y_mean)), y_mean, 'k')
        plt.fill_between(range(len(y_mean)), y_mean - y_std*2,
                         y_mean + y_std*2,
                         alpha=0.5, color='k')
        plt.errorbar(range(len(y_mean)), y_true, yerr=40, fmt='o', c='b')
        plt.legend(['GP prediction',
                    'GP +/- $2\sigma$',
                    'Actual experiments'], loc='lower right')

        plt.title('Blind NMEP Prediction for ' + fuel)
        plt.xlabel('Experiment #')
        plt.ylabel('NMEP [kPa]')
        plt.rc('font', **font)
        plt.tight_layout()
        # plt.show()
        plt.savefig("gpstuff_"+fuel+".pdf")
        plt.close("all")


def train_GP(X, y, scaler):
    """Returns a trained Gaussian Process given training data and scaler."""

    stdev = 20
    kernel = (1.0 * Matern(length_scale=5*np.ones(X.shape[1]),
                           length_scale_bounds=(1e-1, 1e1), nu=2.5)
              + WhiteKernel(noise_level=stdev,
                            noise_level_bounds=(1e-1, 2e1)))

    X_ = scaler.transform(X)

    GP = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10,
                                  normalize_y=True).fit(X_, y)

    return GP, scaler


def predict_GP(GP, scaler, X):
    """Makes GP prediction on test set"""
    Xpred = scaler.transform(X)
    y_mean, y_std = GP.predict(Xpred, return_std=True)
    return y_mean, y_std


def predict_fuel(dfAll, inputs, output, fuel='E40-TRF71'):
    """Creates train and test sets to predict performance of a known fuel."""
    dfTrain = dfAll[dfAll.Fuel != fuel]
    dfTest = dfAll[dfAll.Fuel == fuel]

    dfOutputs = dfTrain[output]
    dfInputs = dfTrain[inputs]

    X = dfInputs.values
    scaler = preprocessing.StandardScaler().fit(X)
    y = dfOutputs.values

    GP, scaler = train_GP(X, y, scaler)

    Xpred = dfTest[inputs].values

    y_mean, y_std = predict_GP(GP, scaler, Xpred)

    return y_mean, y_std, dfTest[output].values


if __name__ == "__main__":
    # A, B  = run_GP()
    run_GP_fuels()
    lll
    print(predict_GP(A, B, np.array([[10, 40, 8., 100., 5., 400]])))

    lower = np.array([ 6.7,  35.,    2.,  99.2,    0., 303.])
    upper = np.array([23.8,  90.,  10.5, 105.6,  12.2, 595.])
