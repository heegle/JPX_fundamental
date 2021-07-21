#!/usr/bin/env python
# coding: utf-8

# In[9]:


# -*- coding: utf-8 -*-
import io
from tqdm.auto import tqdm

from tensorflow.keras.models import load_model

import os
import pickle
import sys
import warnings
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay

import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm.auto import tqdm

import datetime as dt
from datetime import timedelta

import tensorflow as tf


class JpCalendar(AbstractHolidayCalendar):
    rules = [
    Holiday('1231', month=12, day=31),
    Holiday('0101', month=1, day=1),
    Holiday('0102', month=1, day=2),
    Holiday('0103', month=1, day=3),        
    Holiday('成人の日', year =2016, month=1, day=11),
    Holiday('建国記念の日', year =2016, month=2, day=11),
    Holiday('春分の日', year =2016, month=3, day=20),
    Holiday('休日', year =2016, month=3, day=21),
    Holiday('昭和の日', year =2016, month=4, day=29),
    Holiday('憲法記念日', year =2016, month=5, day=3),
    Holiday('みどりの日', year =2016, month=5, day=4),
    Holiday('こどもの日', year =2016, month=5, day=5),
    Holiday('海の日', year =2016, month=7, day=18),
    Holiday('山の日', year =2016, month=8, day=11),
    Holiday('敬老の日', year =2016, month=9, day=19),
    Holiday('秋分の日', year =2016, month=9, day=22),
    Holiday('体育の日', year =2016, month=10, day=10),
    Holiday('文化の日', year =2016, month=11, day=3),
    Holiday('勤労感謝の日', year =2016, month=11, day=23),
    Holiday('天皇誕生日', year =2016, month=12, day=23),
    Holiday('休日', year =2017, month=1, day=2),
    Holiday('成人の日', year =2017, month=1, day=9),
    Holiday('建国記念の日', year =2017, month=2, day=11),
    Holiday('春分の日', year =2017, month=3, day=20),
    Holiday('昭和の日', year =2017, month=4, day=29),
    Holiday('憲法記念日', year =2017, month=5, day=3),
    Holiday('みどりの日', year =2017, month=5, day=4),
    Holiday('こどもの日', year =2017, month=5, day=5),
    Holiday('海の日', year =2017, month=7, day=17),
    Holiday('山の日', year =2017, month=8, day=11),
    Holiday('敬老の日', year =2017, month=9, day=18),
    Holiday('秋分の日', year =2017, month=9, day=23),
    Holiday('体育の日', year =2017, month=10, day=9),
    Holiday('文化の日', year =2017, month=11, day=3),
    Holiday('勤労感謝の日', year =2017, month=11, day=23),
    Holiday('天皇誕生日', year =2017, month=12, day=23),
    Holiday('成人の日', year =2018, month=1, day=8),
    Holiday('建国記念の日', year =2018, month=2, day=11),
    Holiday('休日', year =2018, month=2, day=12),
    Holiday('春分の日', year =2018, month=3, day=21),
    Holiday('昭和の日', year =2018, month=4, day=29),
    Holiday('休日', year =2018, month=4, day=30),
    Holiday('憲法記念日', year =2018, month=5, day=3),
    Holiday('みどりの日', year =2018, month=5, day=4),
    Holiday('こどもの日', year =2018, month=5, day=5),
    Holiday('海の日', year =2018, month=7, day=16),
    Holiday('山の日', year =2018, month=8, day=11),
    Holiday('敬老の日', year =2018, month=9, day=17),
    Holiday('秋分の日', year =2018, month=9, day=23),
    Holiday('休日', year =2018, month=9, day=24),
    Holiday('体育の日', year =2018, month=10, day=8),
    Holiday('文化の日', year =2018, month=11, day=3),
    Holiday('勤労感謝の日', year =2018, month=11, day=23),
    Holiday('天皇誕生日', year =2018, month=12, day=23),
    Holiday('休日', year =2018, month=12, day=24),
    Holiday('成人の日', year =2019, month=1, day=14),
    Holiday('建国記念の日', year =2019, month=2, day=11),
    Holiday('春分の日', year =2019, month=3, day=21),
    Holiday('昭和の日', year =2019, month=4, day=29),
    Holiday('休日', year =2019, month=4, day=30),
    Holiday('休日（祝日扱い）', year =2019, month=5, day=1),
    Holiday('休日', year =2019, month=5, day=2),
    Holiday('憲法記念日', year =2019, month=5, day=3),
    Holiday('みどりの日', year =2019, month=5, day=4),
    Holiday('こどもの日', year =2019, month=5, day=5),
    Holiday('休日', year =2019, month=5, day=6),
    Holiday('海の日', year =2019, month=7, day=15),
    Holiday('山の日', year =2019, month=8, day=11),
    Holiday('休日', year =2019, month=8, day=12),
    Holiday('敬老の日', year =2019, month=9, day=16),
    Holiday('秋分の日', year =2019, month=9, day=23),
    Holiday('体育の日（スポーツの日）', year =2019, month=10, day=14),
    Holiday('休日（祝日扱い）', year =2019, month=10, day=22),
    Holiday('文化の日', year =2019, month=11, day=3),
    Holiday('休日', year =2019, month=11, day=4),
    Holiday('勤労感謝の日', year =2019, month=11, day=23),
    Holiday('成人の日', year =2020, month=1, day=13),
    Holiday('建国記念の日', year =2020, month=2, day=11),
    Holiday('天皇誕生日', year =2020, month=2, day=23),
    Holiday('休日', year =2020, month=2, day=24),
    Holiday('春分の日', year =2020, month=3, day=20),
    Holiday('昭和の日', year =2020, month=4, day=29),
    Holiday('憲法記念日', year =2020, month=5, day=3),
    Holiday('みどりの日', year =2020, month=5, day=4),
    Holiday('こどもの日', year =2020, month=5, day=5),
    Holiday('休日', year =2020, month=5, day=6),
    Holiday('海の日', year =2020, month=7, day=23),
    Holiday('スポーツの日', year =2020, month=7, day=24),
    Holiday('山の日', year =2020, month=8, day=10),
    Holiday('敬老の日', year =2020, month=9, day=21),
    Holiday('秋分の日', year =2020, month=9, day=22),
    Holiday('文化の日', year =2020, month=11, day=3),
    Holiday('勤労感謝の日', year =2020, month=11, day=23),
    Holiday('成人の日', year =2021, month=1, day=11),
    Holiday('建国記念の日', year =2021, month=2, day=11),
    Holiday('天皇誕生日', year =2021, month=2, day=23),
    Holiday('春分の日', year =2021, month=3, day=20),
    Holiday('昭和の日', year =2021, month=4, day=29),
    Holiday('憲法記念日', year =2021, month=5, day=3),
    Holiday('みどりの日', year =2021, month=5, day=4),
    Holiday('こどもの日', year =2021, month=5, day=5),
    Holiday('海の日', year =2021, month=7, day=22),
    Holiday('スポーツの日', year =2021, month=7, day=23),
    Holiday('山の日', year =2021, month=8, day=8),
    Holiday('休日', year =2021, month=8, day=9),
    Holiday('敬老の日', year =2021, month=9, day=20),
    Holiday('秋分の日', year =2021, month=9, day=23),
    Holiday('文化の日', year =2021, month=11, day=3),
    Holiday('勤労感謝の日', year =2021, month=11, day=23),
   ]


class ScoringService(object):
    dates_tr = "2017-01-01"
    datee_tr = "2019-12-31"
    TEST_START = "2020-01-01"
    TARGET_LABELS = ["label_high_20", "label_low_20"]
    dfs = None
    mil = 1000000
    
    tse = JpCalendar()
    
    @classmethod
    def div_aad(cls, rtype, fig):
        return np.where(rtype == "Q2", fig * 1.5, np.where((rtype == "Q1") | (rtype == "Q3"), fig*2.5, fig))

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            #if k == "stock_price":
            #    cls.dfs[k].loc[:, "date"] = pd.to_datetime(
            #        cls.dfs[k].loc[:, "EndOfDayQuote Date"]
            #    )
                #cls.dfs[k].set_index("datetime", inplace=True)
            #elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
            #    cls.dfs[k].loc[:, "date"] = pd.to_datetime(
            #        cls.dfs[k].loc[:, "base_date"]
            #    )
                #cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_model(cls, model_path='../model'):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        
        try:
            with open(os.path.join(model_path, 'model_jpx_high.pkl'), 'rb') as f:
                cls.model_h = pickle.load(f)

            with open(os.path.join(model_path, 'model_jpx_low.pkl'), 'rb') as g:
                cls.model_l = pickle.load(g)

    #        cls.model = load_model(os.path.join(model_path, 'model_jpx.h5'))

            return True
        
        except Exception as e:
            print(e)
            return False

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):

#        cls.get_inputs("..")
        
        mil = cls.mil
        tse = cls.tse
        dates_tr = cls.dates_tr
        datee_tr = cls.datee_tr
        
        cls.get_dataset(inputs)
        
        sl = cls.dfs['stock_list']
        sp = cls.dfs['stock_price']
        sf = cls.dfs['stock_fin']
        #slb = cls.dfs['stock_labels']
        
        m_conf = pd.read_csv("./m_forecast_confidence.csv")
        m_qs = pd.read_csv("./m_qsales.csv")
        m_soyaku = pd.read_csv("./m_soyaku.csv")
        m_yutai = pd.read_csv("./m_yutai.csv")
        
        #####################################
        sl = sl.rename(columns={
                        "prediction_target": "target", 
                        "Local Code": "code",
                        "Name (English)": "name",
                        "Section/Products": "section",
                        "Size Code (New Index Series)": "sizecode",
                        "17 Sector(Code)": "17sec",
                        "33 Sector(Code)": "33sec",
                        "IssuedShareEquityQuote IssuedShare": "issued",
                       })

        sp = sp.rename(columns={
                        "Local Code": "code",
                        "EndOfDayQuote Date": "date",
                        "EndOfDayQuote Open" : "open",
                        "EndOfDayQuote High" : "high",
                        "EndOfDayQuote Low" : "low",
                        "EndOfDayQuote Close" : "close",
                        "EndOfDayQuote ExchangeOfficialClose": "eclose",
                        "EndOfDayQuote Volume": "vol",
                        "EndOfDayQuote CumulativeAdjustmentFactor": "adfac",
                       })

        sf = sf.rename(columns={
                        "base_date": "date",
                        "Local Code": "code",
                        "Result_FinancialStatement AccountingStandard": "accstd",
                        "Result_FinancialStatement FiscalPeriodEnd": "period",
                        "Result_FinancialStatement ReportType": "rtype",
                        "Result_FinancialStatement FiscalYear": "fyear",
                        "Result_FinancialStatement ModifyDate": "moddate",
                        "Result_FinancialStatement CompanyType": "ctype",
                        "Result_FinancialStatement ChangeOfFiscalYearEnd": "FYch",
                        "Result_FinancialStatement NetSales": "sales",
                        "Result_FinancialStatement OperatingIncome": "opein",
                        "Result_FinancialStatement OrdinaryIncome": "ordin",
                        "Result_FinancialStatement NetIncome": "netin",
                        "Result_FinancialStatement TotalAssets": "tasset",
                        "Result_FinancialStatement NetAssets": "nasset",
                        "Result_FinancialStatement CashFlowsFromOperatingActivities": "opecf",
                        "Result_FinancialStatement CashFlowsFromFinancingActivities": "fincf",
                        "Result_FinancialStatement CashFlowsFromInvestingActivities": "invcf",
                        "Forecast_FinancialStatement AccountingStandard": "accstd_f",
                        "Forecast_FinancialStatement FiscalPeriodEnd": "period_f",
                        "Forecast_FinancialStatement ReportType": "rtype_f",
                        "Forecast_FinancialStatement FiscalYear": "fyear_f",
                        "Forecast_FinancialStatement ModifyDate": "moddate_f",
                        "Forecast_FinancialStatement CompanyType": "ctype_f",
                        "Forecast_FinancialStatement ChangeOfFiscalYearEnd": "FYch_f",
                        "Forecast_FinancialStatement NetSales": "sales_f",
                        "Forecast_FinancialStatement OperatingIncome": "opein_f",
                        "Forecast_FinancialStatement OrdinaryIncome": "ordin_f",
                        "Forecast_FinancialStatement NetIncome": "netin_f",
                        "Result_Dividend FiscalPeriodEnd": "divperiod",
                        "Result_Dividend ReportType": "div_rtype",
                        "Result_Dividend ModifyDate": "divmoddate",
                        "Result_Dividend RecordDate": "divdt",
                        "Result_Dividend QuarterlyDividendPerShare": "qdiv",
                        "Result_Dividend AnnualDividendPerShare": "adiv",
                        "Forecast_Dividend FiscalPeriodEnd": "divperiod_f",
                        "Forecast_Dividend ReportType": "div_rtype_f",
                        "Forecast_Dividend ModifyDate": "divmoddate_f",
                        "Forecast_Dividend RecordDate": "divdt_f",
                        "Forecast_Dividend QuarterlyDividendPerShare": "qdiv_f",
                        "Forecast_Dividend AnnualDividendPerShare": "adiv_f",
                       })

        #slb = slb.rename(columns={
        #                "base_date": "date",
        #                "Local Code": "code",
        #               })

        #slb.loc[:, "date"] = pd.to_datetime(slb.loc[:, "date"])

        sp.loc[:, "date"] = pd.to_datetime(sp.loc[:, "date"])
        sf.loc[:, "date"] = pd.to_datetime(sf.loc[:, "date"])
        sf.loc[:, "divdt"] = pd.to_datetime(sf.loc[:, "divdt"])
        sf.loc[:, "divdt_f"] = pd.to_datetime(sf.loc[:, "divdt_f"])
        
        #東証システム障害の日除外
        sp = sp[sp["date"]!="2020-10-01"]

        sl = pd.merge(sl,m_soyaku, on=["code"], how="left")
        sl.loc[sl["soyaku"].isnull(), "soyaku"] = 0

        sf.loc[sf["divdt_f"].isnull(), "divdt_f"] = pd.to_datetime(sf["divperiod_f"].str[:4] + "-" + sf["divperiod_f"].str[5:7] + "-" + "01") + offsets.MonthEnd()

        sf = sf.sort_values(['code', 'date'])
        sp = sp.sort_values(['code', 'date'])

        
        #slb["md"] = slb[slb["date"]<"2020-01-01"]["date"].dt.month * 100 + slb[slb["date"]<"2020-01-01"]["date"].dt.day
        #slbmd = slb.groupby("md")[["label_high_20","label_low_20"]].mean().reset_index()

        #slbmd = slbmd.rename(columns={
        #                        "label_high_20": "md_high_20",
        #                        "label_low_20": "md_low_20",
        #                       })

        #slb = pd.merge(slb,slbmd, on=["md"], how="left")


        sf = pd.merge(sf,m_qs, on=["code"], how="left")
        
        sfan = sf[["code","period","rtype"]].drop_duplicates()
        sfan = sfan[sfan["rtype"]=="Annual"]

        sfan["period"] = np.where(sfan["period"].isnull(), "0000/00", sfan["period"])
        sfan["mondiff"] = np.nan
        sfan["mondiff"][1:] = np.where((sfan["period"][1:].isnull()) | (sfan.shift(1)["period"][1:].isnull()) | (sfan["code"][1:]!=sfan.shift(1)["code"][1:]), np.nan, 
                                 sfan["period"][1:].str[:4].astype(int)*12 + sfan["period"][1:].str[5:7].astype(int)
                                 - (sfan.shift(1)["period"][1:].str[:4].astype(int)*12 + sfan.shift(1)["period"][1:].str[5:7].astype(int)))
        sf["period"] = np.where(sf["period"]=="0000/00", np.nan, sf["period"])
        sf = pd.merge(sf,sfan, on=["code","period","rtype"], how="left")

        sf["qsfac"] = np.where(sf["FYch"] == 1, sf["mondiff"] / 12, 1)
        sf["qsfac"] = np.where(sf["rtype"]=="Q1", sf["qs1"], sf["qsfac"])
        sf["qsfac"] = np.where(sf["rtype"]=="Q2", sf["qs2"], sf["qsfac"])
        sf["qsfac"] = np.where(sf["rtype"]=="Q3", sf["qs3"], sf["qsfac"])


        sfan = sf[["code","period_f","rtype_f"]].drop_duplicates()
        sfan = sfan[sfan["rtype_f"]=="Annual"]

        sfan["period_f"] = np.where(sfan["period_f"].isnull(), "0000/00", sfan["period_f"])
        sfan["mondiff_f"] = np.nan
        sfan["mondiff_f"][1:] = np.where((sfan["period_f"][1:].isnull()) | (sfan.shift(1)["period_f"][1:].isnull()) | (sfan["code"][1:]!=sfan.shift(1)["code"][1:]), np.nan, 
                                 sfan["period_f"][1:].str[:4].astype(int)*12 + sfan["period_f"][1:].str[5:7].astype(int)
                                 - (sfan.shift(1)["period_f"][1:].str[:4].astype(int)*12 + sfan.shift(1)["period_f"][1:].str[5:7].astype(int)))
        sf["period_f"] = np.where(sf["period_f"]=="0000/00", np.nan, sf["period_f"])
        sf = pd.merge(sf,sfan, on=["code","period_f","rtype_f"], how="left")


        sf["qsfac_f"] = np.where(sf["FYch_f"] == 1, sf["mondiff_f"] / 12, 1)
        sf["qsfac_f"] = np.where(sf["rtype_f"]=="Q1", sf["qs1"], sf["qsfac_f"])
        sf["qsfac_f"] = np.where(sf["rtype_f"]=="Q2", sf["qs2"], sf["qsfac_f"])
        sf["qsfac_f"] = np.where(sf["rtype_f"]=="Q3", sf["qs3"], sf["qsfac_f"])

        sf["noresult"] = np.where(sf["sales"].isnull() & sf["opein"].isnull() & sf["ordin"].isnull() & sf["netin"].isnull(), 1, 0)
        sf["nofrct"] = np.where(sf["sales_f"].isnull() & sf["opein_f"].isnull() & sf["ordin_f"].isnull() & sf["netin_f"].isnull(), 1, 0)

        sf["nofrct"] = np.where((sf["code"] == sf.shift(1)["code"]) 
         & (sf["accstd_f"] == sf.shift(1)["accstd_f"]) & (sf["period_f"] == sf.shift(1)["period_f"])
         & (sf["sales_f"] == sf.shift(1)["sales_f"]) & (sf["opein_f"] == sf.shift(1)["opein_f"])
         & (sf["ordin_f"] == sf.shift(1)["ordin_f"]) & (sf["netin_f"] == sf.shift(1)["netin_f"]), 
                        1, sf["nofrct"])

        sf["sales_aad"] = sf["sales"] * 1 / sf["qsfac"]
        sf["sales_aad_f"] = sf["sales_f"] * 1 / sf["qsfac_f"]
        sf["opein_aad"] = sf["opein"] * 1 / sf["qsfac"]
        sf["opein_aad_f"] = sf["opein_f"] * 1 / sf["qsfac_f"]
        sf["ordin_aad"] = sf["ordin"] * 1 / sf["qsfac"]
        sf["ordin_aad_f"] = sf["ordin_f"] * 1 / sf["qsfac_f"]
        sf["netin_aad"] = sf["netin"] * 1 / sf["qsfac"]
        sf["netin_aad_f"] = sf["netin_f"] * 1 / sf["qsfac_f"]

        sf["sales_aad_t"] = np.where(sf["sales_aad_f"].isnull()==False, sf["sales_aad_f"], np.where(sf["nofrct"]==0, np.nan, sf["sales_aad"]))
        sf["opein_aad_t"] = np.where(sf["opein_aad_f"].isnull()==False, sf["opein_aad_f"], np.where(sf["nofrct"]==0, np.nan, sf["opein_aad"]))
        sf["ordin_aad_t"] = np.where(sf["ordin_aad_f"].isnull()==False, sf["ordin_aad_f"], np.where(sf["nofrct"]==0, np.nan, sf["ordin_aad"]))
        sf["netin_aad_t"] = np.where(sf["netin_aad_f"].isnull()==False, sf["netin_aad_f"], np.where(sf["nofrct"]==0, np.nan, sf["netin_aad"]))

        sf["accstd_t"] = np.where(sf["nofrct"]==0, sf["accstd_f"], sf["accstd"])
        sf["fyear_t"] = np.where(sf["nofrct"]==0, sf["fyear_f"], sf["fyear"])


        sf["opecf_aad"] = sf["opecf"] * 1 / sf["qsfac"]
        sf["fincf_aad"] = sf["fincf"] * 1 / sf["qsfac"]
        sf["invcf_aad"] = sf["invcf"] * 1 / sf["qsfac"]

        
        sf_divdt = sf[["code", "divdt"]].drop_duplicates().copy()
        sf_divdt_f = sf[["code", "divdt_f"]].drop_duplicates().copy()
        sf_divdt_f = sf_divdt_f.rename(columns={"divdt_f": "divdt"})
        sf_divdt = pd.concat([sf_divdt, sf_divdt_f]).drop_duplicates()
        sf_divdt = sf_divdt[sf_divdt["divdt"].isnull()==False]

        sp_adfac = sp[["code", "date","adfac"]].drop_duplicates().copy()
        sp_adfac["adfac"] = sp.groupby("code").shift(3)["adfac"]
        sp_adfac = sp_adfac.rename(columns={"date": "divdt", "adfac": "adfac_divdt"})

        sp_adfac = pd.concat([sp_adfac, sf_divdt])

        sp_adfac = sp_adfac.sort_values(['code', 'divdt'])
        sp_adfac["adfac_divdt"] = sp_adfac["adfac_divdt"].fillna(method='ffill')

        sp_adfac = sp_adfac.drop_duplicates()
        sf = pd.merge(sf, sp_adfac, on=['code', 'divdt'], how="left")

        sp_adfac = sp_adfac.rename(columns={"divdt": "divdt_f", "adfac_divdt": "adfac_divdt_f"})
        sf = pd.merge(sf, sp_adfac, on=['code', 'divdt_f'], how="left")

        sf["qdiv"] = sf["qdiv"] / sf["adfac_divdt"]
        sf["adiv"] = sf["adiv"] / sf["adfac_divdt"]

        sf["qdiv_f"] = sf["qdiv_f"] / sf["adfac_divdt_f"]
        sf["adiv_f"] = sf["adiv_f"] / sf["adfac_divdt_f"]

        sf["qdiv_aad"] =  np.where(sf["div_rtype"] == "Annual", sf["adiv"], cls.div_aad(sf["rtype"], sf["qdiv"]))
        sf["qdiv_aad_f"] =  np.where(sf["div_rtype_f"] == "Annual", sf["adiv_f"], cls.div_aad(sf["rtype_f"], sf["qdiv_f"]))
        sf["qdiv_aad_t"] = np.where(sf["qdiv_aad_f"].isnull(), sf["qdiv_aad"], sf["qdiv_aad_f"])

        
        sf["divkenridt"] = np.where(sf["divdt"] >= "2019-07-18", sf["divdt"] + offsets.CustomBusinessDay(-2, calendar=tse),sf["divdt"] + offsets.CustomBusinessDay(-3, calendar=tse))
        sf["divkenridt_f"] = np.where(sf["divdt_f"] >= "2019-07-18", sf["divdt_f"] + offsets.CustomBusinessDay(-2, calendar=tse),sf["divdt_f"] + offsets.CustomBusinessDay(-3, calendar=tse))

        sf["kenri_f"] = np.where(
                     (sf["divkenridt_f"] <= sf["date"] + offsets.CustomBusinessDay(30, calendar=tse))
                        &(sf["divkenridt_f"] > sf["date"]), 1, 0)

        sf["kenriochi_f"] = np.where(
                     (sf["divkenridt_f"] + offsets.CustomBusinessDay(1, calendar=tse) <= sf["date"] + offsets.CustomBusinessDay(20, calendar=tse))
                        & (sf["divkenridt_f"] + offsets.CustomBusinessDay(1, calendar=tse) > sf["date"] ), 1, 0)

        sf["post_kenriochi_f"] = np.where(
                    ( (sf["divkenridt_f"] + offsets.CustomBusinessDay(1, calendar=tse) <= sf["date"])
                        & (sf["divkenridt_f"] + offsets.CustomBusinessDay(20, calendar=tse) > sf["date"]))
                    | ((sf["divkenridt"] + offsets.CustomBusinessDay(1, calendar=tse) <= sf["date"])
                        & (sf["divkenridt"] + offsets.CustomBusinessDay(20, calendar=tse) > sf["date"])), 1, 0)

        
        sp = sp.sort_values(['code', 'date'])

        sp["eclose"] = sp["eclose"].replace(0, np.nan).interpolate()

        sp["eclose_ch_1"] = sp.groupby("code")["eclose"].pct_change(1)
        sp["eclose_ch_3"] = sp.groupby("code")["eclose"].pct_change(3)
        sp["eclose_ch_5"] = sp.groupby("code")["eclose"].pct_change(5)
        sp["eclose_ch_10"] = sp.groupby("code")["eclose"].pct_change(10)
        sp["eclose_ch_20"] = sp.groupby("code")["eclose"].pct_change(20)
        sp["eclose_ch_40"] = sp.groupby("code")["eclose"].pct_change(40)
        sp["eclose_ch_60"] = sp.groupby("code")["eclose"].pct_change(60)
        sp["eclose_ch_120"] = sp.groupby("code")["eclose"].pct_change(120)

        sp["eclose_ch_mid"] = sp.groupby("code")["eclose"].pct_change(240)

        sp["vol_ch"] = np.log(1.01 + np.where((sp.groupby("code").shift(120)["vol"]==0) | (sp.groupby("code").shift(120)["vol"].isnull()), sp["vol"]/100, sp.groupby("code")["vol"].pct_change(120)))

        sp = sp[sp["eclose"].isnull()==False]

        sp["vola"] = (sp["high"] - sp["low"])/ sp["eclose"]

        
        sp_code_max = sp[(sp["date"]>=dates_tr)&(sp["date"]<=datee_tr)][["code","eclose"]].groupby("code").max().reset_index()
        sp_code_min = sp[(sp["date"]>=dates_tr)&(sp["date"]<=datee_tr)][["code","eclose"]].groupby("code").min().reset_index()
        sp_code_median = sp[(sp["date"]>=dates_tr)&(sp["date"]<=datee_tr)][["code","eclose"]].groupby("code").median().reset_index()
        sp_code_max = sp_code_max.rename(columns={"eclose": "eclose_max"})
        sp_code_min = sp_code_min.rename(columns={"eclose": "eclose_min"})
        sp_code_median = sp_code_median.rename(columns={"eclose": "eclose_median"})
        sp_code = pd.merge(sp_code_max, sp_code_min, on=["code"], how="left")
        sp_code = pd.merge(sp_code, sp_code_median, on=["code"], how="left")
        sp_code["vola_code"] = (sp_code["eclose_max"] - sp_code["eclose_min"])/sp_code["eclose_median"] 
        sp = pd.merge(sp, sp_code, on=["code"], how="left")
        
        
        
        sp_macro = sp[["date","eclose_ch_1","eclose_ch_3","eclose_ch_10"]].groupby("date").mean()
        sp_macro = sp_macro.rename(columns={"eclose_ch_1": "eclose_ch_1_macro", "eclose_ch_3": "eclose_ch_3_macro", "eclose_ch_10": "eclose_ch_10_macro"})
        sp = pd.merge(sp, sp_macro, on=["date"], how="left")
        
        
        sp["sellmax_code"] = 0
        sp.loc[(sp["eclose_ch_10"]>-0.03)&(sp["eclose_ch_1"]<-0.03)&(sp.groupby("code").shift(1)["eclose_ch_1"]<-0.03),"sellmax_code"] = 1
        sp["sellmax_macro"] = 0
        sp.loc[(sp["eclose_ch_10_macro"]>-0.08)&(sp["eclose_ch_1_macro"]<-0.025)&(sp["eclose_ch_3_macro"]<-0.03),"sellmax_macro"] = 1

        
        sp["eclose_ori"] = sp["eclose"] * sp["adfac"]

        sp["eclose_tbc_pre_1"] = sp.groupby("code").shift(1)["eclose_ori"]/sp.groupby("code").shift(1)["adfac"]*sp["adfac"]
        sp["eclose_diff_1"] = np.round(sp["eclose_ori"] - sp["eclose_tbc_pre_1"])

        sp["stop"] = 0
        sp.loc[(sp["eclose_tbc_pre_1"]<100)&(sp["eclose_diff_1"]>=30), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<200)&(sp["eclose_diff_1"]>=50), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<500)&(sp["eclose_diff_1"]>=80), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<700)&(sp["eclose_diff_1"]>=100), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<1000)&(sp["eclose_diff_1"]>=150), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<1500)&(sp["eclose_diff_1"]>=300), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<2000)&(sp["eclose_diff_1"]>=400), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<3000)&(sp["eclose_diff_1"]>=500), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<5000)&(sp["eclose_diff_1"]>=700), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<7000)&(sp["eclose_diff_1"]>=1000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<10000)&(sp["eclose_diff_1"]>=1500), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<15000)&(sp["eclose_diff_1"]>=3000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<20000)&(sp["eclose_diff_1"]>=4000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<30000)&(sp["eclose_diff_1"]>=5000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<50000)&(sp["eclose_diff_1"]>=7000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<70000)&(sp["eclose_diff_1"]>=10000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<100000)&(sp["eclose_diff_1"]>=15000), "stop"] = 1
        sp.loc[(sp["eclose_tbc_pre_1"]<100)&(sp["eclose_diff_1"]<=-30), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<200)&(sp["eclose_diff_1"]<=-50), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<500)&(sp["eclose_diff_1"]<=-80), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<700)&(sp["eclose_diff_1"]<=-100), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<1000)&(sp["eclose_diff_1"]<=-150), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<1500)&(sp["eclose_diff_1"]<=-300), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<2000)&(sp["eclose_diff_1"]<=-400), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<3000)&(sp["eclose_diff_1"]<=-500), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<5000)&(sp["eclose_diff_1"]<=-700), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<7000)&(sp["eclose_diff_1"]<=-1000), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<10000)&(sp["eclose_diff_1"]<=-1500), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<15000)&(sp["eclose_diff_1"]<=-3000), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<20000)&(sp["eclose_diff_1"]<=-4000), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<30000)&(sp["eclose_diff_1"]<=-5000), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<50000)&(sp["eclose_diff_1"]<=-7000), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<70000)&(sp["eclose_diff_1"]<=-10000), "stop"] = -1
        sp.loc[(sp["eclose_tbc_pre_1"]<100000)&(sp["eclose_diff_1"]<=-15000), "stop"] = -1

        sp["window"] = np.where((sp.groupby("code").shift(1)["high"]*1.06<sp["low"])|(sp.groupby("code").shift(1)["low"]>sp["high"]*1.06), 1, 0)
        
        #移動平均
        sp["eclose_mean5"] = sp.groupby("code")["eclose"].rolling(5,min_periods=1).mean().reset_index()["eclose"]
        sp["eclose_ch_mean5"] = (sp["eclose"] - sp["eclose_mean5"]) / sp["eclose_mean5"]
        sp["eclose_mean10"] = sp.groupby("code")["eclose"].rolling(10,min_periods=1).mean().reset_index()["eclose"]
        sp["eclose_ch_mean10"] = (sp["eclose"] - sp["eclose_mean10"]) / sp["eclose_mean10"]
        sp["eclose_mean20"] = sp.groupby("code")["eclose"].rolling(20,min_periods=1).mean().reset_index()["eclose"]
        sp["eclose_ch_mean20"] = (sp["eclose"] - sp["eclose_mean20"]) / sp["eclose_mean20"]
        sp["eclose_mean40"] = sp.groupby("code")["eclose"].rolling(40,min_periods=1).mean().reset_index()["eclose"]
        sp["eclose_ch_mean40"] = (sp["eclose"] - sp["eclose_mean40"]) / sp["eclose_mean40"]
        sp["eclose_mean120"] = sp.groupby("code")["eclose"].rolling(120,min_periods=1).mean().reset_index()["eclose"]
        sp["eclose_ch_mean120"] = (sp["eclose"] - sp["eclose_mean120"]) / sp["eclose_mean120"]

        
        sa = pd.merge(sl, sf, on=["code"], how="inner")
        sa = pd.merge(sa, sp, on=["date", "code"], how="left")
        sa = sa.sort_values(['code', 'date'])


        sa = pd.merge(sa, m_conf, on=["code"], how="left")
        sa.loc[sa["fore_conf"].isnull(), "fore_conf"] = 0
        
        sa["jikaso_log"] = np.log((sa["issued"] * sa["eclose"]+1)/mil)


        sa["FYshift"] = np.where((sa["fyear_t"] != sa.groupby("code").shift(1)["fyear_t"]), 1, 0)

        sa["salesr_aad_t_ch"] = np.where((sa["code"] == sa.shift(1)["code"]) & (sa["accstd_t"] == sa.shift(1)["accstd_t"]) & (sa["fyear_t"] <= sa.shift(1)["fyear_t"] + 1), sa.groupby(["code", "accstd_t"])["sales_aad_t"].diff(1) / sa["jikaso_log"], np.nan)
        sa["opeinr_aad_t_ch"] = np.where((sa["code"] == sa.shift(1)["code"]) & (sa["accstd_t"] == sa.shift(1)["accstd_t"]) & (sa["fyear_t"] <= sa.shift(1)["fyear_t"] + 1), sa.groupby(["code", "accstd_t"])["opein_aad_t"].diff(1) / sa["jikaso_log"], np.nan)
        sa["ordinr_aad_t_ch"] = np.where((sa["code"] == sa.shift(1)["code"]) & (sa["accstd_t"] == sa.shift(1)["accstd_t"]) & (sa["fyear_t"] <= sa.shift(1)["fyear_t"] + 1), sa.groupby(["code", "accstd_t"])["ordin_aad_t"].diff(1) / sa["jikaso_log"], np.nan)
        sa["netinr_aad_t_ch"] = np.where((sa["code"] == sa.shift(1)["code"]) & (sa["accstd_t"] == sa.shift(1)["accstd_t"]) & (sa["fyear_t"] <= sa.shift(1)["fyear_t"] + 1), sa.groupby(["code", "accstd_t"])["netin_aad_t"].diff(1) / sa["jikaso_log"], np.nan)
    
        

        sf["opecf_aad_ch"] = sf.groupby("code")["opecf_aad"].pct_change(1)
        sf["fincf_aad_ch"] = sf.groupby("code")["fincf_aad"].pct_change(1)
        sf["invcf_aad_ch"] = sf.groupby("code")["invcf_aad"].pct_change(1)

        
        sa["qdiv_aad_t_ch"] = np.where(sa["qdiv_aad_t"] / sa["eclose"] > 0.06, sa.groupby("code")["qdiv_aad_t"].diff(1)/3, sa.groupby("code")["qdiv_aad_t"].diff(1))
        sa["ayld_t_ch"] = sa["qdiv_aad_t_ch"] / sa["eclose"]

        sa["EPS_sal_f"] = sa["sales_aad_f"] * mil / sa["issued"]
        sa["PER_sal_f"] = sa["eclose"] / sa["EPS_sal_f"]
        sa["PER_sal_log_f"] = np.log(sa["PER_sal_f"])


        eclose_ch_sec = sa.groupby("33sec")["eclose_ch_mid"].mean().reset_index()
        eclose_ch_sec = eclose_ch_sec.rename(columns={"eclose_ch_mid": "eclose_ch_sec",})
        sa = pd.merge(sa, eclose_ch_sec, on=["33sec"], how="left")

        sa = pd.get_dummies(sa, columns=['section'])
        sa = pd.get_dummies(sa, columns=['sizecode'])

        sa["ctype_f"] = np.where(sa["ctype_f"].isnull(), sa["ctype"], sa["ctype_f"])
        sa = pd.get_dummies(sa, columns=['ctype'])
        sa = pd.get_dummies(sa, columns=['ctype_f'])

        
        sp_fd = sp.groupby("code")[["code","date"]].head(1).copy()
        sp_fd = sp_fd.rename(columns={"date": "sp_fdate"})
        sa = pd.merge(sa, sp_fd, on=['code'], how="left")


        sa["lc_mature"] = 0
        sa["lc_mature"] = np.where(sa["sp_fdate"] <= dt.datetime(2016, 1, 4), 1, 0)

        sa["lc_young"] = 0
        sa["lc_young"] = np.where((sa["lc_mature"]==0)
                           & (sa["date"] - sa["sp_fdate"] <= dt.timedelta(days=1440)), 1, 0)

        
        
        sa = pd.merge(sa,m_yutai, on=["33sec"], how="left")
        sa.loc[(sa["yutai"].isnull()), "yutai"] = 0

        sa.loc[(sa["kenri_f"]==0)&(sa["kenriochi_f"]==0)&(sa["post_kenriochi_f"]==0), "yutai"] = 0
        sa.loc[(sa["kenri_f"]==0)&(sa["kenriochi_f"]==0)&(sa["post_kenriochi_f"]==0), "yutai"] = 0

#        sa.loc[(sa["post_kenriochi_f"]==0), "md_high_20"] = 0
#        sa.loc[(sa["post_kenriochi_f"]==0), "md_low_20"] = 0

        
        
        sa = sa[sa["target"]==1]
        

        for j in range(2):
            for i in range(10):
                if len(sa["ordinr_aad_t_ch"].isnull()) > 0:
                    sa["ordinr_aad_t_ch"] = np.where(sa["ordinr_aad_t_ch"].isnull(), sa["opeinr_aad_t_ch"], sa["ordinr_aad_t_ch"])
                else:
                    break

            for i in range(10):
                if len(sa["opeinr_aad_t_ch"].isnull()) > 0:
                    sa["opeinr_aad_t_ch"] = np.where(sa["opeinr_aad_t_ch"].isnull(),  
                                                np.where(sa["ordinr_aad_t_ch"].isnull(), sa["salesr_aad_t_ch"], sa["ordinr_aad_t_ch"]), 
                                            sa["opeinr_aad_t_ch"])
                else:
                    break

            for i in range(10):
                if len(sa["salesr_aad_t_ch"].isnull()) > 0:
                    sa["salesr_aad_t_ch"] = np.where(sa["salesr_aad_t_ch"].isnull(), 
                                            np.where(sa["opeinr_aad_t_ch"].isnull(), sa["ordinr_aad_t_ch"], sa["opeinr_aad_t_ch"]), 
                                            sa["salesr_aad_t_ch"])
                else:
                    break

        for i in range(10):
            if len(sa["netinr_aad_t_ch"].isnull()) > 0:
                sa["netinr_aad_t_ch"] =  np.where(sa["netinr_aad_t_ch"].isnull(),
                                            np.where(sa["ordinr_aad_t_ch"].isnull(), sa["opeinr_aad_t_ch"], sa["ordinr_aad_t_ch"]), 
                                                  sa["netinr_aad_t_ch"])

            else:
                break


#        sa["opeinr_akakuro"] = np.where((sa["opein"] * sa.shift(1)["opein"]<0), 1, 0)
#        sa["opeinr_akaaka"] = np.where((sa["opein"] < 0) & (sa.shift(1)["opein"] < 0), 1, 0)
#        sa["opeinr_f_akakuro"] = np.where((sa["opein_f"] * sa.shift(1)["opein_f"]<0), 1, 0)
#        sa["opeinr_f_akaaka"] = np.where((sa["opein_f"] < 0) & (sa.shift(1)["opein_f"] < 0), 1, 0)
#        sa["ordinr_akakuro"] = np.where((sa["ordin"] * sa.shift(1)["ordin"]<0), 1, 0)
#        sa["ordinr_akaaka"] = np.where((sa["ordin"] < 0) & (sa.shift(1)["ordin"] < 0), 1, 0)
#        sa["ordinr_f_akakuro"] = np.where((sa["ordin_f"] * sa.shift(1)["ordin_f"]<0), 1, 0)
#        sa["ordinr_f_akaaka"] = np.where((sa["ordin_f"] < 0) & (sa.shift(1)["ordin_f"] < 0), 1, 0)
#        sa["netinr_akakuro"] = np.where((sa["netin"] * sa.shift(1)["netin"]<0), 1, 0)
#        sa["netinr_akaaka"] = np.where((sa["netin"] < 0) & (sa.shift(1)["netin"] < 0), 1, 0)
#        sa["netinr_f_akakuro"] = np.where((sa["netin_f"] * sa.shift(1)["netin_f"]<0), 1, 0)
#        sa["netinr_f_akaaka"] = np.where((sa["netin_f"] < 0) & (sa.shift(1)["netin_f"] < 0), 1, 0)

        sa["opeinr_akakuro"] = np.where((sa["opein_aad_t"] * sa.groupby("code").shift(1)["opein_aad_t"]<0), 1, 0)
        sa["opeinr_akaaka"] = np.where((sa["opein_aad_t"] < 0) & (sa.groupby("code").shift(1)["opein_aad_t"] < 0), 1, 0)
        sa["ordinr_akakuro"] = np.where((sa["ordin_aad_t"] * sa.groupby("code").shift(1)["ordin_aad_t"]<0), 1, 0)
        sa["ordinr_akaaka"] = np.where((sa["ordin_aad_t"] < 0) & (sa.groupby("code").shift(1)["ordin_aad_t"] < 0), 1, 0)
        sa["netinr_akakuro"] = np.where((sa["netin_aad_t"] * sa.groupby("code").shift(1)["netin_aad_t"]<0), 1, 0)
        sa["netinr_akaaka"] = np.where((sa["netin_aad_t"] < 0) & (sa.groupby("code").shift(1)["netin_aad_t"] < 0), 1, 0)
        
        sa["nasset_akakuro"] = np.where((sa["nasset"] > 0) & (sa.groupby("code").shift(1)["nasset"] <= 0), 1,  
                                        np.where((sa["nasset"] <= 0) & (sa.groupby("code").shift(1)["nasset"] > 0), -1, 0))
        
        sa["punit_ch_date"] = np.where(sa["date"]>="2018-10-01", 1, 0)


        
        
        sa = sa[sa["date"]>=cls.TEST_START]


        x_col1_g = [
                 "FYshift", "ctype_GB", "ctype_BK","ctype_SE","ctype_IN","ctype_f_GB","ctype_f_BK","ctype_f_SE","ctype_f_IN",
                "salesr_aad_t_ch", "opeinr_aad_t_ch", "ordinr_aad_t_ch", "netinr_aad_t_ch",
                "opeinr_akakuro", "opeinr_akaaka", "ordinr_akakuro", "ordinr_akaaka", "netinr_akakuro", "netinr_akaaka",
                "lc_young",
                "lc_mature",
                "noresult", "nofrct","ayld_t_ch",#"kenriyld",
                "yutai",
                #"md_high_20", "md_low_20",
                "kenri_f", "kenriochi_f","post_kenriochi_f",
                "fore_conf",
                "eclose",
                #"eclose_ori_delta",
                "punit_ch_date",
                "eclose_ch_mean5", "eclose_ch_mean10",
                "eclose_ch_1","eclose_ch_5","eclose_ch_10","eclose_ch_20", "eclose_ch_40", "eclose_ch_120",
                "eclose_ch_sec", "vola_code",
                #"eclose_ch_mid_macro",
                "soyaku",
                #"invcf_aad_ch",
                #"EPS_ope_f","PER_ope_f",
                "PER_sal_log_f",
                  "vola", "vol_ch", "jikaso_log","stop","window",
                 "section_First Section (Domestic)", "section_JASDAQ(Growth/Domestic)", "section_JASDAQ(Standard / Domestic)", "section_Mothers (Domestic)",
                 "section_Second Section(Domestic)",
                "sizecode_1","sizecode_2","sizecode_4","sizecode_6","sizecode_7","sizecode_-",
                ]
        x_col1_h = ["eclose_ch_60"]
        x_col1_l = ["eclose_ch_mean20", "eclose_ch_mean40", "eclose_ch_mean120", "eclose_ch_mid","sellmax_code","sellmax_macro","nasset_akakuro"]

        x_col_h = x_col1_g + x_col1_h
        x_col_l = x_col1_g + x_col1_l
        #sa[x_col_h] = sa[x_col_h].fillna(0)
        #sa[x_col_l] = sa[x_col_l].fillna(0)
        #print(len(sa[sa[x_col_h].isnull()]))
        #print(len(sa[sa[x_col_l].isnull()]))
        #print(sa[sa[x_col_h].isnull()])

        
        ################################
        df = sa.copy()
        
        try:
#            pred = np.array(cls.model.predict(sa[x_col].values))
            pred_h = np.array(cls.model_h.predict(sa[x_col_h].values, num_iteration=cls.model_h.best_iteration))
            pred_l = np.array(cls.model_l.predict(sa[x_col_l].values, num_iteration=cls.model_l.best_iteration))
            pred = np.stack([pred_h,pred_l])
            pred = pred.reshape(2, len(pred[1]))
            pred = pred.T
            df2 = pd.DataFrame(data=pred, index=df.index, columns=cls.TARGET_LABELS)
            df2.reset_index(drop=True)
            df = pd.concat([df[["date","code"]], df2], axis=1)
        except Exception as e:
            raise ValueError("sa[x_col_h]", sa[x_col_h])
#            raise ValueError("df", df[['code', 'label_high_20', 'label_low_20']])
            raise ValueError("pred", pred)

        df.set_index("date", inplace=True)
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(str)
        
        try:
            output_columns = ['code', 'label_high_20', 'label_low_20']
            #df.to_csv("out.csv", header=False, index=False, columns=output_columns)
            out = io.StringIO()
            df.to_csv(out, header=False, index=False, columns=output_columns)
        except Exception as e:
            raise ValueError("df_2", df['code', 'label_high_20', 'label_low_20'])
        
        return out.getvalue()
    


# dataset_dir =".."
# 
# inputs = ScoringService.get_inputs(dataset_dir)
# ScoringService.get_model()
# ScoringService.predict(inputs)

# In[ ]:




