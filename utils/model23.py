import requests
import pandas as pd
from unsync import unsync
from dataclasses import asdict, dataclass
import csv  
from datetime import datetime
from datetime import date
import time
import numpy as np
import json
from web3 import Web3

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import utils.CurveSim as CurveSim 
from utils import constants as c

import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:,.2f}'.format

web3 = Web3(Web3.HTTPProvider(c.TOKEN))

   

def mock(make_request, web3):
    def middleware(method, params):
        if method == "eth_chainId":
            return { "id": 0, "jsonrpc": "2.0", "result": 1 }
        return make_request(method, params)
    return middleware

web3.middleware_onion.add(mock, "chain_id_mock")
# get Bin 1, Bin 2 (subBin with stablecoins as debt) and Bin 3 (subBin with stablecoins, eth and wbtc as debt/collateral)
B1v2 = pd.read_csv("data/b1.csv")
B2v2 = pd.read_csv("data/b2.csv")
B3v2 = pd.read_csv("data/b3.csv")
B21v2= pd.read_csv("data/b21_debt_stables.csv")
B31v2 = pd.read_csv("data/b31_eth_wbtc_stables.csv")
# get v3 Bin 1, Bin 2 (subBin with stablecoins as debt) and Bin 3 (subBin with stablecoins, eth and wbtc as debt/collateral)
B1v3 = pd.read_csv("data/b1_v3.csv")
B2v3 = pd.read_csv("data/b2_v3.csv")
B3v3 = pd.read_csv("data/b3_v3.csv")
B21v3 = pd.read_csv("data/b21_debt_stables_v3.csv")
B31v3 = pd.read_csv("data/b31_eth_wbtc_stables_v3.csv")
def eth_last_price():
    ## function to get current ETH price from coingecko            
    req = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd&include_market_cap=false&include_24hr_vol=false&include_24hr_change=false&include_last_updated_at=false'
                      ).json()
    price = req['ethereum']['usd']
    
    
    return price

def steth_last_price():
    ## function to get current stETH price from coingecko                
    req = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids=staked-ether&vs_currencies=usd&include_market_cap=false&include_24hr_vol=false&include_24hr_change=false&include_last_updated_at=false'
                      ).json()
    price = req['staked-ether']['usd']

    return price

def wsteth_last_price():
    ## function to get current wstETH price from coingecko                
    req = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids=wrapped-steth&vs_currencies=usd&include_market_cap=false&include_24hr_vol=false&include_24hr_change=false&include_last_updated_at=false'
                      ).json()
    price = req['wrapped-steth']['usd']

    return price

def get_wsteth_steth_price():
    ## function to get current wstETH:stETH price 
    abi = json.load(open("utils/WSTETH.json"))
    contract = web3.eth.contract(address= Web3.toChecksumAddress(c.WSTETH_CONTRACT), abi=abi)
    price = contract.functions.getStETHByWstETH(1000000000000000000).call()
    price = price / pow(10,18)
    
    return price

def get_wsteth_usd_price():
    ## function to get current ETH price from Aave Oracle        
    abi = json.load(open("utils/AaveOracle_v3.json"))
    contract = web3.eth.contract(address=c.AAVE3_ORACLE_CONTRACT, abi=abi)
    price = contract.functions.getAssetPrice(Web3.toChecksumAddress('0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0')).call()
    price = price / pow(10,8)
    
    return price

def get_risks(data, collateral_loan_ratio):
    ## calculate the risk level for each position and returns the positions sorted by risk
    dfrisk = pd.DataFrame(data = data)
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    dfrisk['risk_rating'] = [(x > risk_rating_list[0] and 'A') 
                       or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                       or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                       or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                       or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                       or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                       or (risk_rating_list[5] <=x and 'liquidation') for x in dfrisk['healthf']]

    return dfrisk.query('amount > 0').sort_values(by = 'healthf', ascending = False) 


def v2get_distr(data):
    ## calculate and return a pivot table by risk levels    
       
    risk_distr =  data.pivot_table(index = 'risk_rating', values = ['amount'], aggfunc = ['sum', 'count'])
    risk_distr.columns = ['stETH','cnt']

    risk_distr['stETH'] = risk_distr['stETH']
    risk_distr['percent'] = (risk_distr['stETH']/risk_distr['stETH'].sum())*100
    risk_distr['average'] = risk_distr['stETH']/risk_distr['cnt']

    median_a0 = data.query('risk_rating == "A"')['amount'].median()
    data.loc[data['risk_rating']=='A','median'] = median_a0
    median_b0 = data.query('risk_rating == "B+"')['amount'].median()
    data.loc[data['risk_rating']=='B+','median'] = median_b0
    median_b1 = data.query('risk_rating == "B"')['amount'].median()
    data.loc[data['risk_rating']=='B','median'] = median_b1
    median_b2 = data.query('risk_rating == "B-"')['amount'].median()
    data.loc[data['risk_rating']=='B-','median'] = median_b2
    median_c0 = data.query('risk_rating == "C"')['amount'].median()
    data.loc[data['risk_rating']=='C','median'] = median_c0
    median_d0 = data.query('risk_rating == "D"')['amount'].median()
    data.loc[data['risk_rating']=='D','median'] = median_d0
    median_il = data.query('risk_rating == "is liquidating"')['amount'].median()
    data.loc[data['risk_rating']=='liquidation','median'] = median_il

    median_distr =  data.pivot_table(index = 'risk_rating', values = ['median'], aggfunc = 'min')

    risk_distr = risk_distr.merge(median_distr, on = 'risk_rating', how = 'outer') 

    return risk_distr.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)


def v3get_distr(data):
    ## calculate and return a pivot table by risk levels    
       
    risk_distr =  data.pivot_table(index = 'risk_rating', values = ['amount'], aggfunc = ['sum', 'count'])
    risk_distr.columns = ['wstETH','cnt']

    risk_distr['wstETH'] = risk_distr['wstETH']
    risk_distr['percent'] = (risk_distr['wstETH']/risk_distr['wstETH'].sum())*100
    risk_distr['average'] = risk_distr['wstETH']/risk_distr['cnt']

    median_a0 = data.query('risk_rating == "A"')['amount'].median()
    data.loc[data['risk_rating']=='A','median'] = median_a0
    median_b0 = data.query('risk_rating == "B+"')['amount'].median()
    data.loc[data['risk_rating']=='B+','median'] = median_b0
    median_b1 = data.query('risk_rating == "B"')['amount'].median()
    data.loc[data['risk_rating']=='B','median'] = median_b1
    median_b2 = data.query('risk_rating == "B-"')['amount'].median()
    data.loc[data['risk_rating']=='B-','median'] = median_b2
    median_c0 = data.query('risk_rating == "C"')['amount'].median()
    data.loc[data['risk_rating']=='C','median'] = median_c0
    median_d0 = data.query('risk_rating == "D"')['amount'].median()
    data.loc[data['risk_rating']=='D','median'] = median_d0
    median_il = data.query('risk_rating == "is liquidating"')['amount'].median()
    data.loc[data['risk_rating']=='liquidation','median'] = median_il

    median_distr =  data.pivot_table(index = 'risk_rating', values = ['median'], aggfunc = 'min')

    risk_distr = risk_distr.merge(median_distr, on = 'risk_rating', how = 'outer') 

    return risk_distr.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)


def get_scale(collateral_loan_ratio):
    ## calculate and return scale-table       
    risk_ratio1 = [str(f'{100/i:.0f}%') for i in collateral_loan_ratio]
    risk_ratio2 = ['0']+risk_ratio1[:-1]
    risk_ratio = [f'{risk_ratio2[i]} - {risk_ratio1[i]}' for i in range(len(risk_ratio1))]
    risk_ratio.append('>=100%')
    collateral_loan_ratio_list1 = [str(f'{i:.2f}') for i in collateral_loan_ratio]
    collateral_loan_ratio_list2 = ['>']+ collateral_loan_ratio_list1[:-1]
    collateral_loan_ratio_list = [f'{collateral_loan_ratio_list1[i]} - {collateral_loan_ratio_list2[i]}' for i in range(len(collateral_loan_ratio_list2))]
    collateral_loan_ratio_list.append('=< 1.00')
    risk_rating = ['A','B+','B','B-','C','D','liquidation']

    return pd.DataFrame(data={'health factor' : collateral_loan_ratio_list, 'risk ratio' : risk_ratio, 'risk_rating' : risk_rating})


def get_scale_dic(collateral_loan_ratio):
    return [i for i in collateral_loan_ratio]

def v2changepeg_rec_otherdebt(df,collateral_loan_ratio, new_peg, current_peg):
    ## function to calculate risk structure with change of stETH:ETH rate        
    dftemp = df.copy()
    
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    i = new_peg/current_peg
   
    dftemp['coll_ratio'] = ((dftemp['collateral']*0.0001*dftemp['threshold'] - dftemp['collateral_steth_calc']*c.AAVE2_LIQUIDATION_THRESHOLD
                            + dftemp['collateral_steth_calc']*(i)*c.AAVE2_LIQUIDATION_THRESHOLD)/
                           dftemp['debt'])

    dftemp[f'risk'] = [(x > risk_rating_list[0] and 'A') 
                       or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                       or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                       or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                       or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                       or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                       or (risk_rating_list[5] <=x and 'liquidation') for x in dftemp['coll_ratio']]
    
    dftemp['max_debt'] = np.maximum.reduce(dftemp[['ethdebt', 'fei_debt','usdc_debt', 'usdt_debt', 'dai_debt','frax_debt', 'gusd_debt', 'susd_debt', 'tusd_debt', 'wbtc_debt']].values, axis=1)
    dftemp['namount_tmp'] = dftemp['max_debt']*c.AAVE2_CLOSE_FACTOR*c.AAVE2_LIQUIDATION_BONUS/new_peg
    dftemp['namount'] = dftemp[['namount_tmp','amount']].values.min(axis=1)
    dftemp.loc[dftemp['risk'] != False, 'namount'] = dftemp['amount']
    
    #Aave: calc MaxAmountOfCollateral to liquidate  
    dftemp.loc[dftemp['risk'] == False, 'risk'] = 'liquidation'
    
    risk_distr_ch =  dftemp.pivot_table(index = f'risk', values = ['namount'], aggfunc = ['sum'])
    forname = round(new_peg,2) 
    risk_distr_ch[f'1:{forname}'] = risk_distr_ch[('sum', 'namount')]
    risk_distr_ch = risk_distr_ch.drop(('sum', 'namount'), 1)
    
    #liquidation of positions
    liquidated_users = dftemp.query('risk == "liquidation"').index.to_list()
    for u in liquidated_users:
        dftemp.at[u,'amount'] = dftemp.at[u,'amount'] - dftemp.at[u,'namount']

    risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)
    risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
            
    
    return risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0), dftemp

def v3changepeg_rec_otherdebt(df,collateral_loan_ratio, new_peg, current_peg):
    ## function to calculate risk structure with change of stETH:ETH rate        
    dftemp = df.copy()
    
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    i = new_peg/current_peg
    wsteth_steth_price = get_wsteth_steth_price()
   
    dftemp['coll_ratio'] = ((dftemp['collateral']*0.0001*dftemp['threshold'] - dftemp['collateral_steth_calc']*c.AAVE3_LIQUIDATION_THRESHOLD
                            + dftemp['collateral_steth_calc']*(i)*c.AAVE3_LIQUIDATION_THRESHOLD)/
                           dftemp['debt'])

    dftemp[f'risk'] = [(x > risk_rating_list[0] and 'A') 
                       or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                       or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                       or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                       or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                       or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                       or (risk_rating_list[5] <=x and 'liquidation') for x in dftemp['coll_ratio']]
    
    dftemp['max_debt'] = np.maximum.reduce(dftemp[['ethdebt_calc','usdc_debt', 'usdt_debt', 'dai_debt','lusd_debt','wbtc_debt']].values, axis=1)
        
    dftemp['namount_tmp'] = dftemp['max_debt']*c.AAVE3_CLOSE_FACTOR*c.AAVE3_LIQUIDATION_BONUS/(new_peg*wsteth_steth_price*dftemp['eth_price'])
    dftemp['namount'] = dftemp[['namount_tmp','amount']].values.min(axis=1)
    dftemp.loc[dftemp['risk'] != False, 'namount'] = dftemp['amount']
        
    #Aave: calc MaxAmountOfCollateral to liquidate  
    dftemp.loc[dftemp['risk'] == False, 'risk'] = 'liquidation'
    
    risk_distr_ch =  dftemp.pivot_table(index = f'risk', values = ['namount'], aggfunc = ['sum'])
    forname = round(new_peg,2) 
    risk_distr_ch[f'1:{forname}'] = risk_distr_ch[('sum', 'namount')]
    risk_distr_ch = risk_distr_ch.drop(('sum', 'namount'), 1)
    
    #liquidation of positions
    liquidated_users = dftemp.query('risk == "liquidation"').index.to_list()
    for u in liquidated_users:
        dftemp.at[u,'amount'] = dftemp.at[u,'amount'] - dftemp.at[u,'namount']

    risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)
    risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
                
    
    return risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0), dftemp



def v2changepeg_rec_cycle_otherdebt(df, risk_distr, collateral_loan_ratio, current_peg, min_peg, peg_step): 
    ## function to calculate risk structure for stETH:ETH rate's change in range
     
    risk_distr_temp = risk_distr.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
        
    for q in range(int((round(current_peg,2))*100), int((round(min_peg,2) - 0.01)*100), (-1)*int(100*peg_step)):                 
        
        vpg = q/100
        
        (r, df_) = v2changepeg_rec_otherdebt(df,collateral_loan_ratio, vpg, current_peg)
        
        risk_distr_temp = risk_distr_temp.merge(r, how = 'outer', left_index=True, right_index=True)

    risk_distr_temp = risk_distr_temp.drop('stETH', 1)   
    risk_distr_temp.columns = ['_'.join(col).rstrip('_') for col in risk_distr_temp.columns.values]
    return risk_distr_temp.fillna(0)

def v3changepeg_rec_cycle_otherdebt(df, risk_distr, collateral_loan_ratio, current_peg, min_peg, peg_step): 
    ## function to calculate risk structure for stETH:ETH rate's change in range
     
    risk_distr_temp = risk_distr.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
        
    for q in range(int((round(current_peg,2))*100), int((round(min_peg,2) - 0.01)*100), (-1)*int(100*peg_step)):                 
        
        vpg = q/100
        
        (r, df_) = v3changepeg_rec_otherdebt(df,collateral_loan_ratio, vpg, current_peg)
        
        risk_distr_temp = risk_distr_temp.merge(r, how = 'outer', left_index=True, right_index=True)

    risk_distr_temp = risk_distr_temp.drop('wstETH', 1)   
    risk_distr_temp.columns = ['_'.join(col).rstrip('_') for col in risk_distr_temp.columns.values]
    return risk_distr_temp.fillna(0)

def v2changepeg_rec_ethdebt(df,collateral_loan_ratio, new_peg, current_peg):
    ## function to calculate risk structure with change of stETH:ETH rate        
    dftemp = df.copy()
    
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    i = new_peg/current_peg
   
    dftemp['coll_ratio'] = ((dftemp['collateral']*0.0001*dftemp['threshold'] - dftemp['collateral_steth_calc']*c.AAVE2_LIQUIDATION_THRESHOLD
                            + dftemp['collateral_steth_calc']*(i)*c.AAVE2_LIQUIDATION_THRESHOLD)/
                           dftemp['debt'])

    dftemp[f'risk'] = [(x > risk_rating_list[0] and 'A') 
                       or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                       or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                       or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                       or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                       or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                       or (risk_rating_list[5] <=x and 'liquidation') for x in dftemp['coll_ratio']]
    
    #Aave: calc MaxAmountOfCollateral to liquidate  
    dftemp.loc[dftemp['risk'] == False, 'namount'] = dftemp['ethdebt']*c.AAVE2_CLOSE_FACTOR*c.AAVE2_LIQUIDATION_BONUS/new_peg
    dftemp.loc[dftemp['risk'] != False, 'namount'] = dftemp['amount']
    
    dftemp.loc[dftemp['risk'] == False, 'risk'] = 'liquidation'
    
    risk_distr_ch =  dftemp.pivot_table(index = f'risk', values = ['namount'], aggfunc = ['sum'])
    forname = round(new_peg,2) 
    risk_distr_ch[f'1:{forname}'] = risk_distr_ch[('sum', 'namount')]
    risk_distr_ch = risk_distr_ch.drop(('sum', 'namount'), 1)
    
    #liquidation of positions
    liquidated_users = dftemp.query('risk == "liquidation"').index.to_list()
    for u in liquidated_users:
        dftemp.at[u,'amount'] = dftemp.at[u,'amount'] - dftemp.at[u,'namount']
    
    risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)
    risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
    
    return risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0), dftemp


def v2changepeg_rec_cycle_ethdebt(df, risk_distr, collateral_loan_ratio, current_peg, min_peg, peg_step): 
    ## function to calculate risk structure for stETH:ETH rate's change in range
     
    risk_distr_temp = risk_distr.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
        
    for q in range(int((round(current_peg,2))*100), int((round(min_peg,2) - 0.01)*100), (-1)*int(100*peg_step)):                 
        
        vpg = q/100
    
        (r, df_) = v2changepeg_rec_ethdebt(df,collateral_loan_ratio, vpg, current_peg)
        risk_distr_temp = risk_distr_temp.merge(r, how = 'outer', left_index=True, right_index=True)

    risk_distr_temp = risk_distr_temp.drop('stETH', 1)   
    risk_distr_temp.columns = ['_'.join(col).rstrip('_') for col in risk_distr_temp.columns.values]
    return risk_distr_temp.fillna(0)

def v3changepeg_rec_ethdebt_emode(df,collateral_loan_ratio, new_peg, current_peg):
    ## function to calculate risk structure with change of stETH:ETH rate        
    dftemp = df.copy()
    
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    i = new_peg/current_peg
    wsteth_steth_price = get_wsteth_steth_price()
   
    dftemp['coll_ratio'] = ((dftemp['collateral']*0.0001*dftemp['threshold'] - dftemp['collateral_steth_calc']*c.AAVE3_LIQUIDATION_THRESHOLD_EMODE
                            + dftemp['collateral_steth_calc']*i*c.AAVE3_LIQUIDATION_THRESHOLD_EMODE)/dftemp['debt'])

    dftemp[f'risk'] = [(x > risk_rating_list[0] and 'A') 
                       or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                       or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                       or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                       or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                       or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                       or (risk_rating_list[5] <=x and 'liquidation') for x in dftemp['coll_ratio']]
    
    #Aave: calc MaxAmountOfCollateral to liquidate  
    
    dftemp['namount_tmp'] = dftemp['ethdebt_calc']*c.AAVE3_CLOSE_FACTOR*c.AAVE3_LIQUIDATION_BONUS_EMODE/(new_peg*wsteth_steth_price*dftemp['eth_price'])
    dftemp['namount'] = dftemp[['namount_tmp','amount']].values.min(axis=1)
    
    dftemp.loc[dftemp['risk'] != False, 'namount'] = dftemp['amount']
    
    dftemp.loc[dftemp['risk'] == False, 'risk'] = 'liquidation'
    
    risk_distr_ch =  dftemp.pivot_table(index = f'risk', values = ['namount'], aggfunc = ['sum'])
    forname = round(new_peg,2) 
    risk_distr_ch[f'1:{forname}'] = risk_distr_ch[('sum', 'namount')]
    risk_distr_ch = risk_distr_ch.drop(('sum', 'namount'), 1)
    
    #liquidation of positions
    liquidated_users = dftemp.query('risk == "liquidation"').index.to_list()
    for u in liquidated_users:
        dftemp.at[u,'amount'] = dftemp.at[u,'amount'] - dftemp.at[u,'namount']
    
    risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)
    try:
        risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
    except IndexError:
        0
        
    return risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0), dftemp

def v3changepeg_rec_cycle_ethdebt_emode(df, risk_distr, collateral_loan_ratio, current_peg, min_peg, peg_step): 
    ## function to calculate risk structure for stETH:ETH rate's change in range
     
    risk_distr_temp = risk_distr.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
        
    for q in range(int((round(current_peg,2))*100), int((round(min_peg,2) - 0.01)*100), (-1)*int(100*peg_step)):                 
        
        vpg = q/100
        
        (r, df_) = v3changepeg_rec_ethdebt_emode(df,collateral_loan_ratio, vpg, current_peg)
        
        risk_distr_temp = risk_distr_temp.merge(r, how = 'outer', left_index=True, right_index=True)

    risk_distr_temp = risk_distr_temp.drop('wstETH', 1)   
    risk_distr_temp.columns = ['_'.join(col).rstrip('_') for col in risk_distr_temp.columns.values]
    return risk_distr_temp.fillna(0)



def v2changeprice(df,collateral_loan_ratio,eth_price, i): #i - % of change price, e.g if i=0.05 then price is changed by 5%
    ## function to calculate risk structure with ETH price's change            
    dftemp = df.copy()
    
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    changed_price = round(eth_price*(1-i),0)
    
    #if changed_price > 0:
    dftemp['coll_ratio'] = dftemp['collateral']*0.0001*dftemp['threshold']/(dftemp['debt'] - 
                                                                                dftemp['debt_stable_calc'] + 
                                                                                (eth_price/changed_price)*dftemp['debt_stable_calc'])
    dftemp[f'risk'] = [(x > risk_rating_list[0] and 'A') 
                        or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                        or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                        or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                        or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                        or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                        or (risk_rating_list[5] <=x and 'liquidation') for x in dftemp['coll_ratio']]
        
    dftemp['max_debt'] = np.maximum.reduce(dftemp[['ethdebt', 'fei_debt','usdc_debt', 'usdt_debt', 'dai_debt','frax_debt', 'gusd_debt', 'susd_debt', 'tusd_debt']].values, axis=1)
    #Aave: calc MaxAmountOfCollateral to liquidate 
    dftemp['namount_tmp'] = (eth_price/changed_price)*dftemp['max_debt']*c.AAVE2_CLOSE_FACTOR*c.AAVE2_LIQUIDATION_BONUS
    dftemp['namount'] = dftemp[['namount_tmp','amount']].values.min(axis=1)
    dftemp.loc[dftemp['risk'] != False, 'namount'] = dftemp['amount']
    dftemp.loc[dftemp['risk'] == False, 'risk'] = 'liquidation'

    risk_distr_ch =  dftemp.pivot_table(index = f'risk', values = ['namount'], aggfunc = ['sum'])
    risk_distr_ch[f'{changed_price}'] = risk_distr_ch[('sum', 'namount')]
    risk_distr_ch = risk_distr_ch.drop(('sum', 'namount'), 1)
    

    liquidated_users = dftemp.query('risk == "liquidation"').index.to_list()
    for u in liquidated_users:
        dftemp.at[u,'amount'] = dftemp.at[u,'amount'] - dftemp.at[u,'namount']

    risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)
    try:
        risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
    except IndexError:
        0
    #risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
    
    
    return risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)



def v2changeprice_cycle(df, risk_distr, collateral_loan_ratio, min_price, max_price, vstep):
    ## function to calculate risk structure with ETH price's change in range
            
    risk_distr_temp = risk_distr.copy()
    risk_distr_temp = risk_distr_temp. iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()


    for q in range(int(max_price-vstep), int(min_price), (-1)*vstep):                 
        r = v2changeprice(df,collateral_loan_ratio, max_price, (1-q/max_price))
        risk_distr_temp = risk_distr_temp.merge(r, how = 'outer', left_index=True, right_index=True)
    
    risk_distr_temp.columns = ['_'.join(col).rstrip('_') for col in risk_distr_temp.columns.values]
    risk_distr_temp.rename(columns={'s_t_E_T_H': max_price}, inplace=True)
    
    return risk_distr_temp.fillna(0)

def v3changeprice(df,collateral_loan_ratio,eth_price, i): #i - % of change price, e.g if i=0.05 then price is changed by 5%
    ## function to calculate risk structure with ETH price's change            
    dftemp = df.copy()
    
    risk_rating_list = get_scale_dic(collateral_loan_ratio)
    changed_price = round(eth_price*(1-i),0)
    
    dftemp['coll_ratio'] = dftemp['collateral']*0.0001*dftemp['threshold']/(dftemp['debt'] - 
                                                                                dftemp['debt_stable_calc'] + 
                                                                                (eth_price/changed_price)*dftemp['debt_stable_calc'])
    dftemp[f'risk'] = [(x > risk_rating_list[0] and 'A') 
                        or (risk_rating_list[1] < x <= risk_rating_list[0] and 'B+') 
                        or (risk_rating_list[2] < x <= risk_rating_list[1] and 'B') 
                        or (risk_rating_list[3] < x <= risk_rating_list[2] and 'B-') 
                        or (risk_rating_list[4] < x <= risk_rating_list[3] and 'C') 
                        or (risk_rating_list[5] < x <= risk_rating_list[4] and 'D') 
                        or (risk_rating_list[5] <=x and 'liquidation') for x in dftemp['coll_ratio']]
    dftemp['ethdebt_calc'] = dftemp['ethdebt']*dftemp['eth_price']    
    dftemp['max_debt'] = np.maximum.reduce(dftemp[['ethdebt_calc', 'usdc_debt', 'usdt_debt', 'dai_debt', 'lusd_debt']].values, axis=1)
    
    #Aave: calc MaxAmountOfCollateral to liquidate 
    dftemp['namount_tmp'] = dftemp['max_debt']*c.AAVE3_CLOSE_FACTOR*c.AAVE3_LIQUIDATION_BONUS/round(get_wsteth_usd_price()*(1-i),2)
    dftemp['namount'] = dftemp[['namount_tmp','amount']].values.min(axis=1)
    dftemp.loc[dftemp['risk'] != False, 'namount'] = dftemp['amount']
    
    
    dftemp.loc[dftemp['risk'] == False, 'risk'] = 'liquidation'

    risk_distr_ch =  dftemp.pivot_table(index = f'risk', values = ['namount'], aggfunc = ['sum'])
    risk_distr_ch[f'{changed_price}'] = risk_distr_ch[('sum', 'namount')]
    risk_distr_ch = risk_distr_ch.drop(('sum', 'namount'), 1)
    
    
    liquidated_users = dftemp.query('risk == "liquidation"').index.to_list()
    for u in liquidated_users:
        dftemp.at[u,'amount'] = dftemp.at[u,'amount'] - dftemp.at[u,'namount']
    
    risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)
    try:
        risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
    except IndexError:
        0
    #risk_distr_ch.iloc[2,0] = risk_distr_ch.iloc[2,0] + dftemp.query('risk == "liquidation"')['amount'].sum()
    
    return risk_distr_ch.reindex(['A','B+','B','B-','C','D','liquidation']).fillna(0)

def v3changeprice_cycle(df, risk_distr, collateral_loan_ratio, min_price, max_price, vstep):
    ## function to calculate risk structure with ETH price's change in range
            
    risk_distr_temp = risk_distr.copy()
    risk_distr_temp = risk_distr_temp. iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()


    for q in range(int(max_price-vstep), int(min_price), (-1)*vstep):                 
        r = v3changeprice(df,collateral_loan_ratio, max_price, (1-q/max_price))
        risk_distr_temp = risk_distr_temp.merge(r, how = 'outer', left_index=True, right_index=True)
    
    risk_distr_temp.columns = ['_'.join(col).rstrip('_') for col in risk_distr_temp.columns.values]
    risk_distr_temp.rename(columns={'w_s_t_E_T_H': max_price}, inplace=True)
    
    return risk_distr_temp.fillna(0)



def plot_liquidation_amounts(dfresult):    
    ## function to plot liquidation amount
    risk_cde = dfresult.copy()

    risk_cde = risk_cde.loc[['C','D', 'liquidation']]    
    risk_cde.loc['C+D+liquidation'] = risk_cde.sum(axis=0)
                
    risk_cde = risk_cde.T
    risk_cde['D'] = risk_cde['D'].fillna(0)
    risk_cde['liquidation'] = risk_cde['liquidation'].fillna(0) 

    rsize = risk_cde['D'].size + 1
    s1 = pd.Series(risk_cde.index) 
    s1.index = risk_cde.index
        
    
    risk_cde["X"] = s1 
    risk_cde['X'] = risk_cde['X'].fillna(0)

    return risk_cde.plot(x="X", y=["liquidation"])


def v2get_b1_current_risks():
    ## function to get current risk structure in Bin1
    b1scale = get_scale(c.b1v2_collateral_loan_ratio)
    
    print("Current risk structure of Bin1 (Aave v2)")
    b1scale = get_scale(c.b1v2_collateral_loan_ratio)
    b1scale.set_index('risk_rating')
    b1data = get_risks(B1v2, c.b1v2_collateral_loan_ratio)
    b1data

    b1risk_distribution = v2get_distr(b1data)
    b1risk_distribution_amount = b1risk_distribution[['stETH']]
    b1risk_distribution_count =  b1risk_distribution[['cnt']]
    b1risk_distribution_prc =  b1risk_distribution[['percent']]
    b1_total_steth_amount = b1data.amount.sum()
    b1_total_steth_count = b1data.amount.count()
    print(f"Bin1(v2) stats on {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:")
    print(f"Bin1(v2): Total amount of stETH locked: {b1_total_steth_amount:,.0f} stETH")
    print(f"Bin1(v2): Total number of stETH collateral: {b1_total_steth_count:,.0f}")

    b1risk_distribution_to_show = b1risk_distribution.merge(b1scale[['risk_rating', 'health factor', 'risk ratio']], on='risk_rating',  how="left")
    return b1risk_distribution_to_show

def v2get_b1_risks_for_changed_peg(new_peg):
    ## function to get risk structure in Bin1 for stETH:ETH rate's changed
    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b1scale = get_scale(c.b1v2_collateral_loan_ratio)
    b1scale.set_index('risk_rating')
    b1data = get_risks(B1v2, c.b1v2_collateral_loan_ratio)
    b1risk_distribution = v2get_distr(b1data)
    risk_distr_temp = b1risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    df = v2changepeg_rec_cycle_ethdebt(B1v2, b1risk_distribution, c.b1v2_collateral_loan_ratio, current_peg, new_peg, 0.01)
    
    return df.iloc[: , -1].to_frame() 

#get_b1_risks_for_range_of_peg shows table of risk structures for peg in range [new peg, current peg]
#example of use: get_b1_risks_for_range_of_peg(0.87)  
def v2get_b1_risks_for_range_of_peg( new_peg):
    ## function to get risk structure in Bin1 for stETH:ETH rate's changed in range (snapshot)
    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b1scale = get_scale(c.b1v2_collateral_loan_ratio)
    b1scale.set_index('risk_rating')
    b1data = get_risks(B1v2, c.b1v2_collateral_loan_ratio)
    b1risk_distribution = v2get_distr(b1data)
    risk_distr_temp = b1risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v2changepeg_rec_cycle_ethdebt(B1v2, b1risk_distribution, c.b1v2_collateral_loan_ratio, current_peg, new_peg, 0.01)

def v2plot_b1_liquidation_risk_by_peg(new_peg): 
    ## function to plot Bin1 liquidation's risk by stETH:ETH change    
    pl = plot_liquidation_amounts(v2get_b1_risks_for_range_of_peg(new_peg)) 
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v2 Bin 1 Simulation of liquidation risk")
    plt.xlabel("ETH:stETH rate")
    plt.ylabel("stETH amount")
    pl.grid()
    plt.show()
    

def v3get_b1_current_risks():
    ## function to get current risk structure in Bin1
    b1scale = get_scale(c.b1v3_collateral_loan_ratio)
    
    print("Current risk structure of Bin1 (Aave v3)")
    b1scale = get_scale(c.b1v3_collateral_loan_ratio)
    b1scale.set_index('risk_rating')
    b1data = get_risks(B1v3, c.b1v3_collateral_loan_ratio)
    b1data

    b1risk_distribution = v3get_distr(b1data)
    b1risk_distribution_amount = b1risk_distribution[['wstETH']]
    b1risk_distribution_count =  b1risk_distribution[['cnt']]
    b1risk_distribution_prc =  b1risk_distribution[['percent']]
    b1_total_steth_amount = b1data.amount.sum()
    b1_total_steth_count = b1data.amount.count()
    print(f"Bin1(v3) stats on {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:")
    print(f"Bin1(v3): Total amount of wstETH locked: {b1_total_steth_amount:,.0f} stETH")
    print(f"Bin1(v3): Total number of wstETH collateral: {b1_total_steth_count:,.0f}")

    b1risk_distribution_to_show = b1risk_distribution.merge(b1scale[['risk_rating', 'health factor', 'risk ratio']], on='risk_rating',  how="left")
    return b1risk_distribution_to_show

def v3get_b1_risks_for_changed_peg(new_peg):
    ## function to get risk structure in Bin1 for stETH:ETH rate's changed
    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b1scale = get_scale(c.b1v3_collateral_loan_ratio)
    b1scale.set_index('risk_rating')
    b1data = get_risks(B1v3, c.b1v3_collateral_loan_ratio)
    b1risk_distribution = v3get_distr(b1data)
    risk_distr_temp = b1risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    df = v3changepeg_rec_cycle_ethdebt_emode(B1v3, b1risk_distribution,c.b1v3_collateral_loan_ratio, current_peg, new_peg, 0.01)
    
    return df.iloc[: , -1].to_frame() 

#get_b1_risks_for_range_of_peg shows table of risk structures for peg in range [new peg, current peg]
#example of use: get_b1_risks_for_range_of_peg(0.87)  
def v3get_b1_risks_for_range_of_peg( new_peg):
    ## function to get risk structure in Bin1 for stETH:ETH rate's changed in range (snapshot)
    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b1scale = get_scale(c.b1v3_collateral_loan_ratio)
    b1scale.set_index('risk_rating')
    b1data = get_risks(B1v3, c.b1v3_collateral_loan_ratio)
    b1risk_distribution = v3get_distr(b1data)
    risk_distr_temp = b1risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v3changepeg_rec_cycle_ethdebt_emode(B1v3, b1risk_distribution,c.b1v3_collateral_loan_ratio, current_peg, new_peg, 0.01)

def v3plot_b1_liquidation_risk_by_peg(new_peg): 
    ## function to plot Bin1 liquidation's risk by stETH:ETH change    
    pl = plot_liquidation_amounts(v3get_b1_risks_for_range_of_peg(new_peg)) 
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v3 Bin 1 Simulation of liquidation risk")
    plt.xlabel("ETH:stETH rate")
    plt.ylabel("wstETH amount")
    pl.grid()
    plt.show()
    
def v2get_b21_current_risks():
    ## function to get current risk structure in Bin2
    b2scale = get_scale(c.b2v2_collateral_loan_ratio)
    
    print("Current risk structure of SubBin21 (Aave v2)")
    #Positions risk calculation
    b21data = get_risks(B21v2, c.b2v2_collateral_loan_ratio)
        
    #Calculation of table with risk rating (for Bin21)
    b21_risk_distribution = v2get_distr(b21data)
    b21_risk_distribution_amount = b21_risk_distribution[['stETH']]
    b21_risk_distribution_count =  b21_risk_distribution[['cnt']]
    b21_risk_distribution_prc =  b21_risk_distribution[['percent']]
    b21_total_steth_amount = b21data.amount.sum()
    b21_total_steth_count = b21data.amount.count()
    print(f"SubBin21(v2) stats on {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:")
    print(f"SubBin21(v2): Total stETH amount locked: {b21_total_steth_amount:,.0f} stETH")
    print(f"SubBin21(v2): Total number of positions with stETH collateral: {b21_total_steth_count:,.0f}")

    b21risk_distribution_to_show = b21_risk_distribution.merge(b2scale[['risk_rating', 'health factor', 'risk ratio']], on='risk_rating',  how="left")
    b21risk_distribution_to_show
    
    return b21risk_distribution_to_show    

def v3get_b21_current_risks():
    ## function to get current risk structure in Bin2
    b2scale = get_scale(c.b2v3_collateral_loan_ratio)
    
    print("Current risk structure of SubBin21 (Aave v3)")
    #Positions risk calculation
    b21data = get_risks(B21v3, c.b2v3_collateral_loan_ratio)
        
    #Calculation of table with risk rating (for Bin21)
    b21_risk_distribution = v3get_distr(b21data)
    b21_risk_distribution_amount = b21_risk_distribution[['wstETH']]
    b21_risk_distribution_count =  b21_risk_distribution[['cnt']]
    b21_risk_distribution_prc =  b21_risk_distribution[['percent']]
    b21_total_steth_amount = b21data.amount.sum()
    b21_total_steth_count = b21data.amount.count()
    print(f"SubBin21(v3) stats on {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:")
    print(f"SubBin21(v3): Total wstETH amount locked: {b21_total_steth_amount:,.0f} wstETH")
    print(f"SubBin21(v3): Total number of positions with wstETH collateral: {b21_total_steth_count:,.0f}")

    b21risk_distribution_to_show = b21_risk_distribution.merge(b2scale[['risk_rating', 'health factor', 'risk ratio']], on='risk_rating',  how="left")
    b21risk_distribution_to_show
    
    return b21risk_distribution_to_show    


def v2get_b21_risks_for_changed_price(price_change_prc):
    ## function to calculate SubBin21 risk structure with ETH price's change            
    # here price_change_prc is % of change price, e.g if price_change_prc=0.05 then price is changed by 5%

    eth_price = eth_last_price()
    return v2changeprice(B21v2,c.b2v2_collateral_loan_ratio,eth_price, price_change_prc) 


def v3get_b21_risks_for_changed_price(price_change_prc):
    ## function to calculate SubBin21 risk structure with ETH price's change            
    # here price_change_prc is % of change price, e.g if price_change_prc=0.05 then price is changed by 5%

    eth_price = eth_last_price()
    return v3changeprice(B21v3,c.b2v3_collateral_loan_ratio,eth_price, price_change_prc) 


def v2get_b21_risks_for_range_changed_price(max_price_change_prc, step):
    ## function to calculate SubBin21 risk structure with ETH price's change in range [current price, current price*max_price_change_prc]           
    # here max_price_change_prc is % of change price (min in range), e.g if max_price_change_prc=0.05 then price is changed by 5%
    eth_price = eth_last_price()
    b21scale = get_scale(c.b2v2_collateral_loan_ratio)
    b21scale.set_index('risk_rating')
    b21data = get_risks(B21v2, c.b2v2_collateral_loan_ratio)
    b21risk_distribution = v2get_distr(b21data)
    risk_distr_temp = b21risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v2changeprice_cycle(B21v2, b21risk_distribution, c.b2v2_collateral_loan_ratio, eth_price*max_price_change_prc,eth_price, step)

def v3get_b21_risks_for_range_changed_price(max_price_change_prc, step):
    ## function to calculate SubBin21 risk structure with ETH price's change in range [current price, current price*max_price_change_prc]           
    # here max_price_change_prc is % of change price (min in range), e.g if max_price_change_prc=0.05 then price is changed by 5%
    eth_price = eth_last_price()
    b21scale = get_scale(c.b2v3_collateral_loan_ratio)
    b21scale.set_index('risk_rating')
    b21data = get_risks(B21v3, c.b2v3_collateral_loan_ratio)
    b21risk_distribution = v3get_distr(b21data)
    risk_distr_temp = b21risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v3changeprice_cycle(B21v3, b21risk_distribution, c.b2v3_collateral_loan_ratio, eth_price*max_price_change_prc,eth_price, step)

def v3plot_b2_liquidation_risk_by_price(max_price_change_prc, step): 
    ## function to plot Bin2 liquidation's risk by stETH:ETH rate change    
    pl = plot_liquidation_amounts(v3get_b21_risks_for_range_changed_price(max_price_change_prc, step))
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v3 Bin 2 Simulation of liquidation risk")
    plt.xlabel("ETH price")
    plt.ylabel("wstETH amount")
    pl.grid()
    plt.show()  


def v2plot_b2_liquidation_risk_by_price(max_price_change_prc, step): 
    ## function to plot Bin2 liquidation's risk by stETH:ETH rate change    
    pl = plot_liquidation_amounts(v2get_b21_risks_for_range_changed_price(max_price_change_prc, step))
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v2 Bin 2 Simulation of liquidation risk")
    plt.xlabel("ETH price")
    plt.ylabel("stETH amount")
    pl.grid()
    plt.show()  


def v2get_b31_current_risks():
    ## function to get current risk structure in SubBin31
    b3scale = get_scale(c.b3v2_collateral_loan_ratio)
    
    print("Current risk structure of SubBin31 (Aave v2)")
    #Positions risk calculation
    b31data = get_risks(B31v2, c.b3v2_collateral_loan_ratio)
    
    #Calculation of table with risk rating (for SubBin31)
    b31_risk_distribution = v2get_distr(b31data)
    b31_risk_distribution_amount = b31_risk_distribution[['stETH']]
    b31_risk_distribution_count =  b31_risk_distribution[['cnt']]
    b31_risk_distribution_prc =  b31_risk_distribution[['percent']]
    b31_total_steth_amount = b31data.amount.sum()
    b31_total_steth_count = b31data.amount.count()
    print(f"SubBin31(v2) stats on {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:")
    print(f"SubBin31(v2): Total stETH amount locked: {b31_total_steth_amount:,.0f} stETH")
    print(f"SubBin31(v2): Total number of positions with stETH collateral: {b31_total_steth_count:,.0f}")

    b31risk_distribution_to_show = b31_risk_distribution.merge(b3scale[['risk_rating', 'health factor', 'risk ratio']], on='risk_rating',  how="left")
    b31risk_distribution_to_show
    
    return b31risk_distribution_to_show    

def v3get_b31_current_risks():
    ## function to get current risk structure in SubBin31
    b3scale = get_scale(c.b3v3_collateral_loan_ratio)
    
    print("Current risk structure of SubBin31 (Aave v3)")
    #Positions risk calculation
    b31data = get_risks(B31v3, c.b3v3_collateral_loan_ratio)
    
    #Calculation of table with risk rating (for SubBin31)
    b31_risk_distribution = v3get_distr(b31data)
    b31_risk_distribution_amount = b31_risk_distribution[['wstETH']]
    b31_risk_distribution_count =  b31_risk_distribution[['cnt']]
    b31_risk_distribution_prc =  b31_risk_distribution[['percent']]
    b31_total_steth_amount = b31data.amount.sum()
    b31_total_steth_count = b31data.amount.count()
    print(f"SubBin31(v3) stats on {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}:")
    print(f"SubBin31(v3): Total wstETH amount locked: {b31_total_steth_amount:,.0f} wstETH")
    print(f"SubBin31(v3): Total number of positions with wstETH collateral: {b31_total_steth_count:,.0f}")

    b31risk_distribution_to_show = b31_risk_distribution.merge(b3scale[['risk_rating', 'health factor', 'risk ratio']], on='risk_rating',  how="left")
    b31risk_distribution_to_show
    
    return b31risk_distribution_to_show    


def v2get_b31_risks_for_changed_peg(new_peg):
    ## function to get risk structure in SubBin31 for stETH:ETH rate's changed
    
    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b31scale = get_scale(c.b3v2_collateral_loan_ratio)
    b31scale.set_index('risk_rating')
    b31data = get_risks(B31v2, c.b3v2_collateral_loan_ratio)
    b31risk_distribution = v2get_distr(b31data)
    risk_distr_temp = b31risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    df = v2changepeg_rec_cycle_otherdebt(B31v2, b31risk_distribution, c.b3v2_collateral_loan_ratio, current_peg, new_peg, 0.01)
    
    return df.iloc[: , -1].to_frame() 
    
def v3get_b31_risks_for_changed_peg(new_peg):
    ## function to get risk structure in SubBin31 for stETH:ETH rate's changed
    
    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b31scale = get_scale(c.b3v3_collateral_loan_ratio)
    b31scale.set_index('risk_rating')
    b31data = get_risks(B31v3, c.b3v3_collateral_loan_ratio)
    b31risk_distribution = v3get_distr(b31data)
    risk_distr_temp = b31risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    df = v3changepeg_rec_cycle_otherdebt(B31v3, b31risk_distribution, c.b3v3_collateral_loan_ratio, current_peg, new_peg, 0.01)
    
    return df.iloc[: , -1].to_frame() 
 
def v2get_b31_risks_for_range_of_peg( new_peg):
    ## function to get risk structure in SubBin31 for stETH:ETH rate's changed in range (snapshot)
    # params:  new_peg - estimate liquidations risk in range [current stETH:ETH range, new_peg]

    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b31scale = get_scale(c.b3v2_collateral_loan_ratio)
    b31scale.set_index('risk_rating')
    b31data = get_risks(B31v2, c.b3v2_collateral_loan_ratio)
    b31risk_distribution = v2get_distr(b31data)
    risk_distr_temp = b31risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v2changepeg_rec_cycle_otherdebt(B31v2, b31risk_distribution, c.b3v2_collateral_loan_ratio, current_peg, new_peg, 0.01)

def v3get_b31_risks_for_range_of_peg( new_peg):
    ## function to get risk structure in SubBin31 for stETH:ETH rate's changed in range (snapshot)
    # params:  new_peg - estimate liquidations risk in range [current stETH:ETH range, new_peg]

    pool=CurveSim.create_steth_pool()
    current_peg = round(pool.get_exchange_rate().get('stETH_ETH'),4)
    b31scale = get_scale(c.b3v3_collateral_loan_ratio)
    b31scale.set_index('risk_rating')
    b31data = get_risks(B31v3, c.b3v3_collateral_loan_ratio)
    b31risk_distribution = v3get_distr(b31data)
    risk_distr_temp = b31risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v3changepeg_rec_cycle_otherdebt(B31v3, b31risk_distribution, c.b3v3_collateral_loan_ratio, current_peg, new_peg, 0.01)


def v2plot_b3_liquidation_risk_by_peg(new_peg): 
    ## function to plot SubBin31 liquidation's risk by stETH:ETH rate 
    # params:   new_peg - plot liquidation amount in range [current stETH:ETH rate, new_peg]  
    pl = plot_liquidation_amounts(v2get_b31_risks_for_range_of_peg(new_peg)) 
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v2 Bin 3 Simulation of liquidation risk")
    plt.xlabel("ETH:stETH rate")
    plt.ylabel("stETH amount")
    pl.grid()
    plt.show()
    
def v3plot_b3_liquidation_risk_by_peg(new_peg): 
    ## function to plot SubBin31 liquidation's risk by stETH:ETH rate 
    # params:   new_peg - plot liquidation amount in range [current stETH:ETH rate, new_peg]  
    pl = plot_liquidation_amounts(v3get_b31_risks_for_range_of_peg(new_peg)) 
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v3 Bin 3 Simulation of liquidation risk")
    plt.xlabel("ETH:stETH rate")
    plt.ylabel("wstETH amount")
    pl.grid()
    plt.show()
    

def get_current_market_stats():
    ## function to get current ETH price and Curve pool stats
    
    eth_price = eth_last_price()
    print(f'Current ETH price ${eth_price :,.0f}')   
    
    print('Curve pool current stats:')
    pool=CurveSim.create_steth_pool()
    
    balance=pool.get_balances()
    print(f'ETH balance: {round(balance[0],2):,.0f}')
    print(f'stETH balance: {round(balance[1],2):,.0f}')
    exchange_rate=pool.get_exchange_rate()
    print(f"stETH:ETH: {exchange_rate['stETH_ETH']}")
    print(f"ETH:stETH: {exchange_rate['ETH_stETH']}")           

def v2get_b31_risks_for_changed_price(price_change_prc):
    ## function to estimate SubBin31 risk structure with ETH price's change            
    # params: price_change_prc - % of change price, e.g if price_change_prc=0.05 then price is changed by 5%

    eth_price = eth_last_price()
    return v2changeprice(B31v2,c.b3v2_collateral_loan_ratio,eth_price, price_change_prc) 

def v3get_b31_risks_for_changed_price(price_change_prc):
    ## function to estimate SubBin31 risk structure with ETH price's change            
    # params: price_change_prc - % of change price, e.g if price_change_prc=0.05 then price is changed by 5%

    eth_price = eth_last_price()
    return v3changeprice(B31v3,c.b3v3_collateral_loan_ratio,eth_price, price_change_prc) 


def v2get_b31_risks_for_changed_price_with_current_price(price_change_prc, current_eth_price):
    ## function to estimate SubBin31 risk structure with ETH price's change            
    # params: price_change_prc - % of change price, e.g if price_change_prc=0.05 then price is changed by 5%
    #         current_eth_price - current ETH price in USD

    return v2changeprice(B31v2,c.b3v2_collateral_loan_ratio,current_eth_price, price_change_prc) 

def v3get_b31_risks_for_changed_price_with_current_price(price_change_prc, current_eth_price):
    ## function to estimate SubBin31 risk structure with ETH price's change            
    # params: price_change_prc - % of change price, e.g if price_change_prc=0.05 then price is changed by 5%
    #         current_eth_price - current ETH price in USD

    return v3changeprice(B31v3,c.b3v3_collateral_loan_ratio,current_eth_price, price_change_prc) 


def v2get_b31_risks_for_range_changed_price(max_price_change_prc, step):
    ## function to estimate SubBin31 liquidation's risk structure with ETH price's change in range [current price, current price*max_price_change_prc]           
    # params:   max_price_change_prc - price change in range [current price, current price*max_price_change_prc], e.g. for max_price_change_prc=0.2 [1,200;240]
    #           step - step for price change in range, e.g. $100

    eth_price = eth_last_price()
    b31scale = get_scale(c.b3v2_collateral_loan_ratio)
    b31scale.set_index('risk_rating')
    b31data = get_risks(B31v2, c.b3v2_collateral_loan_ratio)
    b31risk_distribution = v2get_distr(b31data)
    risk_distr_temp = b31risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v2changeprice_cycle(B31v2, b31risk_distribution, c.b3v2_collateral_loan_ratio, eth_price*max_price_change_prc,eth_price, step)

def v3get_b31_risks_for_range_changed_price(max_price_change_prc, step):
    ## function to estimate SubBin31 liquidation's risk structure with ETH price's change in range [current price, current price*max_price_change_prc]           
    # params:   max_price_change_prc - price change in range [current price, current price*max_price_change_prc], e.g. for max_price_change_prc=0.2 [1,200;240]
    #           step - step for price change in range, e.g. $100

    eth_price = eth_last_price()
    b31scale = get_scale(c.b3v3_collateral_loan_ratio)
    b31scale.set_index('risk_rating')
    b31data = get_risks(B31v3, c.b3v3_collateral_loan_ratio)
    b31risk_distribution = v3get_distr(b31data)
    risk_distr_temp = b31risk_distribution.copy()
    risk_distr_temp = risk_distr_temp.iloc[:, 0]
    risk_distr_temp = risk_distr_temp.to_frame()
    
    return v3changeprice_cycle(B31v3, b31risk_distribution, c.b3v3_collateral_loan_ratio, eth_price*max_price_change_prc,eth_price, step)


def v2plot_b3_liquidation_risk_by_price(max_price_change_prc, step): 
    ## function to plot SubBin31 liquidation's risk by peg    
    pl = plot_liquidation_amounts(v2get_b31_risks_for_range_changed_price(max_price_change_prc, step))
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v2 Bin 3 Simulation of liquidation risk")
    plt.xlabel("ETH price")
    plt.ylabel("stETH amount")
    pl.grid()
    plt.show()  

def v3plot_b3_liquidation_risk_by_price(max_price_change_prc, step): 
    ## function to plot SubBin31 liquidation's risk by peg    
    pl = plot_liquidation_amounts(v3get_b31_risks_for_range_changed_price(max_price_change_prc, step))
    plt.rcParams["figure.figsize"] = (15,5)
    plt.title("Aave v3 Bin 3 Simulation of liquidation risk")
    plt.xlabel("ETH price")
    plt.ylabel("wstETH amount")
    pl.grid()
    plt.show()  


def get_aave_riskiest_positions(items_number):
    ## function to get items_number riskiest positions in order by locked stETH amount
    
    v2b1scale = get_scale(c.b1v2_collateral_loan_ratio)
    v2b1scale.set_index('risk_rating')
    v2b1data = get_risks(B1v2, c.b1v2_collateral_loan_ratio)
    v2b1data['debt_in_usd'] = v2b1data['debt']*eth_last_price()
    v2b1data.rename(columns={'debt':'debt_in_eth'}, inplace=True)#columns={'oldName1': 'newName1', 'oldName2': 'newName2'}
    v2b1data['aave_vers'] = '2'
    v2b1data['bin'] = '1'
    
    
    v3b1scale = get_scale(c.b1v3_collateral_loan_ratio)
    v3b1scale.set_index('risk_rating')
    v3b1data = get_risks(B1v3, c.b1v3_collateral_loan_ratio)
    v3b1data['debt_in_eth'] = v3b1data['debt']/eth_last_price()
    v3b1data.rename(columns={'debt':'debt_in_usd'}, inplace=True)
    v3b1data['aave_vers'] = '3'
    v3b1data['bin'] = '1'
    

    
    v2b2scale = get_scale(c.b2v2_collateral_loan_ratio)
    v2b2scale.set_index('risk_rating')
    v2b2data = get_risks(B2v2, c.b2v2_collateral_loan_ratio)
    v2b2data['debt_in_usd'] = v2b2data['debt']*eth_last_price()
    v2b2data.rename(columns={'debt':'debt_in_eth'}, inplace=True)
    v2b2data['aave_vers'] = '2'
    v2b2data['bin'] = '2'

    
    v3b2scale = get_scale(c.b2v3_collateral_loan_ratio)
    v3b2scale.set_index('risk_rating')
    v3b2data = get_risks(B2v3, c.b2v3_collateral_loan_ratio)
    v3b2data['debt_in_eth'] = v3b2data['debt']/eth_last_price()
    v3b2data.rename(columns={'debt':'debt_in_usd'}, inplace=True)
    v3b2data['aave_vers'] = '3'
    v3b2data['bin'] = '2'


    
    v2b3scale = get_scale(c.b3v2_collateral_loan_ratio)
    v2b3scale.set_index('risk_rating')
    v2b3data = get_risks(B3v2, c.b3v2_collateral_loan_ratio)
    v2b3data['debt_in_usd'] = v2b3data['debt']*eth_last_price()
    v2b3data.rename(columns={'debt':'debt_in_eth'}, inplace=True)
    v2b3data['aave_vers'] = '2'
    v2b3data['bin'] = '3'

    
    v3b3scale = get_scale(c.b3v3_collateral_loan_ratio)
    v3b3scale.set_index('risk_rating')
    v3b3data = get_risks(B3v3, c.b3v3_collateral_loan_ratio)
    v3b3data['debt_in_eth'] = v3b3data['debt']/eth_last_price()
    v3b3data.rename(columns={'debt':'debt_in_usd'}, inplace=True)
    v3b3data['aave_vers'] = '3'
    v3b3data['bin'] = '3'


    v2riskiest_b1 = v2b1data[['user','amount', 'healthf', 'risk_rating', 'debt_in_eth','debt_in_usd', 'aave_vers', 'bin']].query('risk_rating ==["C","D"] ')
    v2riskiest_b2 = v2b2data[['user','amount', 'healthf', 'risk_rating', 'debt_in_eth','debt_in_usd', 'aave_vers', 'bin']].query('risk_rating ==["C","D"] ')
    v2riskiest_b3 = v2b3data[['user','amount', 'healthf', 'risk_rating', 'debt_in_eth','debt_in_usd', 'aave_vers', 'bin']].query('risk_rating ==["C","D"] ') 
    
    v3riskiest_b1 = v3b1data[['user','amount', 'healthf', 'risk_rating', 'debt_in_eth','debt_in_usd', 'aave_vers', 'bin']].query('risk_rating ==["C","D"]') 
    v3riskiest_b2 = v3b2data[['user','amount', 'healthf', 'risk_rating', 'debt_in_eth','debt_in_usd', 'aave_vers', 'bin']].query('risk_rating ==["C","D"]')
    v3riskiest_b3 = v3b3data[['user','amount', 'healthf', 'risk_rating', 'debt_in_eth','debt_in_usd', 'aave_vers', 'bin']].query('risk_rating ==["C","D"]')
    
    
    riskiest_positions = pd.concat([v2riskiest_b2, v2riskiest_b3, v2riskiest_b1, v3riskiest_b2, v3riskiest_b3, v3riskiest_b1], axis=0)
    riskiest_positions.rename(columns = {'amount':'(w)steth_amount', 'debt_in_usd':'debt denom in usd', 'debt_in_eth':'debt denom in eth'}, inplace = True)
    return riskiest_positions.sort_values(by = '(w)steth_amount', ascending= False).head(items_number)



def get_peg_and_exchange_amount_to_start_cascade_liq():
    ## function to estimate stETH:ETH rate amount of stETH at exchange which can start cascade liquidation   
    exchange_amount = c.STETH_START_AMOUNT_CASCADE_LIQUDATIONS
    steth_pool = 1000000 
    wsteth_steth_price = get_wsteth_steth_price()

    while exchange_amount < steth_pool:
        liq_amount = 0
        itr = 0
        prev_peg = 0
    
        pool = CurveSim.create_steth_pool()
        balance=pool.get_balances()
        steth_pool = round(balance[1],2)

        pool.exchange_tokens('steth', 'eth', exchange_amount)
        exchange_rate=pool.get_exchange_rate()
        step_peg = exchange_rate['stETH_ETH'] 
        changed_peg = exchange_rate['stETH_ETH'] 

        #pool.info()
        
        while (changed_peg > 0.2) and (itr < 5):
            
            if abs(prev_peg - changed_peg) < 0.01:

                break
            
            b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
                    
            b3v2_change_structure =  v2get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]

            b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
                    
            b3v3_change_structure =  v3get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]

            liq_amount = b1v2_liquidated_amount  + b3v2_liquidated_amount + wsteth_steth_price*b1v3_liquidated_amount + wsteth_steth_price*b3v3_liquidated_amount- liq_amount
            
            if liq_amount < 1:
                liq_amount = 1
            

            pool.exchange_tokens('steth', 'eth', int(liq_amount))
            
            exchange_rate=pool.get_exchange_rate()

            prev_peg = changed_peg 
            changed_peg = exchange_rate['stETH_ETH']
            
            itr = itr + 1 
        if round(changed_peg, 2) <= 0.2:
            break 
        
        exchange_amount += 2*c.STEP_CASCADE_LIQUDATIONS

    return round(step_peg, 4),exchange_amount 

def get_peg_and_exchange_amount_to_start_cascade_liq_with_log():
    ## function to estimate stETH:ETH rate amount of stETH at exchange which can start cascade liquidation 
    ## with short log  
    exchange_amount = c.STETH_START_AMOUNT_CASCADE_LIQUDATIONS
    print(f'\nExchanged amount:{exchange_amount:,.0f}')
    steth_pool = 1000000 
    wsteth_steth_price = get_wsteth_steth_price()

    while exchange_amount < steth_pool:
        liq_amount = 0
        itr = 0
        prev_peg = 0
    
        pool = CurveSim.create_steth_pool()
        balance=pool.get_balances()
        print(f'ETH balance: {round(balance[0],2):,.0f}')
        print(f'stETH balance: {round(balance[1],2):,.0f}')
        steth_pool = round(balance[1],2)

        pool.exchange_tokens('steth', 'eth', exchange_amount)
        exchange_rate=pool.get_exchange_rate()
        step_peg = exchange_rate['stETH_ETH'] 
        changed_peg = exchange_rate['stETH_ETH'] 
        
        print(f'Pool stats after exchange:')
        pool.info()
        
        
        while (changed_peg > 0.2) and (itr < 5):
            
            if abs(prev_peg - changed_peg) < 0.001:
                
                break
            
        
            b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
            print(f'b1v2 cumulative liquidated: {b1v2_liquidated_amount:,.2f} stETH')                
                    
            b3v2_change_structure =  v2get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]
            print(f'b3v2 cumulative liquidated: {b3v2_liquidated_amount:,.2f} stETH')                
                    
            b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
            print(f'b1v3 cumulative liquidated: {b1v3_liquidated_amount:,.2f} wstETH ({wsteth_steth_price*b1v3_liquidated_amount :,.2f} stETH)')                
                    
            b3v3_change_structure =  v3get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
            b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
            print(f'b3v3 cumulative liquidated: {b3v3_liquidated_amount:,.2f} wstETH ({wsteth_steth_price*b3v3_liquidated_amount :,.2f} stETH)')                

            liq_amount = b1v2_liquidated_amount  + b3v2_liquidated_amount + wsteth_steth_price*b1v3_liquidated_amount + wsteth_steth_price*b3v3_liquidated_amount- liq_amount
            print (f'liquidated in iteration: {liq_amount:,.2f} stETH')
            if liq_amount < 1:
                liq_amount = 1
            
            pool.exchange_tokens('steth', 'eth', int(liq_amount))
            
            balance=pool.get_balances()
            print(f'ETH balance: {round(balance[0],2):,.0f}')
            print(f'stETH balance: {round(balance[1],2):,.0f}')
            exchange_rate=pool.get_exchange_rate()            
            print(f"stETH->ETH: {exchange_rate['stETH_ETH']}")
            prev_peg = changed_peg 
            changed_peg = exchange_rate['stETH_ETH']
            
            itr = itr + 1 
            
        if round(changed_peg, 2) <= 0.2:
            
            break 
        
        exchange_amount += 2*c.STEP_CASCADE_LIQUDATIONS
        print(f'\nExchanged amount:{exchange_amount:,.0f}')
        
    return round(step_peg, 4),exchange_amount 



def get_peg_after_exchange_steth_for_eth(exchanged_amount):	
## function to estimate stETH:ETH rate after exchange of "exchanged_amount"stETH for ETH in Curve pool and folowing liquidations

    liq_amount = 0
    itr = 0
    prev_peg = 0
    wsteth_steth_price = get_wsteth_steth_price()
    
    
    pool = CurveSim.create_steth_pool()
    balance=pool.get_balances()
        
    pool.exchange_tokens('steth', 'eth', exchanged_amount)
    exchange_rate=pool.get_exchange_rate()
    changed_peg = exchange_rate['stETH_ETH'] 
        

    while (changed_peg > 0.2) and (itr < 5):
        if abs(prev_peg - changed_peg) < 0.01:
            break
            
        b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
                    
        b3v2_change_structure =  v2get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]

        b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
                    
        b3v3_change_structure =  v3get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
    
        liq_amount = b1v2_liquidated_amount  + b3v2_liquidated_amount + wsteth_steth_price*b1v3_liquidated_amount + wsteth_steth_price*b3v3_liquidated_amount- liq_amount
            
        if liq_amount < 1:
            liq_amount = 1
            
        pool.exchange_tokens('steth', 'eth', int(liq_amount))
        exchange_rate=pool.get_exchange_rate()
    
    
        prev_peg = changed_peg 
        changed_peg = exchange_rate['stETH_ETH']
        itr = itr + 1 
    
    return round(changed_peg, 4)

def get_peg_after_remove_eth_one(removed_amount):	
## function to estimate the stETH:ETH rate after remove of "removed_amount" ETH from Curve pool and following liquidations
    
    liq_amount = 0
    itr = 0
    prev_peg = 0
    wsteth_steth_price = get_wsteth_steth_price()
    
    pool = CurveSim.create_steth_pool()
    pool.remove_liquidity_one_coin(removed_amount*10**18, 0)
    exchange_rate=pool.get_exchange_rate()
    changed_peg = exchange_rate['stETH_ETH'] 

    while (changed_peg > 0.2) and (itr < 5):
        if abs(prev_peg - changed_peg) < 0.01:
            break
            
    
        b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
                    
        b3v2_change_structure =  v2get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]

        b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
                    
        b3v3_change_structure =  v3get_b31_risks_for_range_of_peg(changed_peg).iloc[: , -1].to_frame()
        b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
    
        liq_amount = b1v2_liquidated_amount  + b3v2_liquidated_amount + wsteth_steth_price*b1v3_liquidated_amount + wsteth_steth_price*b3v3_liquidated_amount- liq_amount
        
            
        if liq_amount < 1:
            liq_amount = 1

        pool.exchange_tokens('steth', 'eth', int(liq_amount))
        exchange_rate=pool.get_exchange_rate()
        prev_peg = changed_peg 
        changed_peg = exchange_rate['stETH_ETH']
        itr = itr + 1 
    
    return round(changed_peg, 4)


def get_peg_for_eth_price(input_eth_price):
    ## function to estimate stETH:ETH rate for changed ETH price and return it after possible liquidations  
    ## params: new ETH price ($) 
    
    eth_price = eth_last_price()
    pool=CurveSim.create_steth_pool()
    wsteth_steth_price = get_wsteth_steth_price()
    
    change_percent = round((1 - input_eth_price/eth_price),2)
    
    b21v2_changed_structure = v2get_b21_risks_for_changed_price(change_percent)    
    b21v2_liquidated_amount = b21v2_changed_structure.loc['liquidation'][0] 
        
    b31v2_changed_structure = v2get_b31_risks_for_changed_price(change_percent)   
    b31v2_liquidated_amount = b31v2_changed_structure.loc['liquidation'][0]
        
    b21v3_changed_structure = v3get_b21_risks_for_changed_price(change_percent)    
    b21v3_liquidated_amount = b21v3_changed_structure.loc['liquidation'][0]    
    
    b31v3_changed_structure = v3get_b31_risks_for_changed_price(change_percent)    
    b31v3_liquidated_amount = b31v3_changed_structure.loc['liquidation'][0]
    
    
    b2_b3_liquidated_amount = b21v2_liquidated_amount + b31v2_liquidated_amount + wsteth_steth_price * (b21v3_liquidated_amount + b31v3_liquidated_amount)
        
    pool.exchange_tokens('steth', 'eth', round(b2_b3_liquidated_amount))
    new_peg = pool.get_exchange_rate().get('stETH_ETH')
        
    b1v2_liquidated_amount = 0  
    b3v2_liquidated_amount = 0  
    b1v3_liquidated_amount = 0  
    b3v3_liquidated_amount = 0  
    i = 1
    peg = 0
    b1v2prev_liq_amount = 0
    b3v2prev_liq_amount = 0
    b1v3prev_liq_amount = 0
    b3v3prev_liq_amount = 0

    b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
    b1v2prev_liq_amount = b1v2_liquidated_amount
        
    b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
    b1v3prev_liq_amount = b1v3_liquidated_amount
        
    b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]
    b3v2prev_liq_amount = b3v2_liquidated_amount 
    
    b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
    b3v3prev_liq_amount = b3v3_liquidated_amount 
        
    if round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 1: 
        pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
        new_peg = pool.get_exchange_rate().get('stETH_ETH')
        
        i = i + 1
        
        while  (round(peg,3) != round(new_peg,3)) and round(new_peg,2) > 0.2 and round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 0: 
            
            peg = new_peg
            b1v2_change_structure = v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0] - b1v2prev_liq_amount
                           
            b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0] - b3v2prev_liq_amount
                        
            b1v3_change_structure = v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0] - b1v3prev_liq_amount
                            
            b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0] - b3v3prev_liq_amount
                    
            if round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 0: 
                pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
                new_peg = pool.get_exchange_rate().get('stETH_ETH')
                            
            i = i + 1
            b1v2prev_liq_amount += b1v2_liquidated_amount
            b3v2prev_liq_amount += b3v2_liquidated_amount
            b1v3prev_liq_amount += b1v3_liquidated_amount
            b3v3prev_liq_amount += b3v3_liquidated_amount
            
    return new_peg

def get_peg_for_change_eth_price_with_current_price(new_eth_price, current_eth_price):
    ## function to estimates the risk of liquidations for changed ETH price 
    ## return stETH:ETH rate for changed price and after possible liquidations        
    
    pool=CurveSim.create_steth_pool()
    wsteth_steth_price = get_wsteth_steth_price()
    change_percent = round((1 - new_eth_price/current_eth_price),2)
    
    b21v2_changed_structure = v2changeprice(B21v2,c.b2v2_collateral_loan_ratio,current_eth_price, change_percent)       
    b21v2_liquidated_amount = b21v2_changed_structure.loc['liquidation'][0]
    
    b21v3_changed_structure = v3changeprice(B21v3,c.b2v3_collateral_loan_ratio,current_eth_price, change_percent)       
    b21v3_liquidated_amount = b21v3_changed_structure.loc['liquidation'][0]
    
    b31v2_changed_structure = v2changeprice(B31v2,c.b3v2_collateral_loan_ratio,current_eth_price, change_percent) 
    b31v2_liquidated_amount = b31v2_changed_structure.loc['liquidation'][0]
    
    b31v3_changed_structure = v3changeprice(B31v3,c.b3v3_collateral_loan_ratio,current_eth_price, change_percent) 
    b31v3_liquidated_amount = b31v3_changed_structure.loc['liquidation'][0]
    
    b2_b3_liquidated_amount = b21v2_liquidated_amount + b31v2_liquidated_amount + wsteth_steth_price * (b21v3_liquidated_amount + b31v3_liquidated_amount)
    if b2_b3_liquidated_amount == 0:
        b2_b3_liquidated_amount = 1 
    
    pool.exchange_tokens('steth', 'eth', round(b2_b3_liquidated_amount))
    new_peg = pool.get_exchange_rate().get('stETH_ETH')

    b1v2_liquidated_amount = 0  
    b3v2_liquidated_amount = 0  
    b1v3_liquidated_amount = 0  
    b3v3_liquidated_amount = 0  
    i = 1
    peg = 0
    b1v2prev_liq_amount = 0
    b3v2prev_liq_amount = 0
    b1v3prev_liq_amount = 0
    b3v3prev_liq_amount = 0

    b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
    b1v2prev_liq_amount = b1v2_liquidated_amount
    
    b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
    b1v3prev_liq_amount = b1v3_liquidated_amount
    
    b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]
    b3v2prev_liq_amount = b3v2_liquidated_amount 
    
    b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
    b3v3prev_liq_amount = b3v3_liquidated_amount 
    
    if round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 1: 
        pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
        new_peg = pool.get_exchange_rate().get('stETH_ETH')
        
        i = i + 1
        
        while  (round(peg,3) != round(new_peg,3)) and round(new_peg,2) > 0.2 and round(b1v2_liquidated_amount + b3v2_liquidated_amount  + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount),2) > 0: 
            
            peg = new_peg
            b1v2_change_structure = v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0] - b1v2prev_liq_amount
            
            b1v3_change_structure = v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0] - b1v3prev_liq_amount
                
            b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0] - b3v2prev_liq_amount
            
            b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0] - b3v3prev_liq_amount
            
            if round(b1v2_liquidated_amount + b3v2_liquidated_amount  + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 0: 
                pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount  + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
                new_peg = pool.get_exchange_rate().get('stETH_ETH')
                
            i = i + 1
            b1v2prev_liq_amount += b1v2_liquidated_amount
            b3v2prev_liq_amount += b3v2_liquidated_amount
            b1v3prev_liq_amount += b1v3_liquidated_amount
            b3v3prev_liq_amount += b3v3_liquidated_amount
            
    return new_peg

def get_eth_price_to_start_cascade_liq(): 
    ## function to estimate ETH price to start cascade liquidation   
   peg = 1 
   
   current_price = eth_last_price() 
   
   price = current_price - 100
   
   
   while peg >= 0.2 and price > 0:
      peg = get_peg_for_change_eth_price_with_current_price(price, current_price)
   
      price = price - 100
   
   if price > 0:
    for i in range(int(price) + 100, int(price) +200, 10):
        peg = get_peg_for_change_eth_price_with_current_price(i, current_price)
        if peg > 0.2:
            price = i -10
   
            break  
   else:
       price = 0
   return price 
   
def get_peg_for_swap_and_eth_price(input_eth_price, swap_amount):
    ## function to estimate stETH:ETH rate for changed ETH price and return it after possible liquidations  
    ## params: new ETH price ($) 
    
    eth_price = eth_last_price()
    pool=CurveSim.create_steth_pool()
    wsteth_steth_price = get_wsteth_steth_price()
    
    change_percent = round((1 - input_eth_price/eth_price),2)
    
    b21v2_changed_structure = v2get_b21_risks_for_changed_price(change_percent)    
    b21v2_liquidated_amount = b21v2_changed_structure.loc['liquidation'][0] 
        
    b31v2_changed_structure = v2get_b31_risks_for_changed_price(change_percent)   
    b31v2_liquidated_amount = b31v2_changed_structure.loc['liquidation'][0]
        
    b21v3_changed_structure = v3get_b21_risks_for_changed_price(change_percent)    
    b21v3_liquidated_amount = b21v3_changed_structure.loc['liquidation'][0]    
    
    b31v3_changed_structure = v3get_b31_risks_for_changed_price(change_percent)    
    b31v3_liquidated_amount = b31v3_changed_structure.loc['liquidation'][0]    
    
    b2_b3_liquidated_amount = b21v2_liquidated_amount + b31v2_liquidated_amount + wsteth_steth_price * (b21v3_liquidated_amount + b31v3_liquidated_amount)
    
    pool.exchange_tokens('steth', 'eth', round(b2_b3_liquidated_amount + swap_amount))
    new_peg = pool.get_exchange_rate().get('stETH_ETH')
    
    
    b1v2_liquidated_amount = 0  
    b3v2_liquidated_amount = 0  
    b1v3_liquidated_amount = 0  
    b3v3_liquidated_amount = 0  
    i = 1
    peg = 0
    b1v2prev_liq_amount = 0
    b3v2prev_liq_amount = 0
    b1v3prev_liq_amount = 0
    b3v3prev_liq_amount = 0

    b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
    b1v2prev_liq_amount = b1v2_liquidated_amount

    b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
    b1v3prev_liq_amount = b1v3_liquidated_amount
       
    b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]
    b3v2prev_liq_amount = b3v2_liquidated_amount 
        
    b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
    b3v3prev_liq_amount = b3v3_liquidated_amount 
    
    if round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 1: 
        pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
        new_peg = pool.get_exchange_rate().get('stETH_ETH')
        
        i = i + 1
        
        while  (round(peg,3) != round(new_peg,3)) and round(new_peg,2) > 0.2 and round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 0: 
            
            peg = new_peg
            b1v2_change_structure = v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0] - b1v2prev_liq_amount
               
            b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0] - b3v2prev_liq_amount
            
            b1v3_change_structure = v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0] - b1v3prev_liq_amount
                            
            b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0] - b3v3prev_liq_amount
        
            if round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 0: 
                pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
                new_peg = pool.get_exchange_rate().get('stETH_ETH')
                                
            i = i + 1
            b1v2prev_liq_amount += b1v2_liquidated_amount
            b3v2prev_liq_amount += b3v2_liquidated_amount
            b1v3prev_liq_amount += b1v3_liquidated_amount
            b3v3prev_liq_amount += b3v3_liquidated_amount
            
    return new_peg

def get_peg_for_change_eth_price_and_swap_with_current_price(new_eth_price, current_eth_price, swap_amount):
    ## function to estimates the risk of liquidations for changed ETH price and big swap 
    ## return stETH:ETH rate for changed price and after possible liquidations        
    
    
    pool=CurveSim.create_steth_pool()
    wsteth_steth_price = get_wsteth_steth_price()
    change_percent = round((1 - new_eth_price/current_eth_price),2)
    
    b21v2_changed_structure = v2changeprice(B21v2,c.b2v2_collateral_loan_ratio,current_eth_price, change_percent)       
    b21v2_liquidated_amount = b21v2_changed_structure.loc['liquidation'][0]
    
    b21v3_changed_structure = v3changeprice(B21v3,c.b2v3_collateral_loan_ratio,current_eth_price, change_percent)       
    b21v3_liquidated_amount = b21v3_changed_structure.loc['liquidation'][0]
    
    b31v2_changed_structure = v2changeprice(B31v2,c.b3v2_collateral_loan_ratio,current_eth_price, change_percent) 
    b31v2_liquidated_amount = b31v2_changed_structure.loc['liquidation'][0]
    
    b31v3_changed_structure = v3changeprice(B31v3,c.b3v3_collateral_loan_ratio,current_eth_price, change_percent) 
    b31v3_liquidated_amount = b31v3_changed_structure.loc['liquidation'][0]
    
    b2_b3_liquidated_amount = b21v2_liquidated_amount + b31v2_liquidated_amount + wsteth_steth_price * (b21v3_liquidated_amount + b31v3_liquidated_amount)
    if b2_b3_liquidated_amount == 0:
        b2_b3_liquidated_amount = 1 
    
    pool.exchange_tokens('steth', 'eth', round(b2_b3_liquidated_amount + swap_amount))
    new_peg = pool.get_exchange_rate().get('stETH_ETH')

    b1v2_liquidated_amount = 0  
    b3v2_liquidated_amount = 0  
    b1v3_liquidated_amount = 0  
    b3v3_liquidated_amount = 0  
    i = 1
    peg = 0
    b1v2prev_liq_amount = 0
    b3v2prev_liq_amount = 0
    b1v3prev_liq_amount = 0
    b3v3prev_liq_amount = 0

    b1v2_change_structure =  v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0]
    b1v2prev_liq_amount = b1v2_liquidated_amount
    
    b1v3_change_structure =  v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0]
    b1v3prev_liq_amount = b1v3_liquidated_amount
    
    b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0]
    b3v2prev_liq_amount = b3v2_liquidated_amount 
    
    b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame()
    b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0]
    b3v3prev_liq_amount = b3v3_liquidated_amount 
    
    if round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 1: 
        pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
        new_peg = pool.get_exchange_rate().get('stETH_ETH')
        
        i = i + 1
        
        while  (round(peg,3) != round(new_peg,3)) and round(new_peg,2) > 0.2 and round(b1v2_liquidated_amount + b3v2_liquidated_amount  + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount),2) > 0: 
            
            peg = new_peg
            b1v2_change_structure = v2get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v2_liquidated_amount = b1v2_change_structure.loc['liquidation'][0] - b1v2prev_liq_amount
            
            b1v3_change_structure = v3get_b1_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b1v3_liquidated_amount = b1v3_change_structure.loc['liquidation'][0] - b1v3prev_liq_amount
                
            b3v2_change_structure = v2get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v2_liquidated_amount = b3v2_change_structure.loc['liquidation'][0] - b3v2prev_liq_amount
            
            b3v3_change_structure = v3get_b31_risks_for_range_of_peg(new_peg).iloc[: , -1].to_frame() 
            b3v3_liquidated_amount = b3v3_change_structure.loc['liquidation'][0] - b3v3prev_liq_amount
            
            if round(b1v2_liquidated_amount + b3v2_liquidated_amount  + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)) > 0: 
                pool.exchange_tokens('steth', 'eth', round(b1v2_liquidated_amount + b3v2_liquidated_amount  + wsteth_steth_price * (b1v3_liquidated_amount + b3v3_liquidated_amount)))
                new_peg = pool.get_exchange_rate().get('stETH_ETH')
                
            i = i + 1
            b1v2prev_liq_amount += b1v2_liquidated_amount
            b3v2prev_liq_amount += b3v2_liquidated_amount
            b1v3prev_liq_amount += b1v3_liquidated_amount
            b3v3prev_liq_amount += b3v3_liquidated_amount
            
    return new_peg

@unsync
def get_combinations_eth_price_drop_list_and_swap(buf, swap_amount, current_price, prc_drop_list):
## function to estimate stETH:ETH rate for swap of swap_amount and price drop percent from prc_drop_list
## used for searching possible combinations of ETH price drop + big swap to launch cascade liquidations

    tasks = [(prc,get_peg_for_change_eth_price_and_swap_with_current_price(current_price*(1-0.01*prc), current_price, swap_amount))  for prc in prc_drop_list]
    for prc, task in tasks:
        peg = task
        buf.append({"prc":prc,"swap":swap_amount, "peg": peg})
    return buf            
