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
risk_rating = ['A','B+','B','B-','C','D','liquidation']
b1_collateral_loan_ratio = np.array([1.42, 1.21, 1.14, 1.03, 1.01, 1])
b2_collateral_loan_ratio = np.array([2.5, 1.75, 1.5, 1.25, 1.1, 1])
b3_collateral_loan_ratio = np.array([2.5, 1.75, 1.5, 1.25, 1.1, 1])

@dataclass
class LendingPoolResponse:
    ## response from AAVE Lending Pool contract getUserAccountData endpoint

    collateral_eth: int
    debt_eth: int
    available_borrows_eth: int
    current_liquidation_threshold: int
    ltv: int
    healthfactor: int
    
    
@unsync
def get_user_stats(user: str) -> LendingPoolResponse:
   ## function to parse user stat from AAVE Lending Pool

    address = web3.toChecksumAddress(user)
    result = web3.eth.contract(address=c.AAVE3_LPOOL_CONTRACT, abi=json.load(open("utils/AaveLP_v3.json"))).functions.getUserAccountData(address).call()

    return  LendingPoolResponse(*result)

@unsync
def get_user_emode(user: str):
   ## function to parse user emode from AAVE Lending Pool

    address = web3.toChecksumAddress(user)
    return  web3.eth.contract(address=c.AAVE3_LPOOL_CONTRACT, abi=json.load(open("utils/AaveLP_v3.json"))).functions.getUserEMode(address).call()


@unsync
def get_asteth_balance(vuser):
    #gets current aSTETH balance of user        
    abi = json.load(open("utils/AToken_v3.json"))
    contract = web3.eth.contract(address=c.AWSTETH_CONTRACT, abi=abi)
    address = web3.toChecksumAddress(vuser)
    balance = contract.functions.balanceOf(address).call()   
    
    return balance

@unsync
def get_atoken_balance(user: str, vtoken_contract) -> float:
    ## function to get user's aToken balance

    address = web3.toChecksumAddress(user)
    return web3.eth.contract(address=vtoken_contract, abi=json.load(open("utils/aToken_v3.json"))).functions.balanceOf(address).call()


@unsync
def get_debt_balance(vuser,varcontract, stabcontract):
    ## function to get current debt token balance of user    
    
    if  (varcontract == '-'):        
        var_balance = 0        
    else:
        abi = json.load(open("utils/VariableDebtToken_v3.json"))
        contract = web3.eth.contract(address=Web3.toChecksumAddress(varcontract), abi=abi)
        var_balance = contract.functions.balanceOf(Web3.toChecksumAddress(vuser.lower())).call()   

    #get current debt token balance of user    
    if  (stabcontract == '-'):    
        stable_balance = 0  
        
    else:
        abi = json.load(open("utils/StableDebtToken_v3.json"))
        contract = web3.eth.contract(address=Web3.toChecksumAddress(stabcontract), abi=abi)
        stable_balance = contract.functions.balanceOf(Web3.toChecksumAddress(vuser.lower())).call()   

    balance = var_balance + stable_balance
    return balance

@unsync
def parse_stats(df):
    ## function to parse user's stats
        buf = []
        tasks = [(user, get_user_stats(user)) for user in df["user"]]
        for user, task in tasks:
            stat: LendingPoolResponse = task.result()  # type: ignore
            buf.append(
                {
                    "user": user,
                    "collateral": stat.collateral_eth,
                    "debt": stat.debt_eth,
                    "available_borrow": stat.available_borrows_eth,
                    "threshold": stat.current_liquidation_threshold,
                    "ltv": stat.ltv,
                    "healthf": stat.healthfactor
             
                }
            )
        return buf

@unsync
def parse_emode(df):
    ## function to parse user's balance
        buf = []
        tasks = [(user, get_user_emode(user)) for user in df["user"]]
        for user, task in tasks:
            emode: float = task.result()  # type: ignore
            buf.append({"user": user, "emode": emode})
        return buf

@unsync
def parse_balance(df):
    ## function to parse user's balance
        buf = []
        tasks = [(user, get_asteth_balance(user)) for user in df["user"]]
        for user, task in tasks:
            balance: float = task.result()  # type: ignore
            buf.append({"user": user, "amount": balance})
        return buf

@unsync
def parse_atoken_balance(df, vtoken_symbol, vtoken_contract):
    ## function to parse user's atoken balance
        buf = []
        tasks = [(user, get_atoken_balance(user, vtoken_contract)) for user in df["user"]]
        for user, task in tasks:
            balance: float = task.result()  # type: ignore
            buf.append({"user": user, f'{vtoken_symbol}_collateral': balance})
        return buf


@unsync
def parse_ethdebt_balance(df):        
    ## function to parse user's eth debt 
        buf = []
        tasks = [(user, get_debt_balance(user,'0xeA51d7853EEFb32b6ee06b1C12E6dcCA88Be0fFE', '0x102633152313C81cD80419b6EcF66d14Ad68949A')) for user in df["user"]]
        for user, task in tasks:
            balance: float = task.result()  # type: ignore
            buf.append({"user": user, "ethdebt": balance})
        return buf    

@unsync
def parse_debt_balance(df, coinsymb, coincontract, varcontract, stabcontract, decimals, threshold, ethreshold ):
    ## function to parse user's debt
        price = get_asset_usd_price(coincontract) 
        buf = []
        tasks = [(user, get_debt_balance(user,varcontract, stabcontract)) for user in df["user"]]
        for user, task in tasks:
            balance: float = task.result()  # type: ignore
            buf.append({"user": user, f'{coinsymb}': balance, f'{coinsymb}_price': price, f'{coinsymb}_decimals': decimals, f'{coinsymb}_threshold': threshold, f'{coinsymb}_ethreshold': ethreshold})
        return buf

def get_dune_data(query_id,dune_key):
    query_link=f'https://api.dune.com/api/v1/query/{query_id}/execute'
    headers = {'x-dune-api-key': dune_key}
    data = '{}'
    query_execution_link= requests.post(query_link, headers=headers, data=data)
    query_execution_id=query_execution_link.json()['execution_id']
    status_link=f'https://api.dune.com/api/v1/execution/{query_execution_id}/status'
    status_query = requests.get(status_link, headers=headers, data=data)
    while status_query.json()['state'] != 'QUERY_STATE_COMPLETED' :
        time.sleep(10)
        status_query = requests.get(status_link, headers=headers, data=data)
        
       
    result_link=f'https://api.dune.com/api/v1/execution/{query_execution_id}/results'
    result_query = requests.get(result_link, headers=headers, data=data)
    results_df=pd.DataFrame(result_query.json()['result']['rows'])
    results_df=results_df.rename(columns = {'amount':'amount', 'address':'user','ethdebt':'ethdebt'})
    return results_df



def get_userlist(vlink):    
    ## collect information about users that interacted with aSTETH contract, vlink - Flipside API link
    #r = requests.get(f"{vlink}").json()
    df = get_dune_data(vlink,c.DUNE_KEY)
    #df = pd.DataFrame(r)
    df = df.drop_duplicates(subset=['user'])
    return df

def get_wsteth_usd_price():
    ## function to get current ETH price from Aave Oracle        
    abi = json.load(open("utils/AaveOracle_v3.json"))
    contract = web3.eth.contract(address=c.AAVE3_ORACLE_CONTRACT, abi=abi)
    price = contract.functions.getAssetPrice(Web3.toChecksumAddress('0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0')).call()
    price = price / pow(10,8)
    
    return price

def get_wsteth_steth_price():
    ## function to get current wstETH:stETH price 
    abi = json.load(open("utils/WSTETH.json"))
    contract = web3.eth.contract(address= Web3.toChecksumAddress(c.WSTETH_CONTRACT), abi=abi)
    price = contract.functions.getStETHByWstETH(1000000000000000000).call()
    price = price / pow(10,18)
    
    return price

def get_asset_usd_price(vtoken_contract):
    ## function to get current ETH price from Aave Oracle        
    abi = json.load(open("utils/AaveOracle_v3.json"))
    contract = web3.eth.contract(address=c.AAVE3_ORACLE_CONTRACT, abi=abi)
    price = contract.functions.getAssetPrice(Web3.toChecksumAddress(vtoken_contract.lower())).call()
        
    return price

def get_data():
    tmp_df = get_userlist(2610823) #query wstETH on Aave V3 users
    tmp_df.set_index("user")
    del tmp_df["amount"]
    del tmp_df["ethdebt"]     

    tasks = [parse_stats(tmp_df), parse_balance(tmp_df), parse_ethdebt_balance(tmp_df), parse_emode(tmp_df)]
    parts = [pd.DataFrame(task.result()) for task in tasks]  

    for part in parts:
        tmp_df = tmp_df.merge(part, on="user", how="left")

    collateral_tokens_lst = tuple(zip( c.CBETH3, c.RETH3)) # LST
    for q in range(0, len(collateral_tokens_lst[0])):
        tasks = [parse_atoken_balance(tmp_df, collateral_tokens_lst[0][q],collateral_tokens_lst[7][q])]  
        parts = [pd.DataFrame(task.result()) for task in tasks]
        for part in parts:
            tmp_df = tmp_df.merge(part, on="user", how="left")
    
     
    tmp_df['reth_price'] = get_asset_usd_price(c.RETH3[1])/pow(10,8)
    tmp_df['cbeth_price'] = get_asset_usd_price(c.CBETH3[1])/pow(10,8)
    tmp_df['eth_price'] = get_asset_usd_price(c.WETH3[1])/pow(10,8)
    tmp_df['steth_price'] = get_wsteth_usd_price()  
    
    tmp_df['collateral_reth_calc'] = (tmp_df.reth_collateral/pow(10,18))*tmp_df.reth_price
    tmp_df['collateral_cbeth_calc'] = (tmp_df.cbeth_collateral/pow(10,18))*tmp_df.cbeth_price
    tmp_df['collateral_steth_calc'] = (tmp_df.amount/pow(10,18))*get_wsteth_usd_price()    
    tmp_df['ethdebt_calc'] = (tmp_df.ethdebt/pow(10,18))*tmp_df.eth_price
    

    return tmp_df    

def prepare_data(df):
    ## function to prepare data
    df['healthf'] = df['healthf']/pow(10,18) 
    df['collateral'] = df['collateral']/pow(10,8)    
    df['debt'] = df['debt']/pow(10,8)
    df['ethdebt'] = df['ethdebt']/pow(10,18)
    df['amount'] = df['amount']/pow(10,18)
    
    df['reth_collateral'] = df['reth_collateral']/pow(10,18)
    df['cbeth_collateral'] = df['cbeth_collateral']/pow(10,18)


    #Query only users with collateral and debt 
    df = df.query('collateral > 0 and debt > 0')

    #Diff_collateral: difference between total collateral of user and stETH collateral (aSTETH) / total collateral, needed for bins distribution"""
    df['diff_collateral'] = abs(df['collateral']-df['amount']*get_wsteth_usd_price()- df['collateral_reth_calc'] - df['collateral_cbeth_calc'])/df['collateral']
    
    #Diff_collateral: difference between total debt of user and ETH debt / total debt, needed for bins distribution"""
    df['diff_debt'] = abs(eth_last_price()*df['ethdebt']-df['debt'])/df['debt']


    df.to_csv('data/steth_aave_v3.csv')
    return df

def get_subbin2_stablecoin_debt(df):
    ## functon to get Bin2: users with 80% and more of debt not ETH and subBin21 with >=80% of debt in stablecoins
    
    b2 = df.copy()
    b2 = b2.query('diff_debt > 0.8').sort_values(by = (['amount']), ascending = False)
    b2 = b2.fillna(0)
    
    
    #calc debt in stablecoins
    for q in range(0, len(c.debt_tokens_v3[0])):
        tasks = [parse_debt_balance(b2, c.debt_tokens_v3[0][q], c.debt_tokens_v3[1][q].lower(), c.debt_tokens_v3[2][q].lower(), c.debt_tokens_v3[3][q].lower(), c.debt_tokens_v3[4][q], c.debt_tokens_v3[6][q], c.debt_tokens_v3[8][q])]
        parts = [pd.DataFrame(task.result()) for task in tasks]
        for part in parts:
            b2 = b2.merge(part, on="user", how="left")

    
    b2.dai = b2['dai'].astype(float, errors = 'raise')
    b2.usdc = b2['usdc'].astype(float, errors = 'raise')    
    b2.usdt = b2['usdt'].astype(float, errors = 'raise')  
    b2.lusd = b2['lusd'].astype(float, errors = 'raise')    
    
    
    
    b2['usdc_debt'] = (b2.usdc_price/pow(10,8))*b2.usdc/pow(10,b2.usdc_decimals)  
    b2['usdt_debt'] = (b2.usdt_price/pow(10,8))*b2.usdt/pow(10,b2.usdt_decimals)              
    b2['dai_debt'] = (b2.dai_price/pow(10,8))*b2.dai/pow(10,b2.dai_decimals) 
    b2['lusd_debt'] = (b2.lusd_price/pow(10,8))*b2.lusd/pow(10,b2.lusd_decimals)
    

    
    b2['debt_stable_calc'] = b2['usdc_debt'] + b2['usdt_debt'] + b2['dai_debt'] + b2['lusd_debt']
    b2['diff_debt_stable'] = abs(b2.debt - b2.debt_stable_calc)/b2.debt    
    
    
    #calc collateral in stablecoins
    collateral_tokens_b2 = tuple(zip( c.USDC3,  c.DAI3)) #USDT,LUSD,-can not be collateral
    for q in range(0, len(collateral_tokens_b2[0])):
        tasks = [parse_atoken_balance(b2, collateral_tokens_b2[0][q],collateral_tokens_b2[7][q])]  
        parts = [pd.DataFrame(task.result()) for task in tasks]
        for part in parts:
            b2 = b2.merge(part, on="user", how="left")
    
    b2['collateral_stable_calc'] = (b2.usdc_collateral/pow(10,b2.usdc_decimals))*b2.usdc_price/pow(10,8) + (b2.dai_collateral/pow(10,b2.dai_decimals))*b2.dai_price/pow(10,8)
    
    b21_debf_stables = b2.query('diff_debt_stable <= 0.2')   
    b21_debf_stables.to_csv("data/b21_debt_stables_v3.csv") 
    b2.to_csv("data/b2_v3.csv")
    
    return b21_debf_stables

def get_bin1_eth_debt(df):
    ## function to get Bin1: Aave users with >=80% collaterals - stETH and >=80% debt - ETH and emode
    b1 = df.copy()
    
    b1 = b1.query('diff_collateral <= 0.2 and diff_debt <= 0.2 and emode == 1') #and ethdebt > 0
    b1 = b1.fillna(0)
    
    b1.to_csv("data/b1_v3.csv")
    
    return b1

def get_bin12_eth_debt(df):
    ## function to get Bin1: Aave users with >=80% collaterals - stETH and >=80% debt - ETH and emode
    b12 = df.copy()
    
    b12 = b12.query('diff_collateral <= 0.2 and diff_debt <= 0.2 and emode != 1') #and ethdebt > 0
    b12 = b12.fillna(0)
    
    b12.to_csv("data/b12_v3.csv")
    
    return b12

def get_subbin3_stables_wbtc_eth_80(df, b1, b12, b2):
    ## functon to get subBin3: users with >=80% of collateral in stETH, ETH, WBTC and stablecoins
    ## and >=80% of debt in stablecoins & WBTC
    
    b3_users = set(set(set(df.user).symmetric_difference(b1.user)).symmetric_difference(b12.user)).symmetric_difference(b2.user)  
    b3 =  df.query('user in @b3_users')
    
    
    #calc debt in stablecoins & WBTC & ETH and LST
    debt_tokens_b3 = tuple(zip( c.USDC3, c.USDT3, c.DAI3, c.LUSD3, c.WBTC3)) #, RETH, CBETH
    for q in range(0, len(debt_tokens_b3[0])):
        tasks = [parse_debt_balance(b3, debt_tokens_b3[0][q], debt_tokens_b3[1][q].lower(), debt_tokens_b3[2][q].lower(), debt_tokens_b3[3][q].lower(), debt_tokens_b3[4][q], debt_tokens_b3[6][q], debt_tokens_b3[8][q])]
        
        parts = [pd.DataFrame(task.result()) for task in tasks]
        for part in parts:
            b3 = b3.merge(part, on="user", how="left")
    
    
    b3.dai = b3['dai'].astype(float, errors = 'raise')
    b3.usdc = b3['usdc'].astype(float, errors = 'raise')    
    b3.usdt = b3['usdt'].astype(float, errors = 'raise')  
    b3.lusd = b3['lusd'].astype(float, errors = 'raise')    
    b3.wbtc = b3['wbtc'].astype(float, errors = 'raise')       
    
    
    b3['usdc_debt'] = (b3.usdc_price/pow(10,8))*b3.usdc/pow(10,b3.usdc_decimals)  
    b3['usdt_debt'] = (b3.usdt_price/pow(10,8))*b3.usdt/pow(10,b3.usdt_decimals)              
    b3['dai_debt'] = (b3.dai_price/pow(10,8))*b3.dai/pow(10,b3.dai_decimals) 
    b3['lusd_debt'] = (b3.lusd_price/pow(10,8))*b3.lusd/pow(10,b3.lusd_decimals)
    b3['wbtc_debt'] = (b3.wbtc_price/pow(10,8))*b3.wbtc/pow(10,b3.wbtc_decimals)
            
    b3['debt_stable_calc'] = b3['usdc_debt'] + b3['usdt_debt'] + b3['dai_debt'] + b3['lusd_debt']
    
    b3['debt_wbtc_calc'] = b3['wbtc_debt']
       
    
    #calc collateral in stablecoins & WBTC & 
    collateral_tokens_b3 = tuple(zip( c.USDC3,  c.DAI3,  c.WBTC3,c.WETH3)) #USDT,LUSD,-can not be collateral
    for q in range(0, len(collateral_tokens_b3[0])):
        tasks = [parse_atoken_balance(b3, collateral_tokens_b3[0][q],collateral_tokens_b3[7][q])]  
        parts = [pd.DataFrame(task.result()) for task in tasks]
        for part in parts:
            b3 = b3.merge(part, on="user", how="left")
    
    b3['collateral_stable_calc'] = (b3.usdc_collateral/pow(10,b3.usdc_decimals))*b3.usdc_price/pow(10,8) + (b3.dai_collateral/pow(10,b3.dai_decimals))*b3.dai_price/pow(10,8) 
    
    b3['collateral_wbtc_calc'] = (b3.wbtc_collateral/pow(10,b3.wbtc_decimals))*b3.wbtc_price/pow(10,8)
    b3['collateral_eth_calc'] = (b3.weth_collateral/pow(10,18))*get_asset_usd_price(c.WETH3[1])/pow(10,8)
    
    #How many ETH,rETH, cbETH,  Stablecoins and WBTC in debt of positions
    b3['eth_wbtc_stablec_debt_diff'] = 1-(b3['ethdebt']*get_asset_usd_price(c.WETH3[1])/pow(10,8) + b3['debt_stable_calc'] + b3['debt_wbtc_calc'])/b3['debt']
    
    #How many stETH, ETH, Stablecoins and WBTC in collateral of positions
    b3['eth_steth_wbtc_stablec_collateral_diff'] = 1-(b3['collateral_eth_calc'] + b3['collateral_stable_calc'] + b3['collateral_wbtc_calc'] 
                                                      + b3['collateral_steth_calc'] + b3['collateral_reth_calc'] + b3['collateral_cbeth_calc'])/b3['collateral']   
    
    b3.query('eth_wbtc_stablec_debt_diff <= 0.2 and eth_steth_wbtc_stablec_collateral_diff <= 0.2').to_csv("data/b31_eth_wbtc_stables_v3.csv") 
    b3.to_csv("data/b3_v3.csv")
    return b3.query('eth_wbtc_stablec_debt_diff <= 0.2 and eth_steth_wbtc_stablec_collateral_diff <= 0.2')
    
    
def eth_last_price():
    ## function to get current ETH price from coingecko            
    req = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd&include_market_cap=false&include_24hr_vol=false&include_24hr_change=false&include_last_updated_at=false'
                      ).json()
    price = req['ethereum']['usd']
    
    
    return price

