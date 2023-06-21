# aave-liquidation-model
(w)stETH-collateral liquidations & Curve LP: worst-case scenario model that can be use for understanding what can happen with (w)stETH positions on Aave v2/v3 and with Curve stETH:ETH liquidity pool in worst-case scenario if drops ETH price or ETH:stETH rate or both of them.  

```extractor_v2_v3_.ipynb```:  extracts stats for (w)stETH positions on Aave v2 and Aave v3, divides (w)stETH positions into three bins, load data to csv files. Work with the model should start with this file.

```model_v2_v3_.ipynb```: contains examples of function's calls from ```model23.py```

```model23.py```: contens all calculations of liquidation's risks and simulates of what can happen with Curve LP in worst-case scenario (with start of cascade liquidations on Aave) 

Curve LP simulation (CurveSim) taken from [here](https://github.com/curveresearch/curvesim), folder Utils contens ABI of Aave contracts and ```model23.py```

Available functions (```model23.py```):

```get_current_market_stats()```  function to get current ETH price, wstETH:stETH rate and Curve pool stats

```get_peg_after_exchange_steth_for_eth(exchanged_amount)```  function to estimate stETH:ETH rate after exchange of ```exchanged_amount``` stETH for ETH in Curve pool and folowing liquidations


```get_peg_after_remove_eth_one(removed_amount)```  function to estimate the stETH:ETH rate after remove of ```removed_amount``` ETH from Curve pool and following liquidations


```get_peg_and_exchange_amount_to_start_cascade_liq()``` function to estimate stETH:ETH rate and amount of stETH at exchange which could start cascade liquidations  


```get_peg_for_eth_price(input_eth_price)```  function to estimate stETH:ETH rate for changed ETH price and following liquidations

```get_eth_price_to_start_cascade_liq()```  function to estimate ETH price when could start cascade liquidations  

```get_aave_riskiest_positions(items_number)``` function to get ```items_number``` riskiest positions in order by locked (w)stETH amount

```get_combinations_eth_price_drop_list_and_swap(buf, swap_amount, current_price, prc_drop_list)``` function to estimate stETH:ETH rate after swap of ```swap_amount``` and price drop by percent from ```prc_drop_list```,it is used for searching possible combinations of ETH price drop + big swap to launch cascade liquidations


Aave v2 Ethereum stETH market related functions:

```v2get_b1_current_risks()```  function to get current risk structure in Bin1 (v2)

```v2get_b21_current_risks()``` function to get current risk structure in SubBin21 (v2)

```v2get_b31_current_risks()``` function to get current risk structure in SubBin31 (v2)

```v2get_b1_risks_for_range_of_peg(new_peg)```  function to estimate Bin1 (v2) liquidation's risk structure with change in stETH:ETH rate in range [current peg, new_peg] (snapshot)

```v2get_b31_risks_for_range_of_peg(new_peg)```  function to estimate SubBin31 (v2) liquidation's risk structure with change in stETH:ETH rate in range [current peg, new_peg] (snapshot)

```v2plot_b1_liquidation_risk_by_peg(new_peg)```  function to plot Bin1 (v2) liquidation's risk by stETH:ETH rate change in range [current peg, new_peg] 

```v2plot_b3_liquidation_risk_by_peg(new_peg)```  function to plot SubBin31 (v2) liquidation's risk by stETH:ETH rate change in range [current peg, new_peg] 

```v2get_b21_risks_for_changed_price(price_change_prc)``` function to estimate SubBin21 (v2) risk structure with change in ETH price by the specified percentage 

```v2get_b31_risks_for_changed_price(price_change_prc)``` function to estimate SubBin31 (v2) risk structure with change in ETH price by the specified percentage 

```v2get_b21_risks_for_range_changed_price(max_price_change_prc, step)``` function to estimate SubBin21 (v2) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```          

```v2get_b31_risks_for_range_changed_price(max_price_change_prc, step)``` function to estimate SubBin31 (v2) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```          

```v2plot_b2_liquidation_risk_by_price(max_price_change_prc, step)``` function to plot SubBin21 (v2) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```  

```v2plot_b3_liquidation_risk_by_price(max_price_change_prc, step)``` function to plot SubBin31 (v2) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```  

Aave v3 Ethereum wstETH market related functions:

```v3get_b1_current_risks()```  function to get current risk structure in Bin1 (v3)

```v3get_b21_current_risks()``` function to get current risk structure in SubBin21 (v3)

```v3get_b31_current_risks()``` function to get current risk structure in SubBin31 (v3)

```v3get_b1_risks_for_range_of_peg(new_peg)```  function to estimate Bin1 (v3) liquidation's risk structure with change in stETH:ETH rate in range [current peg, new_peg] (snapshot)

```v3get_b31_risks_for_range_of_peg(new_peg)```  function to estimate SubBin31 (v3) liquidation's risk structure with change in stETH:ETH rate in range [current peg, new_peg] (snapshot)

```v3plot_b1_liquidation_risk_by_peg(new_peg)```  function to plot Bin1 (v3) liquidation's risk by stETH:ETH rate change in range [current peg, new_peg] 

```v3plot_b3_liquidation_risk_by_peg(new_peg)```  function to plot SubBin31 (v3) liquidation's risk by stETH:ETH rate change in range [current peg, new_peg] 

```v3get_b21_risks_for_changed_price(price_change_prc)``` function to estimate SubBin21 (v3) risk structure with change in ETH price by the specified percentage 

```v3get_b31_risks_for_changed_price(price_change_prc)``` function to estimate SubBin31 (v3) risk structure with change in ETH price by the specified percentage 

```v3get_b21_risks_for_range_changed_price(max_price_change_prc, step)``` function to estimate SubBin21 (v3) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```          

```v3get_b31_risks_for_range_changed_price(max_price_change_prc, step)``` function to estimate SubBin31 (v3) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```          

```v3plot_b2_liquidation_risk_by_price(max_price_change_prc, step)``` function to plot SubBin21 (v3) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```  

```v3plot_b3_liquidation_risk_by_price(max_price_change_prc, step)``` function to plot SubBin31 (v3) liquidation's risk structure with change in ETH price in range [current price, current price*max_price_change_prc] with specified ```step```  