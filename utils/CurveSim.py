from itertools import combinations, product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from web3 import Web3
from ast import literal_eval
import time
from utils import constants as c


class pool:

    """
    Python model of Curve pool math.
    """

    def __init__(self, A, D, n, p=None, tokens=None, fee=4 * 10**6, feemul=None, r=None):
        """
        A: Amplification coefficient
        D: Total deposit size
        n: number of currencies; if list, assumes meta-pool
        p: precision
        tokens: # of tokens; if meta-pool, this sets # of basepool tokens
        fee: fee with 10**10 precision (default = .004%)
        feemul: fee multiplier for dynamic fee pools
        r: initial redemption price for RAI-like pools
        """

        if isinstance(n, list):  # is metapool
            self.A = A[0]  # actually A * n ** (n - 1) because it's an invariant
            self.n = n[0]
            self.max_coin = self.n - 1
            if not isinstance(fee, list):
                fee = [fee] * n[0]
            self.fee = fee[0]

            self.basepool = pool(A[1], D[1], n[1], fee=fee[1], tokens=tokens)

            if p:
                self.p = p
                self.basepool.p = p
            else:
                self.p = [10**18] * n[0]
                self.basepool.p = [10**18] * n[1]

            if r:
                self.p[0] = r
                self.r = True
            else:
                self.r = False

            if isinstance(D[0], list):
                self.x = D[0]
            else:
                rates = self.p[:]
                rates[self.max_coin] = self.basepool.get_virtual_price()
                self.x = [D[0] // n[0] * 10**18 // _p for _p in rates]

            self.ismeta = True
            self.n_total = n[0] + n[1] - 1
            self.tokens = self.D()
            self.feemul = feemul

        else:
            self.A = A  # actually A * n ** (n - 1) because it's an invariant
            self.n = n
            self.fee = fee

            if p:
                self.p = p
            else:
                self.p = [10**18] * n

            if isinstance(D, list):
                self.x = D
            else:
                self.x = [D // n * 10**18 // _p for _p in self.p]

            if tokens is None:
                self.tokens = self.D()
            else:
                self.tokens = tokens
            self.feemul = feemul
            self.ismeta = False
            self.r = False
            self.n_total = self.n

    def xp(self):
        return [x * p // 10**18 for x, p in zip(self.x, self.p)]

    def D(self, xp=None):
        """
        D invariant calculation in non-overflowing integer operations
        iteratively
        A * sum(x_i) * n**n + D = A * D * n**n + D**(n+1) / (n**n * prod(x_i))
        Converging solution:
        D[j+1] = (A * n**n * sum(x_i) - D[j]**(n+1) / (n**n prod(x_i))) / (A * n**n - 1)
        """
        Dprev = 0
        if xp is None:
            xp = self.xp()
        S = sum(xp)
        D = S
        Ann = self.A * self.n
        while abs(D - Dprev) > 1:
            D_P = D
            for x in xp:
                D_P = D_P * D // (self.n * x)
            Dprev = D
            D = (Ann * S + D_P * self.n) * D // ((Ann - 1) * D + (self.n + 1) * D_P)

        return D

    def y(self, i, j, x, xp=None):
        """
        Calculate x[j] if one makes x[i] = x
        Done by solving quadratic equation iteratively.
        x_1**2 + x1 * (sum' - (A*n**n - 1) * D / (A * n**n)) = D ** (n+1)/(n ** (2 * n) * prod' * A)
     y   x_1**2 + b*x_1 = c
        x_1 = (x_1**2 + c) / (2*x_1 + b)
        """

        if xp is None:
            xx = self.xp()
        else:
            xx = xp[:]
        D = self.D(xx)
        xx[i] = x  # x is quantity of underlying asset brought to 1e18 precision
        xx = [xx[k] for k in range(self.n) if k != j]
        Ann = self.A * self.n
        c = D
        for y in xx:
            c = c * D // (y * self.n)
        c = c * D // (self.n * Ann)
        b = sum(xx) + D // Ann - D
        y_prev = 0
        y = D
        while abs(y - y_prev) > 1:
            y_prev = y
            y = (y**2 + c) // (2 * y + b)
        return y  # the result is in underlying units too

    def y_underlying(self, i, j, x):
        # For meta-pool
        rates = self.p[:]
        rates[self.max_coin] = self.basepool.get_virtual_price()

        # Use base_i or base_j if they are >= 0
        base_i = i - self.max_coin
        base_j = j - self.max_coin
        meta_i = self.max_coin
        meta_j = self.max_coin
        if base_i < 0:
            meta_i = i
        if base_j < 0:
            meta_j = j

        if base_i < 0 or base_j < 0:  # if i or j not in basepool
            xp = [x * p // 10**18 for x, p in zip(self.x, rates)]

            if base_i >= 0:
                # i is from BasePool
                # At first, get the amount of pool tokens
                dx = x - self.basepool.xp()[base_i]
                base_inputs = [0] * self.basepool.n
                base_inputs[base_i] = dx

                dx = self.basepool.calc_token_amount(base_inputs)
                # Need to convert pool token to "virtual" units using rates
                x = dx * rates[self.max_coin] // 10**18
                # Adding number of pool tokens
                x += xp[self.max_coin]

            y = self.y(meta_i, meta_j, x, xp)

            if base_j >= 0:
                dy = xp[meta_j] - y - 1
                dy_fee = dy * self.fee // 10**10

                # Convert all to real units
                # Works for both pool coins and real coins
                dy = (dy - dy_fee) * 10**18 // rates[meta_j]

                D0 = self.basepool.D()
                D1 = D0 - dy * D0 // self.basepool.tokens
                y = self.y_D(base_j, D1)

        else:
            # If both are from the base pool
            y = self.basepool.y(base_i, base_j, x)

        return y

    def y_D(self, i, _D):
        """
        Calculate x[j] if one makes x[i] = x
        Done by solving quadratic equation iteratively.
        x_1**2 + x1 * (sum' - (A*n**n - 1) * D / (A * n**n)) = D ** (n+1)/(n ** (2 * n) * prod' * A)
        x_1**2 + b*x_1 = c
        x_1 = (x_1**2 + c) / (2*x_1 + b)
        """
        xx = self.xp()
        xx = [xx[k] for k in range(self.n) if k != i]
        S = sum(xx)
        Ann = self.A * self.n
        c = _D
        for y in xx:
            c = c * _D // (y * self.n)
        c = c * _D // (self.n * Ann)
        b = S + _D // Ann
        y_prev = 0
        y = _D
        while abs(y - y_prev) > 1:
            y_prev = y
            y = (y**2 + c) // (2 * y + b - _D)
        return y  # the result is in underlying units too

    def dy(self, i, j, dx):
        if self.ismeta:  # note that fees are already included
            rates = self.p[:]
            rates[self.max_coin] = self.basepool.get_virtual_price()

            # Use base_i or base_j if they are >= 0
            base_i = i - self.max_coin
            base_j = j - self.max_coin
            meta_i = self.max_coin
            meta_j = self.max_coin
            if base_i < 0:
                meta_i = i
            if base_j < 0:
                meta_j = j

            if base_i < 0 or base_j < 0:  # if i or j not in basepool
                xp = [x * p // 10**18 for x, p in zip(self.x, rates)]

                if base_i < 0:
                    x = xp[i] + dx * rates[i] // 10**18
                else:
                    # i is from BasePool
                    # At first, get the amount of pool tokens
                    base_inputs = [0] * self.basepool.n
                    base_inputs[base_i] = dx

                    dx = self.basepool.calc_token_amount(base_inputs)
                    # Need to convert pool token to "virtual" units using rates
                    x = dx * rates[self.max_coin] // 10**18
                    # Adding number of pool tokens
                    x += xp[self.max_coin]

                y = self.y(meta_i, meta_j, x, xp)

                # Either a real coin or token
                dy = xp[meta_j] - y - 1
                dy_fee = dy * self.fee // 10**10

                # Convert all to real units
                # Works for both pool coins and real coins
                dy = (dy - dy_fee) * 10**18 // rates[meta_j]

                if base_j >= 0:
                    dy = self.basepool.calc_withdraw_one_coin(dy, base_j)

            else:
                # If both are from the base pool
                dy = self.basepool.dy(base_i, base_j, dx)
                dy = dy - dy * self.fee // 10**10

            return dy

        else:  # if not meta-pool
            # dx and dy are in underlying units
            xp = self.xp()
            return xp[j] - self.y(i, j, xp[i] + dx)

    def exchange(self, i, j, dx):
        if self.ismeta:  # exchange_underlying
            rates = self.p[:]
            rates[self.max_coin] = self.basepool.get_virtual_price()


            # Use base_i or base_j if they are >= 0
            base_i = i - self.max_coin
            base_j = j - self.max_coin
            meta_i = self.max_coin
            meta_j = self.max_coin
            if base_i < 0:
                meta_i = i
            if base_j < 0:
                meta_j = j

            if base_i < 0 or base_j < 0:  # if i or j not in basepool
                xp = [x * p // 10**18 for x, p in zip(self.x, rates)]

                if base_i < 0:
                    x = xp[i] + dx * rates[i] // 10**18
                    self.x[i] += dx
                else:
                    # i is from BasePool
                    # At first, get the amount of pool tokens
                    base_inputs = [0] * self.basepool.n
                    base_inputs[base_i] = dx
                    # Deposit and measure delta
                    dx = self.basepool.add_liquidity(base_inputs)  # dx is # of minted basepool LP tokens
                    self.x[self.max_coin] += dx
                    # Need to convert pool token to "virtual" units using rates
                    x = dx * rates[self.max_coin] // 10**18
                    print(f'Rates is {rates}')
                    # Adding number of pool tokens
                    x += xp[self.max_coin]

                y = self.y(meta_i, meta_j, x, xp)

                # Either a real coin or token
                dy = xp[meta_j] - y - 1
                dy_fee = dy * self.fee // 10**10

                # Convert all to real units
                # Works for both pool coins and real coins
                dy_nofee = dy * 10**18 // rates[meta_j]
                dy = (dy - dy_fee) * 10**18 // rates[meta_j]
                
                self.x[meta_j] -= dy

                # Withdraw from the base pool if needed
                if base_j >= 0:
                    dy = self.basepool.remove_liquidity_one_coin(dy, base_j)
                    dy_nofee = self.basepool.calc_withdraw_one_coin(dy_nofee, base_j, fee=False)
                    dy_fee = dy_nofee - dy

            else:
                # If both are from the base pool
                dy, dy_fee = self.basepool.exchange(base_i, base_j, dx)

            return dy, dy_fee

        else:  # if not meta-pool, normal exchange

            xp = self.xp()
            x = xp[i] + dx
            y = self.y(i, j, x)
            dy = xp[j] - y
            if self.feemul is None:  # if not dynamic fee pool
                fee = dy * self.fee // 10**10
            else:  # if dynamic fee pool
                fee = (
                    dy
                    * self.dynamic_fee((xp[i] + x) // 2, (xp[j] + y) // 2)
                    // 10**10
                )
            #assert dy > 0
            if dy > 0:
                self.x[i] = x * 10**18 // self.p[i]
                self.x[j] = (y + fee) * 10**18 // self.p[j]
                return dy - fee, fee
            else:
                return 0, 0

    def remove_liquidity_imbalance(self, amounts):
        _fee = self.fee * self.n // (4 * (self.n - 1))

        old_balances = self.x
        new_balances = self.x[:]
        D0 = self.D()
        for i in range(self.n):
            new_balances[i] -= amounts[i]
        self.x = new_balances
        D1 = self.D()
        self.x = old_balances
        fees = [0] * self.n
        for i in range(self.n):
            ideal_balance = D1 * old_balances[i] // D0
            difference = abs(ideal_balance - new_balances[i])
            fees[i] = _fee * difference // 10**10
            new_balances[i] -= fees[i]
        self.x = new_balances
        D2 = self.D()
        self.x = old_balances

        token_amount = (D0 - D2) * self.tokens // D0

        return token_amount

    def calc_withdraw_one_coin(self, token_amount, i, fee=True):
        xp = self.xp()
        if self.fee and fee:
            fee = self.fee - self.fee * xp[i] // sum(xp) + 5 * 10**5
        else:
            fee = 0

        D0 = self.D()
        D1 = D0 - token_amount * D0 // self.tokens
        dy = xp[i] - self.y_D(i, D1)

        return dy - dy * fee // 10**10

    def add_liquidity(self, amounts):
        _fee = self.fee * self.n // (4 * (self.n - 1))

        old_balances = self.x
        new_balances = self.x[:]
        D0 = self.D()

        for i in range(self.n):
            new_balances[i] += amounts[i]
        self.x = new_balances
        D1 = self.D()
        self.x = old_balances

        fees = [0] * self.n
        mint_balances = new_balances[:]
        for i in range(self.n):
            ideal_balance = D1 * old_balances[i] // D0
            difference = abs(ideal_balance - new_balances[i])
            fees[i] = _fee * difference // 10**10
            mint_balances[i] -= fees[i]  # used to calculate mint amount

        self.x = mint_balances
        D2 = self.D()
        self.x = new_balances

        mint_amount = self.tokens * (D2 - D0) // D0
        self.tokens += mint_amount

        return mint_amount

    def remove_liquidity_one_coin(self, token_amount, i):
        dy = self.calc_withdraw_one_coin(token_amount, i)
        self.x[i] -= dy
        self.tokens -= token_amount
        return dy

    def calc_token_amount(self, amounts):
        # Based on add_liquidity (more accurate than calc_token_amount in actual contract)
        _fee = self.fee * self.n // (4 * (self.n - 1))

        old_balances = self.x
        new_balances = self.x[:]
        D0 = self.D()

        for i in range(self.n):
            new_balances[i] += amounts[i]
        self.x = new_balances
        D1 = self.D()
        self.x = old_balances

        fees = [0] * self.n
        mint_balances = new_balances[:]
        for i in range(self.n):
            ideal_balance = D1 * old_balances[i] // D0
            difference = abs(ideal_balance - new_balances[i])
            fees[i] = _fee * difference // 10**10
            mint_balances[i] -= fees[i]  # used to calculate mint amount

        self.x = mint_balances
        D2 = self.D()
        self.x = old_balances

        mint_amount = self.tokens * (D2 - D0) // D0

        return mint_amount

    def get_virtual_price(self):
        return self.D() * 10**18 // self.tokens

    def dynamic_fee(self, xpi, xpj):
        xps2 = xpi + xpj
        xps2 *= xps2  # Doing just ** 2 can overflow apparently
        return (self.feemul * self.fee) // (
            (self.feemul - 10**10) * 4 * xpi * xpj // xps2 + 10**10
        )

    def dydx(self, i, j, dx):
        """
        Returns price, dy[j]/dx[i], given some dx[i]
        """
        dy = self.dy(i, j, dx)
        return dy / dx

    def dydxfee(self, i, j, dx):
        """
        Returns price with fee, (dy[j]-fee)/dx[i]) given some dx[i]
        """
        if self.ismeta:  # fees already included
            dy = self.dy(i, j, dx)
        else:
            if self.feemul is None:  # if not dynamic fee pool
                dy = self.dy(i, j, dx)
                fee = dy * self.fee // 10**10
            else:  # if dynamic fee pool
                xp = self.xp()
                x = xp[i] + dx
                y = self.y(i, j, x)
                dy = xp[j] - y
                fee = (
                    dy
                    * self.dynamic_fee((xp[i] + x) // 2, (xp[j] + y) // 2)
                    // 10**10
                )

            dy = dy - fee
        return dy / dx



    def pricedepth(self, size=0.001):
        """
        Estimates proportion of pool holdings needed to move price by "size"; default = .1%
        """
        combos = list(combinations(range(self.n), 2))
        if self.ismeta:
            ismeta = True
            self.ismeta = False  # pretend a normal pool to exchange for basepool LP token
            p_before = self.p[:]
            self.p[self.max_coin] = self.basepool.get_virtual_price() # use virtual price for LP token precision
        else:
            ismeta = False

        sumxp = sum(self.xp())

        depth = []
        for i, j in combos:
            trade, error, res = self.optarb(
                i, j, self.dydxfee(i, j, 10**12) * (1 - size)
            )
            depth.append(trade[2] / sumxp)

            trade, error, res = self.optarb(
                j, i, self.dydxfee(j, i, 10**12) * (1 - size)
            )
            depth.append(trade[2] / sumxp)

        if ismeta:
            self.p = p_before
            self.ismeta = True

        return depth

    def dotrades(self, trades):
        """
        Does trades formatted as the output of optarbs
        Returns list of trades done in format (i,j,dx[i],dy[j]) and total volume
        """

        if self.ismeta:
            p = self.p[0 : self.max_coin] + self.basepool.p[:]
        else:
            p = self.p[:]

        trades_done = []
        volume = 0
        for trade in trades:
            i = trade[0]
            j = trade[1]
            dx = trade[2]

            dy, dy_fee = self.exchange(i, j, dx)
            trades_done.append((i, j, dx, dy))

            if self.ismeta:
                if i < self.max_coin or j < self.max_coin:  # only count trades involving meta-asset
                    volume += dx * p[i] // 10**18  # in "DAI" units
            else:
                volume += dx * p[i] // 10**18  # in "DAI" units

        return trades_done, volume
        
    def orderbook(self, i, j, width=.1, reso=10**23, show=True):
        
        #if j == 'b', get orderbook against basepool token
        p_mult = 1 
        if j == 'b':
            if i >= self.max_coin:
                raise ValueError("Coin i must be in the metapool for 'b' option")
            self.ismeta = False  # pretend a normal pool to exchange for basepool LP token
            p0 = self.p[:]
            self.p[self.max_coin] = self.basepool.get_virtual_price() # use virtual price for LP token precision
            j = 1
            metaRevert = True
            
            if self.r:
                p_mult = self.p[i]
        else:
            metaRevert = False
            
        #Store initial state
        x0 = self.x[:]
        if self.ismeta:
            x0_base = self.basepool.x[:]
            t0_base = self.basepool.tokens
        
        #Bids
        bids = [(self.dydx(i,j,10**12) * p_mult, 10**12/10**18)] #tuples: price, depth
        size = 0
        
        while bids[-1][0] > bids[0][0]*(1-width):
            size += reso
            self.exchange(i,j,size)
            price = self.dydx(i,j,10**12)
            bids.append((price * p_mult, size/10**18))
            
            #Return to initial state
            self.x = x0[:]
            if self.ismeta:
                self.basepool.x = x0_base[:]
                self.basepool.tokens = t0_base
        
        #Asks     
        asks = [(1/self.dydx(j,i,10**12) * p_mult, 10**12/10**18)] #tuples: price, depth
        size = 0
        
        while asks[-1][0] < asks[0][0]*(1+width):
            size += reso
            dy, fee = self.exchange(j,i,size)
            price = 1/self.dydx(j,i,10**12)
            asks.append((price * p_mult, dy/10**18))

            #Return to initial state
            self.x = x0[:]
            if self.ismeta:
                self.basepool.x = x0_base[:]
                self.basepool.tokens = t0_base 
        
        #Format DataFrames
        bids = pd.DataFrame(bids, columns = ['price', 'depth']).set_index('price') 
        asks = pd.DataFrame(asks, columns = ['price', 'depth']).set_index('price') 
        
        if metaRevert:
            self.p[:] = p0[:]
            self.ismeta = True
        
        if show:
            plt.plot(bids, color='red')
            plt.plot(asks, color='green')
            plt.xlabel('Price')
            plt.ylabel('Depth')
            plt.show()
        return bids, asks
    
    def bcurve(self, xs=None, show=True):
        if self.ismeta:
            combos = [(0,1)]
            labels = ['Metapool Token', 'Basepool LP Token']
            
        else:
            combos = list(combinations(range(self.n),2))
            labels = list(range(self.n))
            labels = ['Coin %s' % str(l) for l in labels]
        
        plt_n = 0
        xs_out = []
        ys_out = []
        for combo in combos:
            i = combo[0]
            j = combo[1]
        
            if xs is None:
                xs_i = np.linspace(int(self.D()*.0001),self.y(j,i,int(self.D()*.0001)), 1000).round()
            else:
                xs_i = xs
                
        
            ys_i = []  
            for x in xs_i:
                ys_i.append(self.y(i,j,int(x))/10**18)
            
            xs_i = xs_i/10**18
            xs_out.append(xs_i)
            ys_out.append(ys_i)
        
            xp = self.xp()[:]
            
            if show: 
                if plt_n == 0:
                    fig, axs = plt.subplots(1, len(combos), constrained_layout=True)
                    
                if len(combos) == 1:
                    ax = axs
                else:
                    ax = axs[plt_n]
                    
                ax.plot(xs_i, ys_i, color='black')
                ax.scatter(xp[i]/10**18, xp[j]/10**18, s=40, color='black')
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[j])
                plt_n += 1
        
        if show:
            plt.show()
            
        return xs_out, ys_out

    def get_peg(self,token_in, token_out, amount):
        '''Return peg for token_in if amount of token_in will be exchanged for token_out, this result will not be saved '''
        numbers = {'eth': 0, 'steth': 1}
        return self.dydxfee(numbers[token_in], numbers[token_out], amount * 10 ** 18)

    def exchange_tokens(self,token_in, token_out, amount):
        '''Perform exchange in the pool, this result will be saved '''
        numbers = {'eth': 0, 'steth': 1}
        return self.exchange(numbers[token_in], numbers[token_out], amount * 10 ** 18)

    def remove_eth(self, amount):
        '''Perform remove eth from the pool, this result will be saved '''
        #remove_liquidity_one_coin(self, amount * 10 ** 18, 0)
        return self.remove_liquidity_one_coin(amount * 10 ** 18, 0)

    def get_exchange_rate(self):
        '''Return exchange rates, i '''
        st_eth_eth = self.dydxfee(1,0,100000)
        eth_st_eth = self.dydxfee(0,1,100000)
        dict={}
        dict['stETH_ETH']=st_eth_eth
        dict['ETH_stETH']=eth_st_eth
        return dict

    def get_balances(self):
        '''Return current balances in pool'''
        return [x / 10 ** 18 for x in self.xp()]

    def info(self):
        balance=self.get_balances()
        print(f'ETH balance: {round(balance[0],2):,.0f}')
        print(f'stETH balance: {round(balance[1],2):,.0f}')
        exchange_rate=self.get_exchange_rate()
        print(f"stETH->ETH: {exchange_rate['stETH_ETH']}")
        print(f"ETH->stETH: {exchange_rate['ETH_stETH']}")

##Functions to get pool data


def pooldata(poolname, csv="utils/poolDF_cg.csv", balanced=False):
    w3 = Web3(Web3.HTTPProvider(c.TOKEN))

    pools = pd.read_csv(
        csv,
        sep=";",
        header=0,
        index_col=0,
        converters={
            "coins": literal_eval,
            "precmul": literal_eval,
            "tokentype": literal_eval,
        },
    )
    p = pools.loc[poolname]

    ABItypes = ["uint256", "int128"]  # some old contracts used int128
    for ABItype in ABItypes:
        abi = (
            '[{"name":"A","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":5227},{"name":"balances","outputs":[{"type":"uint256","name":""}],"inputs":[{"type":"'
            + ABItype
            + '","name":"arg0"}],"stateMutability":"view","type":"function","gas":2250},{"name":"fee","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":2171},{"name":"get_virtual_price","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":1133537},{"name":"coins","outputs":[{"type":"address","name":""}],"inputs":[{"type":"'
            + ABItype
            + '","name":"arg0"}],"stateMutability":"view","type":"function","gas":2310}]'
        )
        contract = w3.eth.contract(address=p.address, abi=abi)
        try:
            contract.functions.balances(0).call()
            break
        except:
            pass

    coins = p.coins

    if p.feemul == "None":
        feemul = None
    else:
        feemul = int(p.feemul)

    if p.precmul[0] == "r":
        # load redemption price data as r
        # update precmul based on redemption price
        r = redemptionprices(1000)
        p.precmul = [r.price[-1] / 10**18]
    else:
        r = None
    
    A = contract.functions.A().call()
    fee = contract.functions.fee().call()

    if p.basepool == "None":  # normal pool

        D = []
        for i in range(len(p.coins)):
            if p.tokentype:  # if any assets are ctokens/ytokens
                if p.tokentype[i]:  # if asset[i] is ctoken/ytoken
                    cAddress = contract.functions.coins(i).call()
                    rate = tokenrate(p.tokentype[i], cAddress)
                else:
                    rate = 10**18
            else:
                rate = 10**18

            D.append(
                contract.functions.balances(i).call()
                * p.precmul[i]
                * rate
                // 10**18
            )
        
        n = len(coins)
        A_base = None
        fee_base = None
        addresses = [p.address.lower()]
        
        pl = pool(A,D,n)
        D_balanced = pl.D()
        tokens = D_balanced * 10**18 // contract.functions.get_virtual_price().call()
        
        if balanced:
            D = D_balanced
        

    else:  # meta-pool
        basepool = pools.loc[p.basepool]
        
        for ABItype in ABItypes:
            abi = (
            '[{"name":"A","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":5227},{"name":"balances","outputs":[{"type":"uint256","name":""}],"inputs":[{"type":"'
            + ABItype
            + '","name":"arg0"}],"stateMutability":"view","type":"function","gas":2250},{"name":"fee","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":2171},{"name":"get_virtual_price","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":1133537},{"name":"coins","outputs":[{"type":"address","name":""}],"inputs":[{"type":"'
            + ABItype
            + '","name":"arg0"}],"stateMutability":"view","type":"function","gas":2310}]'
            )
            base_contract = w3.eth.contract(address=basepool.address, abi=abi)
            try:
                base_contract.functions.balances(0).call()
                break
            except:
                pass


        D = []
        precmul = p.precmul
        precmul.append(base_contract.functions.get_virtual_price().call() / 10**18)
        for i in range(len(p.coins) + 1):
            D.append(int(contract.functions.balances(i).call() * precmul[i]))


        D_base = []
        for i in range(len(basepool.coins)):
            D_base.append(int(base_contract.functions.balances(i).call() * basepool.precmul[i]))

        D = [D, D_base]
        n = [len(p.coins) + 1, len(basepool.coins)]
        coins.extend(basepool.coins)
        A_base = base_contract.functions.A().call()
        fee_base = base_contract.functions.fee().call()
        addresses = [p.address.lower(), basepool.address.lower()]
        
        pl = pool([A, A_base], D, n)
        D_base_balanced = pl.basepool.D()
        tokens = D_base_balanced * 10**18 // base_contract.functions.get_virtual_price().call()
        
        if balanced:
            pl = pool([A, A_base], D, n, tokens = tokens)
            rates = pl.p[:]
            rates[pl.max_coin] = pl.basepool.get_virtual_price()
            if r is not None:
                rates[pl.max_coin - 1] = int(r.price[-1])
            xp = [x * p // 10**18 for x, p in zip(pl.x, rates)]
            D_balanced = pl.D(xp=xp)
            D = [D_balanced, D_base_balanced]

    # Get historical volume
    url = "https://api.thegraph.com/subgraphs/name/convex-community/volume-mainnet"

    histvolume = []
    '''
    for address in addresses:
        query = (
            """{
  swapVolumeSnapshots(
    where: {pool: "%s", period: 86400},
    orderBy: timestamp,
    orderDirection: desc,
    first:60
  ) {
    volume
  }
}"""
            % address
        )
        req = requests.post(url, json={"query": query})
        try:
            volume = pd.DataFrame(req.json()["data"]["swapVolumeSnapshots"], dtype="float").sum()[0]
        except Exception:
            print("[" + poolname + "] No historical volume info from Curve Subgraph.")
            volume = float(input("[" + poolname + "] Please input estimated volume for 2 months: "))
        histvolume.append(volume)

    histvolume = np.array(histvolume)
'''
    # Format output as dict
    data = {
        "D": D,
        "coins": coins,
        "n": n,
        "A": A,
        "A_base": A_base,
        "fee": fee,
        "fee_base": fee_base,
        "tokens": tokens,
        "feemul": feemul,
        "histvolume": histvolume,
        "r": r,
    }

    return data

def pooldata_on_block(poolname, vblock, csv="poolDF_cg.csv", balanced=False):
    w3 = Web3(Web3.HTTPProvider(c.TOKEN))

    pools = pd.read_csv(
        csv,
        sep=";",
        header=0,
        index_col=0,
        converters={
            "coins": literal_eval,
            "precmul": literal_eval,
            "tokentype": literal_eval,
        },
    )
    p = pools.loc[poolname]

    ABItypes = ["uint256", "int128"]  # some old contracts used int128
    for ABItype in ABItypes:
        abi = (
            '[{"name":"A","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":5227},{"name":"balances","outputs":[{"type":"uint256","name":""}],"inputs":[{"type":"'
            + ABItype
            + '","name":"arg0"}],"stateMutability":"view","type":"function","gas":2250},{"name":"fee","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":2171},{"name":"get_virtual_price","outputs":[{"type":"uint256","name":""}],"inputs":[],"stateMutability":"view","type":"function","gas":1133537},{"name":"coins","outputs":[{"type":"address","name":""}],"inputs":[{"type":"'
            + ABItype
            + '","name":"arg0"}],"stateMutability":"view","type":"function","gas":2310}]'
        )
        contract = w3.eth.contract(address=p.address, abi=abi)
        try:
            contract.functions.balances(0).call(block_identifier=int(vblock))
            break
        except:
            pass

    coins = p.coins

    if p.feemul == "None":
        feemul = None
    else:
        feemul = int(p.feemul)

    if p.precmul[0] == "r":
        # load redemption price data as r
        # update precmul based on redemption price
        r = redemptionprices(1000)
        p.precmul = [r.price[-1] / 10**18]
    else:
        r = None
    
    A = contract.functions.A().call(block_identifier=int(vblock))
    fee = contract.functions.fee().call(block_identifier=int(vblock))

    if p.basepool == "None":  # normal pool

        D = []
        for i in range(len(p.coins)):
            if p.tokentype:  # if any assets are ctokens/ytokens
                if p.tokentype[i]:  # if asset[i] is ctoken/ytoken
                    cAddress = contract.functions.coins(i).call(block_identifier=int(vblock))
                    rate = tokenrate(p.tokentype[i], cAddress)
                else:
                    rate = 10**18
            else:
                rate = 10**18

            D.append(
                contract.functions.balances(i).call(block_identifier=int(vblock))
                * p.precmul[i]
                * rate
                // 10**18
            )
        
        n = len(coins)
        A_base = None
        fee_base = None
        addresses = [p.address.lower()]
        
        pl = pool(A,D,n)
        D_balanced = pl.D()
        tokens = D_balanced * 10**18 // contract.functions.get_virtual_price().call(block_identifier=int(vblock))
        
        if balanced:
            D = D_balanced
        

    # Get historical volume
    url = "https://api.thegraph.com/subgraphs/name/convex-community/volume-mainnet"

    histvolume = []
    '''
    for address in addresses:
        query = (
            """{
  swapVolumeSnapshots(
    where: {pool: "%s", period: 86400},
    orderBy: timestamp,
    orderDirection: desc,
    first:60
  ) {
    volume
  }
}"""
            % address
        )
        req = requests.post(url, json={"query": query})
        try:
            volume = pd.DataFrame(req.json()["data"]["swapVolumeSnapshots"], dtype="float").sum()[0]
        except Exception:
            print("[" + poolname + "] No historical volume info from Curve Subgraph.")
            volume = float(input("[" + poolname + "] Please input estimated volume for 2 months: "))
        histvolume.append(volume)

    histvolume = np.array(histvolume)
'''
    # Format output as dict
    data = {
        "D": D,
        "coins": coins,
        "n": n,
        "A": A,
        "A_base": A_base,
        "fee": fee,
        "fee_base": fee_base,
        "tokens": tokens,
        "feemul": feemul,
        "histvolume": histvolume,
        "r": r,
    }

    return data


def tokenrate(tokentype, address):
    w3 = Web3(Web3.HTTPProvider(c.TOKEN))

    if tokentype == "c":
        abi = '[{"constant":true,"inputs":[],"name":"exchangeRateStored","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"}]'
        contract = w3.eth.contract(address=address, abi=abi)
        rate = contract.functions.exchangeRateStored().call()

    elif tokentype == "y":
        abi = '[{"constant":true,"inputs":[],"name":"getPricePerFullShare","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"}]'
        contract = w3.eth.contract(address=address, abi=abi)
        rate = contract.functions.getPricePerFullShare().call()

    return rate


def redemptionprices(n=100):
    url = c.TOKEN
    query = (
        """
{
  redemptionPrices(orderBy: timestamp, orderDirection: desc, first: %d) {
    timestamp
    value
  }
}""" % n)

    r = requests.post(url, json={"query": query})
    data = pd.DataFrame(r.json()["data"]["redemptionPrices"])
    data.columns = ["timestamp", "price"]
    data.price = (data.price.astype(float) * 10**18).astype(int)
    data.timestamp = pd.to_datetime(data.timestamp, unit="s", utc=True)
    data.sort_values("timestamp", inplace=True)
    data.set_index("timestamp", inplace=True)

    return data


def create_steth_pool():
    res = pooldata('steth')
    steth_pool = pool(A=res['A'], D=res['D'], n=res['n'], tokens=res['tokens'], fee=res['fee'])
    #steth_pool = pool(A=30, D=[100000000000000000000000, 100000000000000000000000], n=res['n'], tokens=res['tokens'], fee=res['fee'])

    return steth_pool

def create_steth_pool_a(vA):
    res = pooldata('steth')
    #print(res)
    steth_pool = pool(A=vA, D=res['D'], n=res['n'], tokens=res['tokens'], fee=res['fee'])

    return steth_pool

def create_steth_pool_a_reserves(vA,vD):
    res = pooldata('steth')
    #print(res)
    steth_pool = pool(A=vA, D=vD, n=res['n'], tokens=res['tokens'], fee=res['fee'])

    return steth_pool


def create_steth_pool_block(vblock):
    res = pooldata_on_block('steth', vblock, csv="poolDF_cg.csv", balanced=False)
    steth_pool = pool(A=res['A'], D=res['D'], n=res['n'], tokens=res['tokens'], fee=res['fee'])

    return steth_pool



##Plotters
