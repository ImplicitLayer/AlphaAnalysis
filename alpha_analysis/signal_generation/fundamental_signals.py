import numpy as np


class FundamentalSignals:
    def __init__(self, data):
        """
        A class for generating fundamental trading signals.

        :param data: DataFrame with columns. Example :['Ticker', 'Date', 'EPS', 'Price', 'Book Value',
                                                      'Net Income', 'Total Assets', 'Total Debt',
                                                      'Total Equity', 'Dividends']
        """
        self.data = data.copy()

    def pe_ratio(self):
        """ Calculates P/E (Price-to-Earnings) and generates a signal. """
        self.data['P/E'] = self.data['Price'] / self.data['EPS']
        self.data['P/E_Signal'] = np.where(self.data['P/E'] < 15, 1, np.where(self.data['P/E'] > 25, -1, 0))
        return self.data[['Ticker', 'Date', 'P/E_Signal']]

    def pb_ratio(self):
        """ Calculates P/B (Price-to-Book) and generates a signal. """
        self.data['P/B'] = self.data['Price'] / self.data['Book Value']
        self.data['P/B_Signal'] = np.where(self.data['P/B'] < 1, 1, np.where(self.data['P/B'] > 3, -1, 0))
        return self.data[['Ticker', 'Date', 'P/B_Signal']]

    def roe(self):
        """ Calculates ROE (Return on Equity) and generates a signal. """
        self.data['ROE'] = self.data['Net Income'] / self.data['Total Equity']
        self.data['ROE_Signal'] = np.where(self.data['ROE'] > 0.15, 1, np.where(self.data['ROE'] < 0.05, -1, 0))
        return self.data[['Ticker', 'Date', 'ROE_Signal']]

    def roa(self):
        """ Calculates ROA (Return on Assets) and generates a signal. """
        self.data['ROA'] = self.data['Net Income'] / self.data['Total Assets']
        self.data['ROA_Signal'] = np.where(self.data['ROA'] > 0.05, 1, np.where(self.data['ROA'] < 0.01, -1, 0))
        return self.data[['Ticker', 'Date', 'ROA_Signal']]

    def debt_to_equity(self):
        """ Calculates Debt-to-Equity and generates a signal. """
        self.data['Debt/Equity'] = self.data['Total Debt'] / (self.data['Total Equity'] + 1e-10)
        self.data['Debt/Equity_Signal'] = np.where(self.data['Debt/Equity'] < 1, 1,
                                                   np.where(self.data['Debt/Equity'] > 2.5, -1, 0))
        return self.data[['Ticker', 'Date', 'Debt/Equity_Signal']]

    def dividend_yield(self):
        """ Calculates Dividend Yield and generates a signal. """
        self.data['Dividend Yield'] = self.data['Dividends'] / self.data['Price']
        self.data['Dividend_Signal'] = np.where(self.data['Dividend Yield'] > 0.04, 1,
                                                np.where(self.data['Dividend Yield'] < 0.02, -1, 0))
        return self.data[['Ticker', 'Date', 'Dividend_Signal']]

    def generate_signals(self):
        """ Combines all fundamental signals into a single DataFrame. """
        pe = self.pe_ratio()
        pb = self.pb_ratio()
        roe = self.roe()
        roa = self.roa()
        debt_eq = self.debt_to_equity()
        dividend = self.dividend_yield()

        signals = pe.merge(pb, on=['Ticker', 'Date']).merge(roe, on=['Ticker', 'Date']) \
            .merge(roa, on=['Ticker', 'Date']).merge(debt_eq, on=['Ticker', 'Date']) \
            .merge(dividend, on=['Ticker', 'Date'])

        return signals
