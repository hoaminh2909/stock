import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

import google.generativeai as genai
from datetime import datetime
from datetime import timedelta

st.set_page_config(layout="wide")
st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)

try:
    excel = pd.read_excel('LIST-VN.xlsx')
    VN_stock_list = excel['Mã VN']
    # Tạo hộp chọn trong sidebar để chọn mã cổ phiếu

    ticker = None
    tick_col, op_tick_col, tcol1, tcol2, tcol3 = st.columns(5)
    with tick_col:
        tick = st.text_input('Ticker', value=None, placeholder="Type here")
    with op_tick_col:
        option_tick = st.selectbox("Vietnamese Ticker", VN_stock_list, index=None, placeholder="Type here")

    if not tick and not option_tick:
        ticker = None
        st.title('WELCOME TO OUR WEBSITE')
    elif not tick and not option_tick:
        ticker = None
        st.title('WELCOME TO OUR WEBSITE')
    elif tick and not option_tick:
        ticker = tick
    elif not tick and option_tick:
        ticker = option_tick + '.VN'
    elif tick and option_tick:
        ticker = None  # Reset ticker to None if both tick and option_tick have values
        st.subheader('Select only one Ticker')

    if ticker is not None:
        caplock_ticker = ticker.title().upper()
        st.title(caplock_ticker)
        mck = yf.Ticker(ticker)
        name = mck.info['longName']
        st.subheader(name)

        bsheet = mck.balance_sheet
        income = mck.income_stmt
        cfs = mck.cashflow
        statistic = mck.info
        years = bsheet.columns[-5:]  # 4 cột cho 4 năm và 1 cột cho TTM

        if bsheet.empty:
            st.caption('The information is not sufficient for evaluation')

        elif income.empty:
            st.caption('The information is not sufficient for evaluation')

        elif cfs.empty:
            st.caption('The information is not sufficient for evaluation')

        else:
            quarter_bsheet = mck.quarterly_balance_sheet
            first_column_index = quarter_bsheet.columns[0]
            TTM_bsheet = quarter_bsheet[first_column_index]
            second_column_index = quarter_bsheet.columns[1]
            TTM_bsheet2 = quarter_bsheet[second_column_index]
            TTM_bsheet3 = quarter_bsheet.iloc[:, :4].sum(axis=1)
            five_column_index = quarter_bsheet.columns[len(quarter_bsheet.columns) - 1]
            TTM_bsheet4 = quarter_bsheet[five_column_index]

            quarter_income = mck.quarterly_income_stmt
            TTM = quarter_income.iloc[:, :4].sum(axis=1)

            quarter_cfs = mck.quarterly_cashflow
            TTM_cfs = quarter_cfs.iloc[:, :4].sum(axis=1)

            # F-score
            # Score #1 - change in Total Revenue (Thay đổi doanh thu)
            revenue_values = [income[year]['Total Revenue'] for year in years[::-1]]
            rv_scores = [1 if revenue_values[i] < revenue_values[i + 1] else 0 for i in
                         range(len(revenue_values) - 1)]
            annual_rv_score = sum(rv_scores)
            TTM_rv_score = 1 if TTM['Total Revenue'] > income[years[-(len(revenue_values) - 1)]][
                'Total Revenue'] else 0
            total_rv_score = annual_rv_score + TTM_rv_score

            # Score #2 - change in Net Income (Thay đổi lợi nhuận)
            ni_values = [income[year]['Net Income'] for year in years[::-1]]
            ni_scores = [1 if ni_values[i] < ni_values[i + 1] else 0 for i in range(len(ni_values) - 1)]
            annual_ni_score = sum(ni_scores)
            TTM_ni_score = 1 if TTM['Net Income'] > income[years[-(len(revenue_values) - 1)]][
                'Net Income'] else 0
            total_ni_score = annual_ni_score + TTM_ni_score

            # Score #3 - change in Operating Cash Flow (Thay đổi dòng tiền đầu tư)
            opcf_values = [cfs[year]['Operating Cash Flow'] for year in years[::-1]]
            opcf_scores = [1 if opcf_values[i] < opcf_values[i + 1] else 0 for i in range(len(opcf_values) - 1)]
            annual_opcf_score = sum(opcf_scores)
            TTM_opcf_score = 1 if TTM_cfs['Operating Cash Flow'] > cfs[years[-(len(revenue_values) - 1)]][
                'Operating Cash Flow'] else 0
            total_opcf_score = annual_opcf_score + TTM_opcf_score

            # Score #4 - change in Free Cash Flow
            fcf_values = [cfs[year]['Free Cash Flow'] for year in years[::-1]]
            fcf_scores = [1 if fcf_values[i] < fcf_values[i + 1] else 0 for i in range(len(fcf_values) - 1)]
            annual_fcf_score = sum(fcf_scores)
            TTM_fcf_score = 1 if TTM_cfs['Free Cash Flow'] > cfs[years[-(len(revenue_values) - 1)]][
                'Free Cash Flow'] else 0
            total_fcf_score = annual_fcf_score + TTM_fcf_score

            # Score #5 - change in EPS
            eps_values = [income[year]['Basic EPS'] for year in years[::-1]]
            eps_scores = [1 if eps_values[i] < eps_values[i + 1] else 0 for i in range(len(eps_values) - 1)]
            annual_eps_score = sum(eps_scores)
            TTM_eps_score = 1 if TTM['Basic EPS'] > income[years[-(len(revenue_values) - 1)]][
                'Basic EPS'] else 0
            total_eps_score = annual_eps_score + TTM_eps_score

            # Score #6 - change in ROE
            roe_values = [income[year]['Net Income'] / bsheet[year]['Total Equity Gross Minority Interest']
                          for year in years[::-1]]
            roe_scores = [1 if roe_values[i] < roe_values[i + 1] else 0 for i in range(len(roe_values) - 1)]
            annual_roe_score = sum(roe_scores)
            TTM_roe_score = 1 if TTM['Net Income'] / TTM_bsheet['Total Equity Gross Minority Interest'] > \
                                 income[years[-(len(revenue_values) - 1)]][
                                     'Net Income'] / bsheet[years[-(len(revenue_values) - 1)]][
                                     'Total Equity Gross Minority Interest'] else 0
            total_roe_score = annual_roe_score + TTM_roe_score

            # Score #7 - change in Current Ratio
            cr_ratio_history = [bsheet[year]['Current Assets'] / bsheet[year]['Current Liabilities'] for year in
                                years[::-1]]
            cr_scores = [1 if cr_ratio_history[i] < cr_ratio_history[i + 1] else 0 for i in
                         range(len(cr_ratio_history) - 1)]
            annual_cr_score = sum(cr_scores)
            TTM_cr_score = 1 if TTM_bsheet['Current Assets'] / TTM_bsheet['Current Liabilities'] > \
                                bsheet[years[-(len(revenue_values) - 1)]][
                                    'Current Assets'] / bsheet[years[-(len(revenue_values) - 1)]][
                                    'Current Liabilities'] else 0
            total_cr_score = annual_cr_score + TTM_cr_score

            # Score #8 - change in Debt to Equity Ratio
            der_values = [bsheet[year]['Total Debt'] / bsheet[year]['Total Equity Gross Minority Interest'] for
                          year
                          in
                          years[::-1]]
            der_scores = [1 if der_values[i] > der_values[i + 1] else 0 for i in range(len(der_values) - 1)]
            annual_der_score = sum(der_scores)
            TTM_der_score = 1 if TTM_bsheet['Total Debt'] / TTM_bsheet['Total Equity Gross Minority Interest'] < \
                                 bsheet[years[-(len(revenue_values) - 1)]][
                                     'Total Debt'] / bsheet[years[-(len(revenue_values) - 1)]][
                                     'Total Equity Gross Minority Interest'] else 0
            total_der_score = annual_der_score + TTM_der_score

            # Score #9 - change in Accounts Receivable
            ar_values = [bsheet[year]['Accounts Receivable'] for year in years[::-1]]
            ar_scores = [1 if ar_values[i] > ar_values[i + 1] else 0 for i in range(len(ar_values) - 1)]
            annual_ar_score = sum(ar_scores)
            TTM_ar_score = 1 if TTM_bsheet['Accounts Receivable'] < bsheet[years[-(len(revenue_values) - 1)]][
                'Accounts Receivable'] else 0
            total_ar_score = annual_ar_score + TTM_ar_score

            # Calculate the total score
            total_score = total_rv_score + total_ni_score + total_opcf_score + total_fcf_score + total_roe_score + total_eps_score + total_cr_score + total_der_score + total_ar_score

            # GURU
            # Liquidity + Dividend
            cr_ratio = round((TTM_bsheet['Current Assets'] / TTM_bsheet['Current Liabilities']), 2)
            # Lấy dữ liệu từ năm trước đến năm hiện tại
            cr_list = cr_ratio_history + [cr_ratio]
            cr_rank = sorted(cr_list)
            cr_ratio_p = cr_rank.index(cr_ratio) + 1
            cr_len = len(cr_rank)
            cr_values2 = cr_ratio_p / cr_len

            if 'Inventory' in TTM_bsheet:
                qr_ratio = round(
                    ((TTM_bsheet['Current Assets'] - TTM_bsheet['Inventory']) / TTM_bsheet['Current Liabilities']), 2)
                qr_ratio_history = [(bsheet[year]['Current Assets'] - bsheet[year]['Inventory']) / bsheet[year]['Current Liabilities'] for year in
                                    years[::-1]]

                qr_ratio_history = []
                for year in years[::-1]:

                    inventory_values = bsheet.loc['Inventory', year]
                    if pd.isnull(inventory_values):  # Kiểm tra xem inventory_values có phải là NaN hay không
                        inventory_values = bsheet.loc['Inventory', year - pd.DateOffset(years=1)]

                    cr_liabilities = bsheet.loc['Current Liabilities', year]
                    cr_asset = bsheet.loc['Current Assets', year]

                    if pd.notnull(cr_liabilities) and pd.notnull(cr_asset) and pd.notnull(inventory_values):
                        qr_ratio = (cr_asset - inventory_values) / cr_liabilities
                        qr_ratio_history.append(qr_ratio)
                    else:
                        qr_ratio_history.append(0)
            else:
                inventory_values = 0
                qr_ratio = cr_ratio
                qr_ratio_history = cr_ratio_history
                 
            qr_list = qr_ratio_history + [qr_ratio] 
            qr_rank = sorted(qr_list)
            qr_ratio_p = qr_rank.index(qr_ratio) + 1
            qr_len = len(qr_rank)
            qr_values = qr_ratio_p / qr_len

            car_ratio = round((TTM_bsheet['Cash And Cash Equivalents'] / TTM_bsheet['Current Liabilities']), 2)
            car_ratio_history = [
                bsheet.loc['Cash And Cash Equivalents', year] / (bsheet.loc['Current Liabilities', year] or 1) for year
                in
                years[::-1]]

            car_list = car_ratio_history + [car_ratio]
            car_rank = sorted(car_list)
            car_ratio_p = car_rank.index(car_ratio) + 1
            car_len = len(car_rank)
            car_values = car_ratio_p / car_len

            dso_ratio = round((TTM_bsheet['Accounts Receivable'] / TTM['Total Revenue']) * 365, 2)
            dso_ratio_history = [
                bsheet.loc['Accounts Receivable', year] * 365 / (income.loc['Total Revenue', year] or 1) for year in
                years[::-1]]

            dso_list = dso_ratio_history + [dso_ratio]
            dso_rank = sorted(dso_list, reverse=True)
            dso_ratio_p = dso_rank.index(dso_ratio) + 1
            dso_len = len(dso_rank)
            dso_values = dso_ratio_p / dso_len

            ap_average_values = (TTM_bsheet4['Accounts Payable'] + TTM_bsheet['Accounts Payable']) / 2
            dp_ratio = round((ap_average_values / TTM['Cost Of Revenue']) * 365, 2)
            dp_ratio_history = [bsheet.loc['Accounts Payable', year] * 365 / (income.loc['Cost Of Revenue', year] or 1)
                                for year in years[::-1]]

            dp_list = dp_ratio_history + [dp_ratio]
            dp_rank = sorted(dp_list)
            dp_ratio_p = dp_rank.index(dp_ratio) + 1
            dp_len = len(dp_rank)
            dp_values = dp_ratio_p / dp_len

            if 'Inventory' in TTM_bsheet:

                inv_average = (TTM_bsheet4['Inventory'] + TTM_bsheet['Inventory']) / 2
                dio_ratio = round((inv_average / TTM['Cost Of Revenue']) * 365, 2)
                dio_ratio_history = [bsheet.loc['Inventory', year] * 365 / (income.loc['Cost Of Revenue', year] or 1)
                                     for
                                     year in years[::-1]]
                dio_ratio_history = []
                for year in years[::-1]:
                    inventory_values = bsheet.loc['Inventory', year]
                    if pd.isnull(inventory_values):  # Kiểm tra xem inventory_values có phải là NaN hay không
                        inventory_values = bsheet.loc['Inventory', year - pd.DateOffset(years=1)]

                    cost_of_rvn = income.loc['Cost Of Revenue', year]

                    if pd.notnull(cost_of_rvn) and pd.notnull(inventory_values):
                        dio_ratio = (inventory_values) * 365 / cost_of_rvn
                        dio_ratio_history.append(dio_ratio)
                    else:
                        dio_ratio_history.append(0)

                dio_list = dio_ratio_history + [dio_ratio]
                dio_rank = sorted(dio_list, reverse=True)
                dio_ratio_p = dio_rank.index(dio_ratio) + 1
                dio_len = len(dio_rank)
                dio_values = dio_ratio_p / dio_len
            else:
                dio_values = 0
                dio_ratio = 0
                dio_ratio_p = 0
                dio_len = 0
                dio_list = 0

            div_ratio = mck.info[
                            'trailingAnnualDividendYield'] * 100 if 'trailingAnnualDividendYield' in statistic else 0
            pr_ratio = mck.info['payoutRatio'] if 'payoutRatio' in statistic else 0
            five_years_ratio = mck.info['fiveYearAvgDividendYield'] if 'fiveYearAvgDividendYield' in statistic else 0
            forward_ratio = mck.info['dividendYield'] * 100 if 'dividendYield' in statistic else 0

            # PE Ratio
            shares_outstanding = mck.info['sharesOutstanding']
            PE_ratio = mck.info['trailingPE'] if 'trailingPE' in mck.info else 0

            pe_ratio_history = []
            for year in years[::-1]:
                basic_eps = income.loc['Basic EPS', year]
                if pd.isnull(basic_eps):  # Kiểm tra xem basic_eps có phải là NaN hay không
                    basic_eps = income.loc['Basic EPS', year - pd.DateOffset(years=1)]

                total_capitalization = bsheet.loc['Total Capitalization', year]
                share_issued = bsheet.loc['Share Issued', year]

                if pd.notnull(total_capitalization) and pd.notnull(share_issued) and pd.notnull(basic_eps):
                    pe_ratio = (total_capitalization / share_issued) / basic_eps
                    pe_ratio_history.append(pe_ratio)
                else:
                    pe_ratio_history.append(0)

            PE_ratio_list = pe_ratio_history + [PE_ratio]
            PE_ratio_rank = sorted(PE_ratio_list)
            PE_ratio_p = PE_ratio_rank.index(PE_ratio) + 1
            PE_ratio_len = len(PE_ratio_rank)
            PE_ratio_values = PE_ratio_p / PE_ratio_len

            # P/S Ratio
            PS_ratio = mck.info['currentPrice'] / mck.info["revenuePerShare"]
            ps_ratio_history = [(bsheet.loc['Total Capitalization', year] / bsheet.loc['Share Issued', year]) / (
                    income.loc['Total Revenue', year] / bsheet.loc['Share Issued', year]) for year in years[::-1]]
            PS_ratio_list = ps_ratio_history + [PS_ratio]
            PS_ratio_rank = sorted(PS_ratio_list)
            PS_ratio_p = PS_ratio_rank.index(PS_ratio) + 1
            PS_ratio_len = len(PS_ratio_rank)
            PS_ratio_values = PS_ratio_p / PS_ratio_len

            # P/B Ratio
            PB_ratio = 0
            if 'priceToBook' in mck.info:
                PB_ratio = mck.info['priceToBook']
            else:
                PB_ratio = 0
            pb_ratio_history = []
            for year in years[::-1]:
                basic_eps = income.loc['Basic EPS', year]
                if pd.isnull(basic_eps):  # Kiểm tra xem basic_eps có phải là NaN hay không
                    basic_eps = income.loc['Basic EPS', year - pd.DateOffset(years=1)]

                stock_holderequity = bsheet.loc['Stockholders Equity', year]
                net_income1 = income.loc['Net Income', year]

                if pd.notnull(stock_holderequity) and pd.notnull(net_income1) and pd.notnull(basic_eps):
                    pb_ratio = (stock_holderequity / net_income1) / basic_eps
                    pb_ratio_history.append(pb_ratio)
                else:
                    pb_ratio_history.append(0)

            PB_ratio_list = pb_ratio_history + [PB_ratio]
            PB_ratio_rank = sorted(PB_ratio_list)
            PB_ratio_p = PB_ratio_rank.index(PB_ratio) + 1
            PB_ratio_len = len(PB_ratio_rank)
            PB_ratio_values = PB_ratio_p / PB_ratio_len

            # Price-to-tangible-book Ratio
            Price_to_TBV = mck.info['currentPrice'] / (TTM_bsheet['Tangible Book Value'] / shares_outstanding)
            Price_to_TBV_history = [(bsheet.loc['Total Capitalization', year] / bsheet.loc['Share Issued', year]) / (
                    bsheet.loc['Tangible Book Value', year] / bsheet.loc['Share Issued', year]) for year in years[::-1]]
            Price_to_TBV_list = Price_to_TBV_history + [Price_to_TBV]
            Price_to_TBV_rank = sorted(Price_to_TBV_list)
            Price_to_TBV_p = Price_to_TBV_rank.index(Price_to_TBV) + 1
            Price_to_TBV_len = len(Price_to_TBV_rank)
            Price_to_TBV_values = Price_to_TBV_p / Price_to_TBV_len

            # Price-to-Free-Cash_Flow Ratio
            price_to_FCF = mck.info['currentPrice'] / (TTM_cfs['Free Cash Flow'] / shares_outstanding)
            Price_to_FCF_history = [(bsheet.loc['Total Capitalization', year] / bsheet.loc['Share Issued', year]) / (
                    cfs.loc['Free Cash Flow', year] / bsheet.loc['Share Issued', year]) for year in years[::-1]]
            price_to_FCF_list = Price_to_FCF_history + [price_to_FCF]
            price_to_FCF_rank = sorted(price_to_FCF_list)
            price_to_FCF_p = price_to_FCF_rank.index(price_to_FCF) + 1
            price_to_FCF_len = len(price_to_FCF_rank)
            price_to_FCF_values = price_to_FCF_p / price_to_FCF_len

            # EV-to-EBIT
            EV_to_EBIT = mck.info['enterpriseValue'] / TTM['EBIT']
            EV_to_EBIT_history = [(bsheet.loc['Total Capitalization', year] - bsheet.loc[
                'Total Liabilities Net Minority Interest', year] - bsheet.loc[
                                       'Cash Cash Equivalents And Short Term Investments', year]) / (
                                      income.loc['EBIT', year]) for year in years[::-1]]
            EV_to_EBIT_list = EV_to_EBIT_history + [EV_to_EBIT]
            EV_to_EBIT_rank = sorted(EV_to_EBIT_list)
            EV_to_EBIT_p = EV_to_EBIT_rank.index(EV_to_EBIT) + 1
            EV_to_EBIT_len = len(EV_to_EBIT_rank)
            EV_to_EBIT_values = EV_to_EBIT_p / EV_to_EBIT_len

            # EV-to-EBITDA
            EV_to_EBITDA = mck.info['enterpriseValue'] / TTM['EBITDA']
            EV_to_EBITDA_history = [(bsheet.loc['Total Capitalization', year] - bsheet.loc[
                'Total Liabilities Net Minority Interest', year] - bsheet.loc[
                                         'Cash Cash Equivalents And Short Term Investments', year]) / (
                                        income.loc['EBITDA', year]) for year in years[::-1]]
            EV_to_EBITDA_list = EV_to_EBITDA_history + [EV_to_EBITDA]
            EV_to_EBITDA_rank = sorted(EV_to_EBITDA_list)
            EV_to_EBITDA_p = EV_to_EBITDA_rank.index(EV_to_EBITDA) + 1
            EV_to_EBITDA_len = len(EV_to_EBITDA_rank)
            EV_to_EBITDA_values = EV_to_EBITDA_p / EV_to_EBITDA_len

            # EV-to-Revenue
            EV_to_Revenue = mck.info['enterpriseValue'] / TTM['Total Revenue']
            EV_to_Revenue_history = [(bsheet.loc['Total Capitalization', year] - bsheet.loc[
                'Total Liabilities Net Minority Interest', year] - bsheet.loc[
                                          'Cash Cash Equivalents And Short Term Investments', year]) / (
                                         income.loc['Total Revenue', year]) for year in years[::-1]]
            EV_to_Revenue_list = EV_to_Revenue_history + [EV_to_Revenue]
            EV_to_Revenue_rank = sorted(EV_to_Revenue_list)
            EV_to_Revenue_p = EV_to_Revenue_rank.index(EV_to_Revenue) + 1
            EV_to_Revenue_len = len(EV_to_Revenue_rank)
            EV_to_Revenue_values = EV_to_Revenue_p / EV_to_Revenue_len

            # EV-to-FCF
            EV_to_FCF = mck.info['enterpriseValue'] / TTM_cfs['Free Cash Flow']
            EV_to_FCF_history = [(bsheet.loc['Total Capitalization', year] - bsheet.loc[
                'Total Liabilities Net Minority Interest', year] - bsheet.loc[
                                      'Cash Cash Equivalents And Short Term Investments', year]) / (
                                     cfs.loc['Free Cash Flow', year]) for year in years[::-1]]
            EV_to_FCF_list = EV_to_FCF_history + [EV_to_FCF]
            EV_to_FCF_rank = sorted(EV_to_FCF_list)
            EV_to_FCF_p = EV_to_FCF_rank.index(EV_to_FCF) + 1
            EV_to_FCF_len = len(EV_to_FCF_rank)
            EV_to_FCF_values = EV_to_FCF_p / EV_to_FCF_len

            # Price-to-Net-Current-Asset-Value
            Price_to_Net_CAV = mck.info['currentPrice'] / (
                    (TTM_bsheet['Current Assets'] - TTM_bsheet['Current Liabilities']) / shares_outstanding)
            Price_to_Net_CAV_history = [
                (bsheet.loc['Total Capitalization', year] / bsheet.loc['Share Issued', year]) / (
                        (bsheet.loc['Current Assets', year] - bsheet.loc['Current Liabilities', year])
                        / bsheet.loc['Share Issued', year]) for year in years[::-1]]
            Price_to_Net_CAV_list = Price_to_Net_CAV_history + [Price_to_Net_CAV]
            Price_to_Net_CAV_rank = sorted(Price_to_Net_CAV_list)
            Price_to_Net_CAV_p = Price_to_Net_CAV_rank.index(Price_to_Net_CAV) + 1
            Price_to_Net_CAV_len = len(Price_to_Net_CAV_rank)
            Price_to_Net_CAV_values = Price_to_Net_CAV_p / Price_to_Net_CAV_len

            # Earnings Yields (Greenblatt) %
            EarningsYields = (TTM['EBIT'] / mck.info['enterpriseValue']) * 100
            EarningsYields_history = [
                ((income.loc['EBIT', year] / (bsheet.loc['Total Capitalization', year] - bsheet.loc[
                    'Total Liabilities Net Minority Interest', year] - bsheet.loc[
                                                  'Cash Cash Equivalents And Short Term Investments', year])) * 100)
                for year in years[::-1]]
            EarningsYields_list = EarningsYields_history + [EarningsYields]
            EarningsYields_rank = sorted(EarningsYields_list)
            EarningsYields_p = EarningsYields_rank.index(EarningsYields) + 1
            EarningsYields_len = len(EarningsYields_rank)
            EarningsYields_values = EarningsYields_p / EarningsYields_len

            # FCF Yield %
            FCFYield = (TTM_cfs['Free Cash Flow'] / mck.info['marketCap']) * 100 if 'marketCap' in mck.info else (
                                                                                                                             TTM_cfs[
                                                                                                                                 'Free Cash Flow'] /
                                                                                                                             mck.basic_info[
                                                                                                                                 'marketCap']) * 100
            FCFYield_history = [((cfs.loc['Free Cash Flow', year] / bsheet.loc['Total Capitalization', year]) * 100) for
                                year in
                                years[::-1]]
            FCFYield_list = FCFYield_history + [FCFYield]
            FCFYield_rank = sorted(FCFYield_list)
            FCFYield_p = FCFYield_rank.index(FCFYield) + 1
            FCFYield_len = len(FCFYield_rank)
            FCFYield_values = FCFYield_p / FCFYield_len

            # Profitability Rank
            # Gross Margin %
            gr_margin = round((TTM['Gross Profit'] * 100 / TTM['Total Revenue']), 2)
            gr_margin_history = [
                income.loc['Gross Profit', year] * 100 / (income.loc['Total Revenue', year] or 1) for year in
                years[::-1]]
            # Tìm min max
            gr_margin_list = gr_margin_history + [gr_margin]
            gr_margin_rank = sorted(gr_margin_list)
            gr_margin_p = gr_margin_rank.index(gr_margin) + 1
            gr_margin_len = len(gr_margin_rank)
            gr_margin_values = gr_margin_p / gr_margin_len

            # Operating Margin %
            op_margin = round((TTM['Operating Income'] * 100 / TTM['Total Revenue']), 2)
            op_margin_history = [
                income.loc['Operating Income', year] * 100 / (income.loc['Total Revenue', year] or 1) for year in
                years[::-1]]
            # Tìm min max
            op_margin_list = op_margin_history + [op_margin]
            op_margin_rank = sorted(op_margin_list)
            op_margin_p = op_margin_rank.index(op_margin) + 1
            op_margin_len = len(op_margin_rank)
            op_margin_values = op_margin_p / op_margin_len
            # Net Margin %
            net_margin = round((TTM['Net Income'] * 100 / TTM['Total Revenue']), 2)
            net_margin_history = [
                income.loc['Net Income', year] * 100 / (income.loc['Total Revenue', year] or 1) for year in
                years[::-1]]
            # Tìm min max
            net_margin_list = net_margin_history + [net_margin]
            net_margin_rank = sorted(net_margin_list)
            net_margin_p = net_margin_rank.index(net_margin) + 1
            net_margin_len = len(net_margin_rank)
            net_margin_values = net_margin_p / net_margin_len
            # FCF margin %
            fcf_margin = round((TTM_cfs['Free Cash Flow'] * 100 / TTM['Total Revenue']), 2)
            fcf_margin_history = [
                cfs.loc['Free Cash Flow', year] * 100 / (income.loc['Total Revenue', year] or 1) for year in
                years[::-1]]
            # Tìm min max
            fcf_margin_list = fcf_margin_history + [fcf_margin]
            fcf_margin_rank = sorted(fcf_margin_list)
            fcf_margin_p = fcf_margin_rank.index(fcf_margin) + 1
            fcf_margin_len = len(fcf_margin_rank)
            fcf_margin_values = fcf_margin_p / fcf_margin_len

            # ROE%
            roe_stock_average = (TTM_bsheet2['Total Equity Gross Minority Interest'] + TTM_bsheet[
                'Total Equity Gross Minority Interest']) / 2
            roe_margin = round((TTM['Net Income'] * 100 / roe_stock_average), 2)
            roe_margin_history = [
                income.loc['Net Income', year] * 100 / (bsheet.loc['Total Equity Gross Minority Interest', year] or 1)
                for year in years[::-1]]
            # Tìm min max
            roe_margin_list = roe_margin_history + [roe_margin]
            roe_margin_rank = sorted(roe_margin_list)
            roe_margin_p = roe_margin_rank.index(roe_margin) + 1
            roe_margin_len = len(roe_margin_rank)
            roe_margin_values = roe_margin_p / roe_margin_len

            # ROA%
            roa_tta_average = (TTM_bsheet2['Total Assets'] + TTM_bsheet['Total Assets']) / 2
            roa_margin = round((TTM['Net Income'] * 100 / roa_tta_average), 2)
            roa_margin_history = [income.loc['Net Income', year] * 100 / (bsheet.loc['Total Assets', year] or 1)
                                  for year in years[::-1]]
            # Tìm min max
            roa_margin_list = roa_margin_history + [roa_margin]
            roa_margin_rank = sorted(roa_margin_list)
            roa_margin_p = roa_margin_rank.index(roa_margin) + 1
            roa_margin_len = len(roa_margin_rank)
            roa_margin_values = roa_margin_p / roa_margin_len

            # ROC (Joel Greenblatt) %
            fix_work_average = (TTM_bsheet['Net Tangible Assets'] + TTM_bsheet['Working Capital']) / 2
            roc_margin = round((TTM['EBIT'] * 100 / fix_work_average), 2)
            roc_margin_history = [income.loc['EBIT', year] * 100 / (
                    (bsheet.loc['Net Tangible Assets', year] + bsheet.loc['Working Capital', year]) / 2 or 1)
                                  for year in years[::-1]]
            # Tìm min max
            roc_margin_list = roc_margin_history + [roc_margin]
            roc_margin_rank = sorted(roc_margin_list)
            roc_margin_p = roc_margin_rank.index(roc_margin) + 1
            roc_margin_len = len(roc_margin_rank)
            roc_margin_values = roc_margin_p / roc_margin_len

            # ROCE%
            cap_em_1 = (TTM_bsheet['Total Assets'] - TTM_bsheet['Current Liabilities'])
            cap_em_2 = (TTM_bsheet2['Total Assets'] - TTM_bsheet2['Current Liabilities'])
            cap_em_average = (cap_em_1 + cap_em_2) / 2
            roce_margin = round((TTM['EBIT'] * 100 / cap_em_average), 2)
            roce_margin_history = [income.loc['EBIT', year] * 100 / (
                    bsheet.loc['Total Assets', year] - bsheet.loc['Current Liabilities', year] or 1)
                                   for year in years[::-1]]
            # Tìm min max
            roce_margin_list = roce_margin_history + [roce_margin]
            roce_margin_rank = sorted(roce_margin_list)
            roce_margin_p = roce_margin_rank.index(roce_margin) + 1
            roce_margin_len = len(roce_margin_rank)
            roce_margin_values = roce_margin_p / roce_margin_len

            # Financial Strength

            # Cash_to_debt
            cash_debt = TTM_bsheet['Cash Cash Equivalents And Short Term Investments'] / TTM_bsheet['Total Debt']
            cash_debt_history = [bsheet.loc['Cash Cash Equivalents And Short Term Investments', year] / (
                        bsheet.loc['Total Debt', year] or 1) for year in
                                 years[::-1]]

            cash_debt_list = cash_debt_history + [cash_debt]
            cash_debt_rank = sorted(cash_debt_list)
            cash_debt_p = cash_debt_rank.index(cash_debt) + 1
            cash_debt_len = len(cash_debt_rank)
            cash_debt_values = cash_debt_p / cash_debt_len

            # Equity to Asset
            equity_asset = TTM_bsheet['Stockholders Equity'] / TTM_bsheet['Total Assets']
            equity_asset_history = [bsheet.loc['Stockholders Equity', year] / (bsheet.loc['Total Assets', year] or 1)
                                    for year in
                                    years[::-1]]

            equity_asset_list = equity_asset_history + [equity_asset]
            equity_asset_rank = sorted(equity_asset_list)
            equity_asset_p = equity_asset_rank.index(equity_asset) + 1
            equity_asset_len = len(equity_asset_rank)
            equity_asset_values = equity_asset_p / equity_asset_len

            # Debt to Equity
            debt_equity = TTM_bsheet['Total Debt'] / TTM_bsheet['Stockholders Equity']
            debt_equity_history = [bsheet.loc['Total Debt', year] / (bsheet.loc['Stockholders Equity', year] or 1) for
                                   year in
                                   years[::-1]]

            debt_equity_list = debt_equity_history + [debt_equity]
            debt_equity_rank = sorted(debt_equity_list, reverse=True)
            debt_equity_p = debt_equity_rank.index(debt_equity) + 1
            debt_equity_len = len(debt_equity_rank)
            debt_equity_values = debt_equity_p / debt_equity_len

            # Debt to EBITDA
            debt_ebitda = TTM_bsheet['Total Debt'] / TTM['EBITDA'] if 'Total Debt' in TTM_bsheet else 0
            debt_ebitda_history = [bsheet.loc['Total Debt', year] / (income.loc['EBITDA', year] or 1) for year in
                                   years[::-1]]

            debt_ebitda_list = debt_ebitda_history + [debt_ebitda]
            debt_ebitda_rank = sorted(debt_ebitda_list)
            debt_ebitda_p = debt_ebitda_rank.index(debt_ebitda) + 1
            debt_ebitda_len = len(debt_ebitda_rank)
            debt_ebitda_values = debt_ebitda_p / debt_ebitda_len

            # Interest Coverage
            if 'Interest Expense' in TTM and TTM['Interest Expense'] !=0:
                interest_coverage = TTM['Operating Income'] / TTM['Interest Expense'] 
                interest_coverage_history = [
                    income.loc['Operating Income', year] / (income.loc['Interest Expense', year] or 1) for year in
                    years[::-1]]
                interest_coverage_list = interest_coverage_history + [interest_coverage]
                interest_coverage_rank = sorted(interest_coverage_list)
                interest_coverage_p = interest_coverage_rank.index(interest_coverage) + 1
                interest_coverage_len = len(interest_coverage_rank)
                interest_coverage_values = interest_coverage_p / interest_coverage_len
            else: 
                interest_coverage_values = 0
                interest_coverage = 'None'
                interest_coverage_list = [0]
                interest_coverage_p = 0
                interest_coverage_len = 0
            # Altman F-Score
            a = TTM_bsheet['Working Capital'] / TTM_bsheet['Total Assets']
            b = TTM_bsheet['Retained Earnings'] / TTM_bsheet['Total Assets']
            c = TTM['EBIT'] / TTM_bsheet['Total Assets']
            d = mck.info['marketCap'] / TTM_bsheet[
                'Total Liabilities Net Minority Interest'] if 'marketCap' in mck.info else mck.basic_info['marketCap'] / \
                                                                                           TTM_bsheet['Total Liabilities Net Minority Interest']
            e = TTM['Total Revenue'] / TTM_bsheet['Total Assets']
            altmanz_score = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + e

            # Piotroski F-Score
            # Score #1 - ROA
            roa_score = 1 if TTM['Net Income'] > 0 else 0

            # Score #2 - Operating Cash Flow
            ocf_score = 1 if TTM_cfs['Operating Cash Flow'] > 0 else 0

            # Score #3 - change in ROA
            roa_1 = TTM['Net Income'] / TTM_bsheet4['Total Assets']
            roa_2 = income[years[1 - len(years)]]['Net Income'] / bsheet[years[2 - len(years)]]['Total Assets']
            croa_score = 1 if roa_1 > roa_2 else 0

            # Score #4 - Quality of Earnings (Accrual)
            acc_score = 1 if TTM_cfs['Operating Cash Flow'] > TTM['Net Income'] else 0

            # Score #5 - Leverage (long term debt/average total assets) (Moi lay 2 quy gan nhat 2022, yf khum co)
            t_assets = quarter_bsheet.sum(axis=1)
            ave_assets = t_assets / 5
            lv_1 = TTM_bsheet['Long Term Debt And Capital Lease Obligation'] / ave_assets[
                'Total Assets'] if 'Long Term Debt And Capital Lease Obligation' in TTM_bsheet else 0
            pre_assets = 1 / 2 * (bsheet[years[1 - len(years)]]['Total Assets'] + TTM_bsheet4['Total Assets'])
            lv_2 = TTM_bsheet4[
                       'Long Term Debt And Capital Lease Obligation'] / pre_assets if 'Long Term Debt And Capital Lease Obligation' in TTM_bsheet4 else 0
            lv_score = 0 if lv_1 > lv_2 else 1

            # Score #6 - change in Working Capital (Liquidity)
            cr_1 = TTM_bsheet['Current Assets'] / TTM_bsheet['Current Liabilities']
            cr_2 = bsheet[years[1 - len(years)]]['Current Assets'] / bsheet[years[1 - len(years)]][
                'Current Liabilities']
            cr_score = 1 if cr_1 > cr_2 else 0

            # Score #7 - change in Share Issued
            si_score = 0 if TTM_bsheet['Share Issued'] > bsheet[years[1 - len(years)]]['Share Issued'] else 1

            # Score #8 - change in Gross Margin
            gm_1 = TTM['Gross Profit'] / TTM['Total Revenue']
            gm_2 = income[years[1 - len(years)]]['Gross Profit'] / income[years[1 - len(years)]]['Total Revenue']
            gm_score = 1 if gm_1 > gm_2 else 0

            # Score #9 - change in Asset Turnover
            at_1 = TTM['Total Revenue'] / TTM_bsheet4['Total Assets']
            at_2 = income[years[1 - len(years)]]['Total Revenue'] / bsheet[years[2 - len(years)]]['Total Assets']
            at_score = 1 if at_1 > at_2 else 0

            piotroski = at_score + gm_score + si_score + cr_score + acc_score + lv_score + croa_score + roa_score + ocf_score

            # Beneish M-Score
            # Day Sales in Receivables Index
            if 'Receivables' in TTM_bsheet:
                t1 = TTM_bsheet['Receivables']/TTM['Total Revenue'] 
                pre_t1 = bsheet[years[1-len(years)]]['Receivables']/income[years[1-len(years)]]['Total Revenue'] if 'Receivables' in TTM_bsheet else 0
                dsri = t1 / pre_t1
            else: dsri = 0
            # Gross Margin Index
            t2 = TTM['Gross Profit'] / TTM['Total Revenue'] #is gm_1 and gm_2
            pre_t2 = income[years[1-len(years)]]['Gross Profit'] / income[years[1-len(years)]]['Total Revenue']
            gmi = pre_t2/t2
            # Asset Quality Index
            t3 = 1 - TTM_bsheet['Current Assets'] + TTM_bsheet['Net PPE']
            pre_t3 = 1 - bsheet[years[1-len(years)]]['Current Assets'] + bsheet[years[1-len(years)]]['Net PPE']
            aqi = t3/pre_t3
            # Sales Growth Index
            t4 = TTM['Total Revenue']
            pre_t4 = income[years[1-len(years)]]['Total Revenue']
            sgi = t4/pre_t4
            # Sales, General & Administrative expense index
            t5 = TTM['Selling General And Administration']/TTM['Total Revenue']
            pre_t5 = income[years[1-len(years)]]['Selling General And Administration']/income[years[1-len(years)]]['Total Revenue']
            sgai = t5/pre_t5
            # Depreciation Index
            if 'Depreciation Amortization Depletion' in cfs:
                t6 = TTM_cfs['Depreciation Amortization Depletion']/(TTM_cfs['Depreciation Amortization Depletion']+TTM_bsheet['Net PPE'])
                pre_t6 = cfs[years[1-len(years)]]['Depreciation Amortization Depletion']/(cfs[years[1-len(years)]]['Depreciation Amortization Depletion']+bsheet[years[1-len(years)]]['Net PPE'])
                depi = pre_t6/t6
            else: depi = 0
            # leverage Index
            if 'Long Term Debt' in bsheet:
                t7 = (TTM_bsheet['Long Term Debt'] + TTM_bsheet['Current Liabilities'])/TTM_bsheet['Total Assets']
                pre_t7 = (bsheet[years[1-len(years)]]['Long Term Debt'] + bsheet[years[1-len(years)]]['Current Liabilities'])/bsheet[years[1-len(years)]]['Total Assets']
                lvgi = t7/pre_t7
            else: lvgi = 0
            # Total Accruals to Total Assets
            tata = (TTM['Net Income Continuous Operations']-TTM_cfs['Operating Cash Flow'])/TTM_bsheet['Total Assets']
            
            m = -4.84 + 0.92 * dsri + 0.52 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi

            summary, f_score, estimation, valuation, guru  = st.tabs(
                ["Summary", "F-Score", 'Discount Rate', "Valuation", "Financial Analysis"])

            # Tính điểm cho guru
            liquidity_score = round(
                (cr_ratio_p + qr_ratio_p + car_ratio_p + dio_ratio_p + dso_ratio_p + dp_ratio_p) * 10 / (
                            cr_len + qr_len + car_len + dio_len + dso_len + dp_len), 0)
            profitability_score = round((
                                                gr_margin_p + op_margin_p + net_margin_p + fcf_margin_p + roe_margin_p + roa_margin_p + roc_margin_p + roce_margin_p) * 10 / (
                                                gr_margin_len + op_margin_len + net_margin_len + fcf_margin_len + roe_margin_len + roa_margin_len + roc_margin_len + roce_margin_len),
                                        0)

            gfvalues_score = round((
                                               PE_ratio_p + PS_ratio_p + PB_ratio_p + Price_to_TBV_p + price_to_FCF_p + EV_to_EBIT_p + EV_to_EBITDA_p + EV_to_Revenue_p + EV_to_FCF_p + Price_to_Net_CAV_p + EarningsYields_p + FCFYield_p) * 10 / (
                                               PE_ratio_len + PS_ratio_len + PB_ratio_len + Price_to_TBV_len + price_to_FCF_len + EV_to_EBIT_len + EV_to_EBITDA_len + EV_to_Revenue_len + EV_to_FCF_len + Price_to_Net_CAV_len + EarningsYields_len + FCFYield_len),
                                   0)
            financial_score = round(
                (cash_debt_p + equity_asset_p + debt_equity_p + debt_ebitda_p + interest_coverage_p) * 10 / (
                            cash_debt_len + equity_asset_len + debt_equity_len + debt_ebitda_len + interest_coverage_len),
                0)

            with summary:
                st.subheader('Candlestick Chart')
                current = datetime.today().date()
                start_date = st.date_input('Start Date', current - timedelta(days=365))
                end_date = st.date_input('End Date', current)

                dataa = yf.download(ticker, start=start_date, end=end_date)

                # Các đường trung bình
                dataa['EMA20'] = dataa['Close'].ewm(span=20, adjust=False).mean()
                dataa['MA50'] = dataa['Close'].rolling(50).mean()
                dataa['MA100'] = dataa['Close'].rolling(100).mean()
                dataa['MA150'] = dataa['Close'].rolling(150).mean()

                if dataa.empty:
                    st.write("<p style='color:red'><strong>Please reset the date to see the chart</strong></p>",
                             unsafe_allow_html=True)
                else:
                    fig = go.Figure(data=[
                        go.Candlestick(x=dataa.index, open=dataa['Open'], high=dataa['High'], low=dataa['Low'],
                                       close=dataa['Close'],
                                       name='Candle Stick'),
                        go.Scatter(x=dataa.index, y=dataa['EMA20'], line=dict(color='green', width=1.5, dash='dot'),
                                   name='EMA20'),
                        go.Scatter(x=dataa.index, y=dataa['MA50'], line=dict(color='blue', width=1.5), name='MA50'),
                        go.Scatter(x=dataa.index, y=dataa['MA100'], line=dict(color='yellow', width=1.5), name='MA100'),
                        go.Scatter(x=dataa.index, y=dataa['MA150'], line=dict(color='red', width=1.5), name='MA150'),
                    ])

                    fig.update_layout(autosize=True, width=1100, height=750,
                                      legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1))

                    st.plotly_chart(fig)

                    # Lấy thông tin cơ bản (profile) của mã cổ phiếu
                    mck_info = mck.get_info()

                    # Tạo DataFrame từ thông tin cơ bản
                    df = pd.DataFrame({
                        'Tiêu đề': ['Address', 'City', 'Country', 'Website', 'Industry', 'Sector', 'Description'],
                        'Thông tin': [
                            mck_info.get('address1', 'N/A'),
                            mck_info.get('city', 'N/A'),
                            mck_info.get('country', 'N/A'),
                            mck_info.get('website', 'N/A'),
                            mck_info.get('industry', 'N/A'),
                            mck_info.get('sector', 'N/A'),
                            mck_info.get('longBusinessSummary', 'N/A')
                        ]
                    })

                    # Hiển thị bảng thông tin cơ bản
                    st.table(df)

                genai.configure(api_key="AIzaSyCjnPWgJGDlqIh_il8SzbzbXZmPBywFbFU")
              
                # Set up the model
                generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 8192,
                }

                safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                ]

                model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                            generation_config=generation_config,
                                            safety_settings=safety_settings)

                convo = model.start_chat(history=[
                {
                    "role": "user",
                    "parts": ["Phân tích MOAT của " + str(name) + "(" + str(mck) + ")" + "theo thang điểm 10"]
                },
                
                ])

                convo.send_message("User input")
                st.write(convo.last.text)

                datav = {
                    'Time': [year.date().strftime("%Y-%m-%d") for year in years[::-1]] + ['TTM'],
                    'Revenue': [income.loc['Total Revenue', year] for year in years[::-1]] + [
                        TTM['Total Revenue']],
                    'Net Income': [income.loc['Net Income', year] for year in years[::-1]] + [
                        TTM['Net Income']],
                    'Free Cash Flow': [cfs.loc['Free Cash Flow', year] for year in years[::-1]] + [
                        TTM_cfs['Free Cash Flow']],
                    'Operating Cash Flow': [cfs.loc['Operating Cash Flow', year] for year in years[::-1]] + [
                        TTM_cfs['Operating Cash Flow']],
                    'ROE': [income.loc['Net Income', year] / (
                            bsheet.loc['Total Equity Gross Minority Interest', year] or 1)
                            for
                            year in years[::-1]] + [
                               TTM['Net Income'] / TTM_bsheet['Total Equity Gross Minority Interest']],
                    'EPS': [income.loc['Net Income', year] / (mck.info['sharesOutstanding'] or 1) for year in
                            years[::-1]] + [
                               TTM['Basic EPS']],
                    'Current Ratio': [bsheet.loc['Current Assets', year] / (
                            bsheet.loc['Current Liabilities', year] or 1)
                                      for
                                      year in years[::-1]] + [
                                         TTM_bsheet['Current Assets'] / TTM_bsheet['Current Liabilities']],
                    'Debt to Equity Ratio': [bsheet.loc['Total Debt', year] / (
                            bsheet.loc['Total Equity Gross Minority Interest', year] or 1) for year in
                                             years[::-1]] + [
                                                TTM_bsheet['Total Debt'] / TTM_bsheet[
                                                    'Total Equity Gross Minority Interest']],
                    'Accounts Receivable': [bsheet.loc['Accounts Receivable', year] for year in years[::-1]] + [
                        TTM_bsheet['Accounts Receivable']],
                    'EBITDA': [income.loc['EBITDA', year] for year in years[::-1]] + [TTM['EBITDA']],
                    'Cash': [bsheet.loc['Cash Cash Equivalents And Short Term Investments', year] for year in
                             years[::-1]] + [TTM_bsheet['Cash Cash Equivalents And Short Term Investments']],
                    'Debt': [bsheet.loc['Total Debt', year] for year in years[::-1]] + [TTM_bsheet['Total Debt']],
                    'Stockholders Equity': [bsheet.loc['Stockholders Equity', year] for year in years[::-1]] + [
                        TTM_bsheet['Stockholders Equity']],
                    'Total Assets': [bsheet.loc['Total Assets', year] for year in years[::-1]] + [
                        TTM_bsheet['Total Assets']],
                    'Stock Based Compensation': [cfs.loc['Stock Based Compensation', year] for year in years[::-1]] + [
                        TTM_cfs['Stock Based Compensation']] if 'Stock Based Compensation' in cfs else 0,
                    'Cash Flow for Dividends': [cfs.loc['Cash Dividends Paid', year] for year in years[::-1]] + [
                        TTM_cfs['Cash Dividends Paid']] if 'Cash Dividend Paid' in cfs else 0,
                    'Capital Expenditure': [cfs.loc['Capital Expenditure', year] for year in years[::-1]] + [
                        TTM_cfs['Capital Expenditure']] if 'Capital Expenditure' in TTM_cfs else 0
                }

                dfv = pd.DataFrame(datav)
                # Plot the chart using Plotly Express
                # Revenue, Net Income, Operating Cash Flow
                # create plot
                columns_to_plot = ['Revenue', 'Net Income', 'Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure']
                x = ['['] + dfv['Time'] + [']']
                # Plot grouped bar chart
                fig = px.bar(dfv, x, y=columns_to_plot, title='Financial Ratios',
                             height=530, width=1100, barmode='group')
    
                # Add text on top of each bar
                for col in columns_to_plot:
                    new_values = dfv[col] / 1e9
                    fig.update_traces(text=new_values.apply(lambda x: f'{x:.2f}B'), textposition='outside',
                                      selector=dict(name=col))
                fig.update_xaxes(fixedrange=True, title_text="Time")
                fig.update_layout(legend_title_text=None)
                fig.update_xaxes(fixedrange=True, title_text="")
                fig.update_yaxes(fixedrange=True, title_text="")
                # Display the chart in Streamlit app
                st.plotly_chart(fig)

                col3, col4 = st.columns(2)
                with col3:

                    # EPS
                    # Vẽ biểu đồ đường sử dụng Plotly Express
                    dfv['EPS'] = dfv['EPS'].round(2)
                    fig = px.line(dfv, x, y='EPS', title='EPS', markers='o', line_shape='spline', text='EPS')
                    fig.update_traces(textposition='top center')
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    # Thay đổi màu của đường
                    fig.update_traces(line=dict(color='firebrick'))
                    fig.update_layout(width=500, height=400)
                    # Hiển thị biểu đồ trong ứng dụng Streamlit
                    st.plotly_chart(fig)

                    # ROE
                    # Vẽ biểu đồ đường sử dụng Plotly Express
                    dfv['ROE'] = dfv['ROE'].round(2)
                    fig = px.line(dfv, x, y='ROE', title='ROE', markers='o', line_shape='spline', text='ROE')
                    fig.update_traces(textposition='top center')
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    # Thay đổi màu của đường
                    fig.update_traces(line=dict(color='mediumspringgreen'))
                    fig.update_layout(width=500, height=400)
                    # Hiển thị biểu đồ trong ứng dụng Streamlit
                    st.plotly_chart(fig)

                with col4:
                    # Debt to Equity Ratio
                    # Vẽ biểu đồ đường sử dụng Plotly Express
                    dfv['Debt to Equity Ratio'] = dfv['Debt to Equity Ratio'].round(2)
                    fig = px.line(dfv, x, y='Debt to Equity Ratio', title='Debt to Equity Ratio', markers='o',
                                  line_shape='spline', text='Debt to Equity Ratio')
                    fig.update_traces(textposition='top center')
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    # Thay đổi màu của đường
                    fig.update_traces(line=dict(color='dodgerblue'))
                    fig.update_layout(width=500, height=400)
                    # Hiển thị biểu đồ trong ứng dụng Streamlit
                    st.plotly_chart(fig)

                    # Current Ratio
                    # Vẽ biểu đồ đường sử dụng Plotly Express
                    dfv['Current Ratio'] = dfv['Current Ratio'].round(2)
                    fig = px.line(dfv, x, y='Current Ratio', title='Current Ratio', markers='o', line_shape='spline',
                                  text='Current Ratio')
                    fig.update_traces(textposition='top center')
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    # Thay đổi màu của đường
                    fig.update_traces(line=dict(color='rosybrown'))
                    fig.update_layout(width=500, height=400)
                    # Hiển thị biểu đồ trong ứng dụng Streamlit
                    st.plotly_chart(fig)

            with f_score:
                st.subheader('F-Score')
                data00 = [
                    ('Time', *[year.date().strftime("%Y-%m-%d") for year in years[::-1]], 'TTM', 'Total'),
                    ('Revenue', '-', *[rv_scores[i] for i in range(len(rv_scores))], str(TTM_rv_score),
                     str(total_rv_score) + ' / ' + str(len(rv_scores) + 1)),
                    ('Net Income', '-', *[ni_scores[i] for i in range(len(ni_scores))], str(TTM_ni_score),
                     str(total_ni_score) + ' / ' + str(len(ni_scores) + 1)),
                    (
                    'Operating Cash Flow', '-', *[opcf_scores[i] for i in range(len(opcf_scores))], str(TTM_opcf_score),
                    str(total_opcf_score) + ' / ' + str(len(opcf_scores) + 1)),
                    ('Free Cash Flow', '-', *[fcf_scores[i] for i in range(len(fcf_scores))], str(TTM_fcf_score),
                     str(total_fcf_score) + ' / ' + str(len(fcf_scores) + 1)),
                    ('EPS', '-', *[eps_scores[i] for i in range(len(eps_scores))], str(TTM_eps_score),
                     str(total_eps_score) + ' / ' + str(len(eps_scores) + 1)),
                    ('ROE', '-', *[roe_scores[i] for i in range(len(roe_scores))], str(TTM_roe_score),
                     str(total_roe_score) + ' / ' + str(len(roe_scores) + 1)),
                    ('Current Ratio', '-', *[cr_scores[i] for i in range(len(cr_scores))], str(TTM_cr_score),
                     str(total_cr_score) + ' / ' + str(len(cr_scores) + 1)),
                    ('Debt to Equity Ratio', '-', *[der_scores[i] for i in range(len(der_scores))], str(TTM_der_score),
                     str(total_der_score) + ' / ' + str(len(der_scores) + 1)),
                    ('Accounts Receivable', '-', *[ar_scores[i] for i in range(len(ar_scores))], str(TTM_ar_score),
                     str(total_ar_score) + ' / ' + str(len(ar_scores) + 1)),
                ]

                df00 = pd.DataFrame(data00[1:], columns=data00[0])
                pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
                st.write(df00)

                st.subheader('Total F-Score: ' + str(total_score) + ' / ' + str((len(rv_scores) + 1) * 9))

                percentage = (total_score / ((len(rv_scores) + 1) * 9)) * 100
                st.subheader('Percentage: {:.2f} %'.format(percentage))

            with estimation:
                cola, colb = st.columns(2)
                with cola: 
                    st.subheader('Discount Rate for US Stocks')
                    risk_free1 = st.number_input("Risk Free Rate (%)", value=float(2.19))
                    market_risk1 = st.number_input("Average Market Risk Premium (%)", value=float(3.7))
                    capm1 = risk_free1 + mck.info['beta']*market_risk1
                    st.write('Discount Rate (CAPM): ' + str(round(capm1,3)) + '%')
                    bt = [0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
                    dc_us = {
                        'BETA': ['Less than 0.8', '1', '1.1', '1.2', '1.3', '1.4', '1.5', 'More than 1.6'],
                        'DISCOUNT RATE': [round((risk_free1 + num*market_risk1),2) for num in bt]
                    }
                    st.data_editor(
                        dc_us,
                        hide_index=True,
                    )
                with colb: 
                    st.subheader('Discount Rate for VN Stocks')
                    risk_free2 = st.number_input("Risk Free Rate (%)", value=float(4.0))
                    market_risk2 = st.number_input("Average Market Risk Premium (%)", value=float(7.1))
                    capm2 = risk_free2 + mck.info['beta']*market_risk2
                    st.write('Discount Rate (CAPM): ' + str(round(capm2,3)) + '%')
                    dc_vn = {
                        'BETA': ['Less than 0.8', '1', '1.1', '1.2', '1.3', '1.4', '1.5', 'More than 1.6'],
                        'DISCOUNT RATE': [round((risk_free2 + num*market_risk2),2) for num in bt]
                    }
                    st.data_editor(
                        dc_vn,
                        hide_index=True,
                    )    
                if mck.info['country'] == 'Vietnam':
                    capm = capm2
                else: capm = capm1
            with valuation:
                col5, col6 = st.columns(2)
                with col5:
                    # Current year
                    current_year = datetime.now().year
                    st.subheader("Current Year")
                    number0 = st.number_input("Current Year:", value=current_year, placeholder="Type a number...")

                    # debt
                    ttm_cr = TTM_bsheet['Total Debt'] if 'Total Debt' in TTM_bsheet else 0
                    st.subheader("Total Debt")
                    cr_debt = TTM_bsheet['Current Debt'] if 'Current Debt' in TTM_bsheet else 0
                    st.write('Current Debt: ' + str(round(cr_debt,2)))
                    lt_debt = TTM_bsheet['Long Term Debt'] if 'Long Term Debt' in TTM_bsheet else 0
                    st.write('Long Term Debt: ' + str(round(lt_debt,2)))
                    formatted_ttm_cr = "{:,.2f}".format(ttm_cr)
                    number5_str = st.text_input("Total Debt (Current Debt + Long Term Debt)", value=formatted_ttm_cr, placeholder="Type a number...")
                    number5 = float(number5_str.replace(',', ''))

                    # cash
                    cash = TTM_bsheet.loc['Cash Cash Equivalents And Short Term Investments']
                    st.subheader("Cash and Short Term Investments:")
                    formatted_cash = "{:,.2f}".format(cash)
                    number6_str = st.text_input("Cash and Short Term Investments:", value=formatted_cash,
                                                placeholder="Type a number...")
                    number6 = float(number6_str.replace(',', ''))

                    # Nhập growth rate
                    st.subheader("Growth Rate")
                    
                    growth_rate1_str = st.text_input('Growth rate Y1-5 (Type a number)', value="%")
                    if growth_rate1_str.strip() == "%":  # Kiểm tra nếu chuỗi chỉ là ký tự '%'
                        growth_rate1 = 0.0  # Gán giá trị mặc định cho trường hợp này
                    else:
                        growth_rate1 = float(growth_rate1_str.replace(',', '').replace('%', '')) / 100
                    
                    growth_rate2_str = st.text_input('Growth rate Y6-10 (Type a number)', value="%")
                    if growth_rate2_str.strip() == "%":  # Kiểm tra nếu chuỗi chỉ là ký tự '%'
                        growth_rate2 = 0.0  # Gán giá trị mặc định cho trường hợp này
                    else:
                        growth_rate2 = float(growth_rate2_str.replace(',', '').replace('%', '')) / 100

                    growth_rate3_str = st.text_input('Growth rate Y11-20 (Type a number)', value="%")
                    if growth_rate3_str.strip() == "%":  # Kiểm tra nếu chuỗi chỉ là ký tự '%'
                        growth_rate3 = 0.0  # Gán giá trị mặc định cho trường hợp này
                    else:
                        growth_rate3 = float(growth_rate3_str.replace(',', '').replace('%', '')) / 100


                with col6:
                    st.subheader("Currency Unit: " + str(mck.info['financialCurrency']))
                    # fcf
                    st.subheader("Free Cash Flow/Net Income/Operating Cash Flow")
                    display_options = ["Free Cash Flow", "Net Income", "Operating Cash Flow"]
                    selected_display_option = st.radio("Select display option:", display_options)

                    if selected_display_option == "Free Cash Flow":
                        free_cash_flow = TTM_cfs['Free Cash Flow']
                        formatted_free_cash_flow = "{:,.2f}".format(free_cash_flow)
                        number2_str = st.text_input("Free Cash Flow (current):", value=formatted_free_cash_flow,
                                                    placeholder="Type a number...")
                        number2 = float(number2_str.replace(',', ''))


                    elif selected_display_option == "Net Income":
                        net_income = TTM['Net Income']
                        formatted_net_income = "{:,.2f}".format(net_income)
                        number2_str = st.text_input("Net Income:", value=formatted_net_income,
                                                    placeholder="Type a number...")
                        number2 = float(number2_str.replace(',', ''))

                    elif selected_display_option == "Operating Cash Flow":
                        operating_cash_flow = TTM_cfs['Operating Cash Flow']
                        formatted_operating_cash_flow = "{:,.2f}".format(operating_cash_flow)
                        number2_str = st.text_input("Operating Cash Flow:", value=formatted_operating_cash_flow,
                                                    placeholder="Type a number...")
                        number2 = float(number2_str.replace(',', ''))

                    # shares
                    shares_outstanding = mck.info['sharesOutstanding']
                    st.subheader("Shares Outstanding:")
                    formatted_shares_outstanding = "{:,.2f}".format(shares_outstanding)
                    number7_str = st.text_input("Shares Outstanding:", value=formatted_shares_outstanding,
                                                placeholder="Type a number...")
                    number7 = float(number7_str.replace(',', ''))

                    # beta
                    beta_value = mck.info['beta']
                    st.subheader("Company Beta")
                    formatted_beta = "{:,.3f}".format(beta_value)
                    number1_str = st.text_input("Company Beta:", value=formatted_beta, placeholder="Type a number...")
                    number1 = float(number1_str.replace(',', '')) 
                    # Tính toán discount rate dựa trên giá trị beta
                    st.subheader(" Discount Rate ")
                    discount_rate_value = capm
                    # Tính toán discount rate tương ứng
                    formatted_discount_rate_value = "{:,.3f}".format(discount_rate_value)
                    # Hiển thị discount rate trên Streamlit
                    
                    number8_str = st.text_input('Discount Rate (%):', value=formatted_discount_rate_value,
                                                placeholder="Type a number...")
                    number8 = float(number8_str.replace(',', '').replace(' %', '')) 
               
                col7, col8 = st.columns(2)
                with col7:
                    # Creating the first table
                    data1 = {
                        'Operating Cash Flow/Free Cash Flow/Net Income': [number2],
                        'Growth rate (Y 1-5)': growth_rate1,
                        'Growth rate (Y 6-10)': growth_rate2,
                        'Growth rate (Y 11-20)': growth_rate3,
                        'Discount rate': number8,
                        'Current year': number0
                    }

                    table1 = pd.DataFrame(data=data1)

                    # Creating the second table with calculations based on the first table
                    years = [
                        ((table1['Current year'][0]) + i)
                        for i in range(21)
                    ]
                    cash_flows = [
                        (table1['Operating Cash Flow/Free Cash Flow/Net Income'][0] * (
                                (1 + table1['Growth rate (Y 1-5)'][0]) ** i)) if i <= 5
                        else ((table1['Operating Cash Flow/Free Cash Flow/Net Income'][0] * (
                                (1 + table1['Growth rate (Y 1-5)'][0]) ** 5)) * (
                                      (1 + table1['Growth rate (Y 6-10)'][0]) ** (i - 5))) if 6 <= i <= 10
                        else ((table1['Operating Cash Flow/Free Cash Flow/Net Income'][0] * (
                                (1 + table1['Growth rate (Y 1-5)'][0]) ** 5)) * (
                                      (1 + table1['Growth rate (Y 6-10)'][0]) ** 5) * (
                                      (1 + table1['Growth rate (Y 11-20)'][0]) ** (i - 10)))
                        for i in range(21)
                    ]

                    discount_factors = [(1 / ((1 + table1['Discount rate'][0]/100) ** i)) for i in range(21)]
               
                    discounted_values = [cf * df for cf, df in zip(cash_flows, discount_factors)]

                    data2 = {
                        'Year': years[1:],
                        'Cash Flow': cash_flows[1:],
                        'Discount Factor': discount_factors[1:],
                        'Discounted Value': discounted_values[1:]
                    }

                    table2 = pd.DataFrame(data=data2)
                    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
                    table2['Year'] = table2['Year'].astype(str).str.replace(',', '')
                    st.subheader('Discounted Cash Flow')
                    st.write(table2)
                with col8:
                    # Tính Intrinsic Value
                    total_discounted_value = sum(discounted_values)
                    intrinsic_value = sum(discounted_values) / number7
                    debt_per_share = number5 / number7
                    cash_per_share = number6 / number7
                    st.subheader('Value')
                    data3 = pd.DataFrame({
                        "STT": [1, 2, 3, 4],
                        "Index": ['PV of 20 yr Cash Flows', 'Intrinsic Value before cash/debt', 'Debt per Share',
                                  'Cash per Share'],
                        "Value": [total_discounted_value, intrinsic_value, debt_per_share, cash_per_share],
                    })
                    st.data_editor(data3, hide_index=True)

                    final_intrinsic_value = intrinsic_value - debt_per_share + cash_per_share
                    st.subheader(f"Final Intrinsic Value per Share: {final_intrinsic_value:,.2f}")

                # Chart
                cash_flow = table2['Cash Flow']
                discounted_value = table2['Discounted Value']
                years = table2['Year']
                columns_to_plot = ['Cash Flow', 'Discounted Value']
                fig = px.line(table2, x=years, y=columns_to_plot,
                              title='Intrinsic Value Calculator (Discounted Cash Flow Method 10 years)',
                              labels={'value': 'Value', 'variable': 'Legend'},
                              height=500, width=1100, markers='o')
                fig.update_xaxes(fixedrange=True)

                # Thay đổi các chú thích trên trục x
                fig.update_xaxes(
                    tickvals=years[0:]
                )
                fig.update_xaxes(title_text="Time")
                st.plotly_chart(fig)

            with guru:

                ave_debt = TTM_bsheet3/4
                cost_of_debt = TTM['Interest Expense']/ave_debt['Total Debt']
                tax_rate = TTM['Tax Provision']/TTM['Pretax Income']
                market_cap = mck.info['marketCap'] if 'marketCap' in mck.info else mck.basic_info['marketCap']
                wacc = (market_cap/(market_cap+ave_debt['Total Debt'])) * discount_rate_value + (ave_debt['Total Debt']/(market_cap+ave_debt['Total Debt'])) * cost_of_debt *(1-tax_rate)
            
                #roic
                invest_cap_dec = TTM_bsheet['Total Assets'] - TTM_bsheet['Payables And Accrued Expenses'] - (TTM_bsheet['Cash Cash Equivalents And Short Term Investments'] 
                                - max(0,(TTM_bsheet['Current Liabilities'] - TTM_bsheet['Current Assets'] + TTM_bsheet['Cash Cash Equivalents And Short Term Investments']))) if 'Payables And Accrued Expenses' in TTM_bsheet else TTM_bsheet['Total Assets'] - (TTM_bsheet['Cash Cash Equivalents And Short Term Investments'] 
                                - max(0,(TTM_bsheet['Current Liabilities'] - TTM_bsheet['Current Assets'] + TTM_bsheet['Cash Cash Equivalents And Short Term Investments'])))
                invest_cap_sep = TTM_bsheet2['Total Assets'] - TTM_bsheet2['Payables And Accrued Expenses'] - (TTM_bsheet2['Cash Cash Equivalents And Short Term Investments'] 
                                - max(0,(TTM_bsheet2['Current Liabilities'] - TTM_bsheet2['Current Assets'] + TTM_bsheet2['Cash Cash Equivalents And Short Term Investments']))) if 'Payables And Accrued Expenses' in TTM_bsheet2 else TTM_bsheet2['Total Assets'] - (TTM_bsheet2['Cash Cash Equivalents And Short Term Investments'] 
                                - max(0,(TTM_bsheet2['Current Liabilities'] - TTM_bsheet2['Current Assets'] + TTM_bsheet2['Cash Cash Equivalents And Short Term Investments'])))
                roic = TTM['Operating Income'] * (1-tax_rate) / (1/2 * (invest_cap_dec + invest_cap_sep)) 

                col1, col2 = st.columns(2)
                with col1:

                    st.subheader('Profitability Rank: ' + str(profitability_score) + '/' + '10')
                    data_profitability = pd.DataFrame(
                        {
                            "STT": [1, 2, 3, 4, 5, 6, 7, 8],
                            "Index": ['	Gross Margin %', '	Operating Margin %', '	Net Margin %',
                                      '	FCF Margin %',
                                      '	ROE %',
                                      '	ROA %', '	ROC (Joel Greenblatt) %', '	ROCE %'],
                            "-": [gr_margin_list, op_margin_list, net_margin_list, fcf_margin_list, roe_margin_list,
                                  roa_margin_list,
                                  roc_margin_list, roce_margin_list],
                            "Current": [gr_margin, op_margin, net_margin, fcf_margin, roe_margin, roa_margin,
                                        roc_margin,
                                        roce_margin],
                            "Vs History": [gr_margin_values, op_margin_values, net_margin_values, fcf_margin_values,
                                           roe_margin_values, roa_margin_values,
                                           roc_margin_values, roce_margin_values],
                        }
                    )
                    st.data_editor(
                        data_profitability,
                        column_config={
                            "-": st.column_config.BarChartColumn(
                                "-", width="small",
                            ),
                            "Vs History": st.column_config.ProgressColumn(
                                "Vs History",
                            ),
                        },
                        hide_index=True,
                    )

                    st.subheader('Financial Strength: ' + str(financial_score) + '/' + '10')
                    
                    data_financial = pd.DataFrame(
                        {
                            "STT": [1, 2, 3, 4, 5],
                            "Index": ['Cash to Debt', 'Equity to Assets', 'Debt to Equity', 'Debt to EBITDA',
                                      'Interest Coverage'],
                            "-": [cash_debt_list, equity_asset_list, debt_equity_list, debt_ebitda_list,
                                  interest_coverage_list],
                            "Current": [cash_debt, equity_asset, debt_equity, debt_ebitda, interest_coverage],
                            "Vs History": [cash_debt_values, equity_asset_values, debt_equity_values,
                                           debt_ebitda_values, interest_coverage_values],
                        }
                    )
                    st.data_editor(
                        data_financial,
                        column_config={
                            "-": st.column_config.BarChartColumn(
                                "-", width="small",
                            ),
                            "Vs History": st.column_config.ProgressColumn(
                                "Vs History",
                            ),
                        },
                        hide_index=True,
                    )

                    st.subheader('Score')
                    data_score = pd.DataFrame(
                        {
                            "STT": [1, 2, 3, 4, 5],
                            "Index": ['Altman Z-Score', 'Beneish M-Score', 'Piotroski F-Score (Scale of 9)','WACC','ROIC'],
                            "Value": [altmanz_score, m, piotroski, wacc, roic],
                        }
                    )
                    st.data_editor(data_score, hide_index=True)

                    st.write('Conclusion: ')
                    if altmanz_score <=1.81:
                        st.write('1. Altman Z-Score = ' + str(round(altmanz_score,2)) + ': Distress Zone - High Likelihood of Bankruptcy')
                    elif 1.81 < altmanz_score <2.99:
                        st.write('1. Altman Z-Score = ' + str(round(altmanz_score,2)) + ':  Grey - Moderate Likelihood of Bankruptcy')
                    else:
                        st.write('1. Altman Z-Score = ' + str(round(altmanz_score,2)) + ': Safe Zone - Low Likelihood of Bankruptcy')
                    
                    if m <=-1.78:
                        st.write('2. Beneish M-Score = ' + str(round(m,2)) + ': Unlikely to be a manipulator')
                    else:
                        st.write('2. Beneish M-Score = ' + str(round(m,2)) + ': Likely to be a manipulator')

                with col2:
                    st.subheader('Liquidity Ratio: ' + str(liquidity_score) + '/' + '10')
                    data_liquidity = pd.DataFrame(
                        {
                            "STT": [1, 2, 3, 4, 5, 6],
                            "Index": ['Current Ratio', 'Quick Ratio', 'Cash Ratio', 'Days Inventory',
                                      'Days Sales Outstanding',
                                      'Days Payable'],
                            "-": [cr_list, qr_list, car_list, dio_list, dso_list, dp_list],
                            "Current": [cr_ratio, qr_ratio, car_ratio, dio_ratio, dso_ratio, dp_ratio],
                            "Vs History": [cr_values2, qr_values, car_values, dio_values, dso_values, dp_values],
                        }
                    )
                    st.data_editor(
                        data_liquidity,
                        column_config={
                            "-": st.column_config.BarChartColumn(
                                "-", width="small",
                            ),
                            "Vs History": st.column_config.ProgressColumn(
                                "Vs History",
                            ),
                        },
                        hide_index=True,
                    )

                    st.subheader('GF Values')
                    data_GF_Value = pd.DataFrame(
                        {
                            "STT": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                            "Index": ['PE Ratio', 'PS Ratio', 'PB Ratio', 'Price-to-tangible-book Ratio',
                                      'Price-to-Free-Cash_Flow Ratio',
                                      'EV-to-EBIT', 'EV-to-EBITDA', 'EV-to-Revenue', 'EV-to-FCF',
                                      'Price-to-Net-Current-Asset-Value', 'Earnings Yields (Greenblatt) %',
                                      'FCF Yield %'],
                            "Current": [PE_ratio, PS_ratio, PB_ratio, Price_to_TBV, price_to_FCF, EV_to_EBIT,
                                        EV_to_EBITDA,
                                        EV_to_Revenue, EV_to_FCF, Price_to_Net_CAV, EarningsYields, FCFYield],
                        },
                    )
                    st.data_editor(
                        data_GF_Value,
                        hide_index=True,
                    )

                    st.subheader('Dividend & Buy Back')
                    data_dividend = pd.DataFrame(
                        {
                            "STT": [1, 2, 3, 4],
                            "Index": ['Dividend Yield', 'Dividend Payout Ratio', '5-Year Yield-on-Cost',
                                      'Forward Dividend Yield'],
                            "Current": [div_ratio, pr_ratio, five_years_ratio, forward_ratio],
                        }
                    )

                    # Hiển thị DataFrame đã chỉnh sửa trong st.data_editor
                    st.data_editor(data_dividend, hide_index=True)
                # Revenue, Net Income, EBITDA
                col9, col10 = st.columns(2)
                with col9:

                    # Cash-debt
                    columns_to_plot2 = ['Cash', 'Debt']
                    x = ['['] + dfv['Time'] + [']']
                    # Plot grouped bar chart
                    fig = px.bar(dfv, x, y=columns_to_plot2,
                                 labels={'value': 'Value', 'variable': 'Legend'},
                                 barmode='group')

                    # Add text on top of each bar
                    for col in columns_to_plot2:
                        new_values = dfv[col] / 1e9
                        fig.update_traces(text=new_values.apply(lambda x: f'{x:.2f}B'), textposition='outside',
                                          selector=dict(name=col))
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    fig.update_layout(legend_title_text=None)
                    fig.update_layout(legend=dict(orientation='h', x=0.5, y=1.2), width=500, height=400)

                    # Display the chart in Streamlit app
                    st.plotly_chart(fig)

                    # Stockholders equity vs total asset
                    columns_to_plot3 = ['Stockholders Equity', 'Total Assets']
                    x = ['['] + dfv['Time'] + [']']
                    # Plot grouped bar chart
                    fig = px.bar(dfv, x, y=columns_to_plot3,
                                 labels={'value': 'Value', 'variable': 'Legend'},
                                 barmode='group')

                    # Add text on top of each bar
                    for col in columns_to_plot3:
                        new_values = dfv[col] / 1e9
                        fig.update_traces(text=new_values.apply(lambda x: f'{x:.2f}B'), textposition='outside',
                                          selector=dict(name=col))
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    fig.update_layout(legend_title_text=None)
                    fig.update_layout(legend=dict(orientation='h', x=0.5, y=1.2), width=520, height=400)
                    # Display the chart in Streamlit app
                    st.plotly_chart(fig)

                with col10:
                    columns_to_plot1 = ['Revenue', 'Net Income', 'EBITDA']
                    x = ['['] + dfv['Time'] + [']']
                    # Plot grouped bar chart
                    fig = px.bar(dfv, x, y=columns_to_plot1,
                                 labels={'value': 'Value', 'variable': 'Legend'},
                                 barmode='group')

                    # Add text on top of each bar
                    for col in columns_to_plot1:
                        new_values = dfv[col] / 1e9
                        fig.update_traces(text=new_values.apply(lambda x: f'{x:.2f}B'), textposition='outside',
                                          selector=dict(name=col))
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    fig.update_layout(legend_title_text=None)
                    fig.update_layout(width=600, height=400, legend=dict(orientation='h', x=0.5, y=1.2))
                    # Display the chart in Streamlit app
                    st.plotly_chart(fig)

                    columns_to_plot4 = ['Operating Cash Flow', 'Free Cash Flow', 'Net Income',
                                        'Cash Flow for Dividends', 'Stock Based Compensation']
                    x = ['['] + dfv['Time'] + [']']
                    # Plot grouped bar chart
                    fig = px.bar(dfv, x, y=columns_to_plot4,
                                 labels={'value': 'Value', 'variable': 'Legend'},
                                 barmode='group')

                    # Add text on top of each bar
                    for col in columns_to_plot4:
                        new_values = dfv[col] / 1e9
                        fig.update_traces(text=new_values.apply(lambda x: f'{x:.2f}B'), textposition='outside',
                                          selector=dict(name=col))
                    fig.update_xaxes(fixedrange=True, title_text='')
                    fig.update_yaxes(fixedrange=True, title_text="")
                    fig.update_layout(legend=dict(orientation='h', y=1.35), width=600, height=400)
                    fig.update_layout(legend_title_text=None)
                    
                    # Display the chart in Streamlit app
                    st.plotly_chart(fig)
except KeyError:
    st.caption('The information is not sufficient for evaluation')