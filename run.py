import pandas as pd
from fontTools.ttLib.tables.S_V_G_ import doc_index_entry_format_0

from Net import Net
import numpy as np
from Requests import Requests


def update(year):
    print(f"Year {year} update:")
    if year > 0:
        requests.update_traffic()
    total_reqs, total_traffic = requests.process_requests_by_year()
    return requests.rb_blocking, total_reqs, requests.tb_blocking, total_traffic


if __name__ == '__main__':
    requests = Requests()
    requests.init_classical_channels()
    # df = requests.tmp_qkd_ch()
    # df.to_csv(f'Outputs/qkd_chs_info.csv')
    # rep = [25600, 51200, 102400, 512000, 2120000]
    rep = 1000
    num_of_requests = 2000
    data_blocking = pd.DataFrame(np.zeros((num_of_requests, rep)))
    methods = ['single','bitwise', 'concatenate']
    method = methods[0]
    for i in range(rep):
        # df_skr = requests.analyze_compromised_secrecy_co(num_of_requests, i)
        df_skr, df_blocking = requests.process_requests(num_of_requests=num_of_requests, method=method)
        data_blocking.iloc[:, i] = df_blocking
        # df_skr.to_csv(f'Outputs/skr_{i}_{method}.csv')
        requests.reset_net()
    data_blocking.to_csv('Outputs/blocking_info.csv')
    #     df[df.columns[i]] = df_i[df_i.columns[-2]].astype(float)
    #     df[df.columns[rep+i]] = df_i[df_i.columns[-1]].astype(float)
    # mean_quantum = df[df.columns[0:rep]].mean(axis=1)
    # mean_classic = df[df.columns[rep:2 * rep]].mean(axis=1)
    # final_df = pd.concat([mean_classic, mean_quantum], axis=1)
    # final_df.to_csv(f'Outputs/blocking_info.csv')


    # df = requests.skr_comparison()
    # df.to_csv('Outputs/skr_comparison.csv')

    # number_of_years = 100
    # for year in range(0, number_of_years):
    #     blocked_req, total_reqs, blocked_traffic, total_traffic = update(year)
    #     lst_year = [blocked_req, total_reqs, blocked_traffic, total_traffic]
    #     data.append(lst_year)
    #     print("Blocked Requests: ", blocked_req, "Blocked Traffic: ", blocked_traffic)
    # row_indices = ['year{}'.format(i) for i in range(number_of_years)]
    # row_indices = ['year{}'.format(i) for i in range(number_of_requests)]
    # column_names = ['blocked requests', 'total requests', 'blocked traffic', 'total traffic']
    # blocking_df = pd.DataFrame(data, index=row_indices, columns=column_names)
    # blocking_df.to_csv('Outputs/blocking_info.csv')
