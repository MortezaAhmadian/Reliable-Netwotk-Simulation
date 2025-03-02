import copy
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np
from numpy.array_api import floor

from Net import Net
import networkx as nx
import json
import random
import pandas as pd
import logging
from networkx.algorithms.simple_paths import shortest_simple_paths
from measurement_Framework.coexistance import coexistance

def _Reverse_lst(lst):
    new_lst = lst[::-1]
    return new_lst


def _init_requests():
    with open('config/requests.json', 'r') as f:
        return json.load(f)


def _load_connection_profile():
    with open('config/All_connections_Profile.json', 'r') as f:
        return json.load(f)


class Requests:
    def __init__(self):
        self.logger = logging.getLogger("Request")
        self.rbs_net = None
        self.tbs_net = Net()
        self.qkey_length = 256
        self.rb_blocking = 0  # number of requests
        self.tb_blocking = 0  # GB
        self.requests = _init_requests()
        self._connection_profile = _load_connection_profile()
        # self.rbs_net.draw_graph_topology()
        self.init_classical_channels()

    def init_classical_channels(self):
        self.rbs_net = Net()
        self.rb_blocking = 0

    def reset_net(self):
        self.rbs_net.reset_network()

    def _find_path(self, source, target):
        try:
            path = nx.shortest_path(self.rbs_net.G, source=source, target=target, weight='distance')
            return path
        except nx.NetworkXNoPath:
            return None

    def _get_total_distance(self, path):
        total = 0
        for u, v in zip(path[:-1], path[1:]):
            total += self.rbs_net.G.edges[u, v]['distance']
        return total

    def _find_top_k_paths(self, source, target, k):
        try:
            paths = list(shortest_simple_paths(self.rbs_net.G, source, target, weight='distance'))
            distances = []
            for path in paths:
                distances.append(self._get_total_distance(path))
            return paths[:k], distances[:k]
        except nx.NetworkXNoPath:
            return []

    def _find_k_node_disjoint_paths(self, source, target, k):
        disjoint_paths = []
        disjoint_distance = []
        temp_graph = self.rbs_net.G.copy()

        # Track if direct link between source and destination exists and was used
        direct_link_used = False
        if temp_graph.has_edge(source, target):
            direct_link_used = True

        for i in range(k):
            try:
                # Find the shortest path in the current graph
                path = nx.shortest_path(temp_graph, source=source, target=target, weight='distance')
                path_distance = sum(temp_graph[u][v]['distance'] for u, v in zip(path[:-1], path[1:]))
                # print(f"Path {i + 1}: {path} with total distance: {path_distance:.2f} km")
                disjoint_paths.append(path)
                disjoint_distance.append(path_distance)

                # If direct link was used, remove it after first use
                if len(path) == 2 and direct_link_used:
                    # print(f"Direct link {source}-{target} used, removing it for future paths.")
                    temp_graph.remove_edge(source, target)

                # Remove intermediate nodes from the graph (keep source and target)
                for node in path[1:-1]:  # Exclude source and target nodes
                    temp_graph.remove_node(node)

            except nx.NetworkXNoPath:
                # print(f"No further node-disjoint paths found after {i} paths.")
                break

        return disjoint_paths, disjoint_distance


    def _check_shpath(self, paths, distances):
        check_list = []
        for i in range(len(paths)):
            selected_path = paths[i].copy()
            if paths[i][0] > paths[i][-1]:
                selected_path = _Reverse_lst(selected_path)
            source = selected_path[0]
            destination = selected_path[-1]
            for c in range(len(self._connection_profile)):
                if (self._connection_profile[c]["source"] == source and
                        self._connection_profile[c]["target"] == destination):
                    if distances[i] == self._connection_profile[c]["distances"][i]:
                        check_list.append(True)
                    else:
                        check_list.append(False)
                    break
        return check_list

    def _path_competition(self, paths, net_obj, requested_traffic, distances):
        k = 0
        ch_freq = []
        path_freq = []
        path_gsnr = []
        path_dist = []
        path_capa = []
        for path in paths:
            channels_capacity = net_obj.find_capacity_channels(path, k)
            GSNR = list(net_obj.get_GSNR(path, k))
            req = copy.copy(requested_traffic)
            if sum(channels_capacity) > requested_traffic:
                ch = 0
                path_capa.append(requested_traffic)
                while req > 0 and ch < 268:
                    if channels_capacity[ch] > 0:
                        if req > channels_capacity[ch]:
                            req -= channels_capacity[ch]
                            cap = channels_capacity[ch]
                        else:
                            cap = req
                            req = 0
                        ch_freq.append([k, ch, cap])
                    ch += 1
                available_ch_lst = [ch_freq[i][1:] for i in range(len(ch_freq)) if ch_freq[i][0] == k]
                path_freq.append(available_ch_lst[0][0])
                path_gsnr.append(GSNR[available_ch_lst[0][0]])
                path_dist.append(distances[k])
            else:
                path_capa.append(sum(channels_capacity))
                path_freq.append(float('inf'))
                path_gsnr.append(float('inf'))
                path_dist.append(float('inf'))
            k += 1
        return ch_freq, path_capa, path_freq, path_gsnr, path_dist

    def _serve_secret_keys(self, path, requested_skr):
        distance = 0
        NTN = 0
        for u, v in zip(path[:-1], path[1:]):
            req = copy.copy(requested_skr)
            if self.rbs_net.G[u][v]['dedicated_skr'] > req:
                self.rbs_net.G[u][v]['dedicated_skr'] -= req
                distance += self.rbs_net.G[u][v]['distance']
                NTN += np.ceil(distance/100) - 1
            else:
                return 1, 0, NTN, distance
        return 0, (5e-6 * distance) + (2e-8 * NTN), NTN, distance

    def _rb_serve_traffic(self, paths, distances, requested_traffic, number):
        used_channels = [0] * 268
        ch_freq, _, path_freq, path_gsnr, path_dist = self._path_competition(paths, self.rbs_net, requested_traffic,
                                                                             distances)
        min_ch_lst = [i for i, x in enumerate(path_freq) if x == min(path_freq)]
        max_gsnr = max([path_gsnr[i] for i, x in enumerate(path_gsnr) if i in min_ch_lst])
        selected_path = [i for i, x in enumerate(path_gsnr) if x == max_gsnr]
        if len(selected_path) > 1:
            min_lst = [path_dist[i] for i, x in enumerate(path_dist) if i in selected_path]
            selected_path = int(np.argmin(min_lst))
        else:
            selected_path = selected_path[0]
        if path_freq[selected_path] < 1000:
            for ch in ch_freq:
                if ch[0] == selected_path:
                    used_channels[ch[1]] = 1
                    self.rbs_net.allocate_channels(paths[selected_path], ch[1], ch[2])
        else:
            self.rb_blocking += 1
            print("Request {} is blocked".format(number))
        return used_channels, selected_path

    def _tb_serve_traffic(self, paths, distances, requested_traffic, number):
        used_channels = [0] * 268
        ch_freq, path_capa, path_freq, path_gsnr, path_dist = self._path_competition(paths, self.tbs_net,
                                                                                     requested_traffic, distances)
        served_traffic = 0
        path_freq = [[i,x] for i, x in enumerate(path_freq) if path_capa[i] == max(path_capa)]
        min_freq = min([path_freq[i][1] for i in range(len(path_freq))])
        min_ch_lst = [path_freq[i][0] for i in range(len(path_freq)) if path_freq[i][1] == min_freq]
        max_gsnr = max([path_gsnr[i] for i, x in enumerate(path_gsnr) if i in min_ch_lst])
        selected_path = [i for i, x in enumerate(path_gsnr) if x == max_gsnr]
        if len(selected_path) > 1:
            min_lst = [path_dist[i] for i, x in enumerate(path_dist) if i in selected_path]
            selected_path = int(np.argmin(min_lst))
        else:
            selected_path = selected_path[0]
        if path_freq[selected_path][1] < 1000:
            for ch in ch_freq:
                if ch[0] == selected_path:
                    used_channels[ch[1]] = 1
                    self.tbs_net.allocate_channels(paths[selected_path], ch[1], ch[2])
                    served_traffic += ch[2]
        if requested_traffic - served_traffic > 0:
            self.tb_blocking += (requested_traffic - served_traffic)
            print("{} GB traffic of request {} is blocked".format((requested_traffic - served_traffic), number))
        return used_channels, selected_path

    def update_traffic(self):
        self.requests = _init_requests()
        for request in self.requests:
            # if random.random() < 0.5:  # 20% of the requests
            increase_percent = random.uniform(0.2, 0.4)
            request['traffic'] = int(request['traffic'] * increase_percent)
            request['total_traffic'] += request['traffic']
            self.rb_blocking = 0  # number of requests
            self.tb_blocking = 0  # GB
            # else:  # 80% of the requests
            #     increase_percent = random.uniform(0.1, 1.0)
            #     request['traffic'] += int(request['traffic'] * increase_percent)
            # request['traffic'] = max(request['traffic'], 0)  # Ensure traffic doesn't go negative

    def _qkd_channel(self, path):
        qber_lst = []
        skr_lst = []
        all_dis = 0
        for u, v in zip(path[:-1], path[1:]):
            distance = self.rbs_net.G.edges[u, v]['distance']
            c = np.ceil(distance/100)
            obj = coexistance(path[0], path[-1], distance/c)
            center_freq, slot_widths, signals, baud_rates = obj.channels_init()
            sprs_value_seo, spectral_info_out, power_out = obj.setup_channels(center_freq, slot_widths, signals,
                                                                              baud_rates)
            skr, qber, _ = obj.get_sek_qber(sprs_value_seo)
            qber_lst.append(sum(qber)/len(qber))
            skr_lst.append(sum(skr))
            all_dis += distance
        return max(qber_lst), min(skr_lst), all_dis

    def process_requests_by_year(self):
        global used_channels_rb, used_channels_tb
        data = np.zeros((len(self.requests), 271))
        ch_col = ['ch{}'.format(i) for i in range(268)]
        columns = ['source', 'destination', 'traffic']
        columns.extend(ch_col)
        df_info = pd.DataFrame(data, columns=columns)
        i = 0
        total_requests = 0
        total_traffic = 0
        for request in self.requests:
            total_requests += 1
            total_traffic += request['traffic']
            source = request['source']
            destination = request['destination']
            requested_traffic = request['traffic']
            paths, distances = self._find_top_k_paths(source, destination, 3)
            checks = self._check_shpath(paths, distances)
            if all(checks):
                used_channels_rb = self._rb_serve_traffic(paths, distances, requested_traffic, i)
                used_channels_tb = self._tb_serve_traffic(paths, distances, requested_traffic, i)
                print(f"Path from {source} to {destination}: {paths} with requested traffic: {requested_traffic}")
            else:
                print(f"No path found from {source} to {destination} with requested traffic: {requested_traffic}")
            df_info.iloc[i, 0] = source
            df_info.iloc[i, 1] = destination
            df_info.iloc[i, 2] = request['total_traffic']
            df_info.iloc[i, 3:] = used_channels_rb
            i += 1

        edge_data_rb = self.rbs_net.return_edge_data()
        edge_data_tb = self.tbs_net.return_edge_data()

        # Create a DataFrame with the edge information and weights
        columns = ['source', 'target'] + [f'channel_{i + 1}' for i in range(268)] + [f'traffic_ch_{i + 1}' for i in
                                                                                     range(268)]
        df_rb = pd.DataFrame(edge_data_rb, columns=columns)
        df_rb.to_csv('Outputs/network_matrix_request_based.csv')
        df_tb = pd.DataFrame(edge_data_tb, columns=columns)
        df_tb.to_csv('Outputs/network_matrix_traffic_based.csv')
        df_info.to_csv('Outputs/traffic_information.csv')
        return total_requests, total_traffic


    def analyze_compromised_secrecy_co(self, num_of_requests, req_skr):
        df_skr = []
        for i in range(num_of_requests):
            distances = [150 + (i * 100), 5000, 7000]
            NTNs = [math.ceil(i/100) - 1 for i in distances]
            # req_skr = 2120000  # + (i * 500)
            delays = []
            for j, path in enumerate(distances):
                delay = (5e-6 * distances[j]) + (2e-8 * NTNs[j])
                delays.append(delay)
            if max(distances) >= 1000:
                achieved_skr = self.qkey_length / (
                        (self.qkey_length / req_skr) + max(delays) - max(0.005, min(delays)))
                if min(delays) > 0.005:
                    diff_delay = [x - min(delays) for x in delays]
                else:
                    delays = [i if i > 0.005 else 0.005 for i in delays]
                    diff_delay = [x - 0.005 for x in delays]
                max_diff = max(diff_delay)
                buffered_kr = [(max_diff - i) * achieved_skr for i in diff_delay]
            else:
                achieved_skr = req_skr
                buffered_kr = [0, 0]

            key_secrecy = [1 - ((i / self.qkey_length) * 0.1) for i in buffered_kr]
            S = math.prod(key_secrecy)
            paths_secrecy = 1 - math.prod([1 - (0.99 ** i) for i in NTNs])
            compromised_secrecy = 1 - (S * paths_secrecy)
            if len(distances) == 3:
                [dis1, dis2, dis3] = distances
                [NTN1, NTN2, NTN3] = NTNs
                [buffered_1, buffered_2, _] = buffered_kr
            else:
                [dis1, dis2] = distances
                [NTN1, NTN2] = NTNs
                [buffered_1, buffered_2] = buffered_kr
                [dis3, NTN3] = ['-', '-']

            df_skr.append({
                'requested_skr': req_skr,
                'distance_1': dis1,
                'distance_2': dis2,
                'distance_3': dis3,
                'NTN_1': NTN1,
                'NTN_2': NTN2,
                'NTN_3': NTN3,
                'achieved_skr': achieved_skr,
                'buffered_1': buffered_1,
                'buffered_2': buffered_2,
                'Num_of_paths': len(distances),
                'compromised_secrecy': compromised_secrecy
            })
        return pd.DataFrame(df_skr)

    def analyze_compromised_secrecy_bw(self, num_of_requests, req_skr):
        df_skr = []
        for i in range(num_of_requests):
            distances = [150 + (i * 100), 5000, 7000]
            NTNs = [math.ceil(d/100) - 1 for d in distances]
            # req_skr = 25600  # + (i * 500)
            delays = []
            for j, path in enumerate(distances):
                delay = (5e-6 * distances[j]) + (2e-8 * NTNs[j])
                delays.append(delay)
            if max(distances) >= 1000:
                achieved_skr = self.qkey_length / (
                        (self.qkey_length / req_skr) + max(delays) - max(0.005, min(delays)))
                if min(delays) > 0.005:
                    diff_delay = [x - min(delays) for x in delays]
                else:
                    delays = [i if i > 0.005 else 0.005 for i in delays]
                    diff_delay = [x - 0.005 for x in delays]
                max_diff = max(diff_delay)
                buffered_kr = [(max_diff - i) * achieved_skr for i in diff_delay]
            else:
                achieved_skr = req_skr
                buffered_kr = [0, 0]

            key_secrecy = [1 - ((i / self.qkey_length) * 0.1) for i in buffered_kr]
            paths_secrecy = [1 - (key_secrecy[i] * (0.99 ** j)) for i, j in enumerate(NTNs)]
            compromised_secrecy = math.prod(paths_secrecy)
            if len(distances) == 3:
                [dis1, dis2, dis3] = distances
                [NTN1, NTN2, NTN3] = NTNs
                [buffered_1, buffered_2, _] = buffered_kr
            else:
                [dis1, dis2] = distances
                [NTN1, NTN2] = NTNs
                [buffered_1, buffered_2] = buffered_kr
                [dis3, NTN3] = ['-', '-']

            df_skr.append({
                'requested_skr': req_skr,
                'distance_1': dis1,
                'distance_2': dis2,
                'distance_3': dis3,
                'NTN_1': NTN1,
                'NTN_2': NTN2,
                'NTN_3': NTN3,
                'achieved_skr': achieved_skr,
                'buffered_1': buffered_1,
                'buffered_2': buffered_2,
                'Num_of_paths': len(distances),
                'compromised_secrecy': compromised_secrecy
            })
        return pd.DataFrame(df_skr)

    def process_requests(self, num_of_requests, method):
        # columns = ['source', 'destination', 'traffic', 'blocked_request', 'blocked_traffic', 'selected_path_r',
        #            'selected_path_t', 'qkd_path_1_blocked','qkd_path_2_blocked', 'qkd_path_3_blocked', 'cum', 'rate']
        # data = np.zeros((num_of_requests, 2))
        df_skr = []
        df_blocking = []
        # df_info = pd.DataFrame(data) # , columns=columns)

        cum_qkd_blocked = 0
        cum_paths_count = 0
        blocking = 0
        for i in range(num_of_requests):
            nodes = [i+1 for i in range(14)]
            source = random.choice(nodes)
            destination = random.choice([node for node in nodes if node != source])
            # paths, distances = self._find_top_k_paths(source, destination, 3)
            # paths = [paths[2]]
            # distances = [distances[2]]
            paths, distances = self._find_k_node_disjoint_paths(source, destination, [1 if method == 'single' else 3][0])
            checks = [True, True, True] # self._check_shpath(paths, distances)
            traffic = random.choice([(i*50)+100 for i in range(11)])
            achieved_skr = None
            buffered_kr = None
            distances = []
            NTNs = []
            req_skr = 51200
            qkd_traffic_per_path = [req_skr / len(paths) if method == 'concatenate' else req_skr][0]
            if all(checks):
                delays = []
                # _, selected_path_r = self._rb_serve_traffic(paths, distances, traffic, i)
                # _, selected_path_t = self._tb_serve_traffic(paths, distances, traffic, i)
                if random.randint(1, 10) <= 1:
                    for j, path in enumerate(paths):
                        blocked_qkd_request, delay, NTN, distance = self._serve_secret_keys(path, qkd_traffic_per_path)
                        if blocked_qkd_request:
                            blocking += 1
                        cum_qkd_blocked += blocked_qkd_request
                        delays.append(delay)
                        NTNs.append(NTN)
                        distances.append(distance)
                    cum_paths_count += len(paths)
                    final_skr = [req_skr - (blocking*req_skr/len(paths)) if method == 'concatenate' else req_skr][0]
                    if final_skr > 0:
                        if max(distances) >= 1000 and max(delays) > 0 and method != 'single':
                            achieved_skr = self.qkey_length / ((self.qkey_length / final_skr) + max(delays) - max(0.005, min(delays)))
                            if min(delays) > 0.005:
                                diff_delay = [x - min(delays) for x in delays]
                            else:
                                delays = [i if i > 0.005 else 0.005 for i in delays]
                                diff_delay = [x - 0.005 for x in delays]
                            max_diff = max(diff_delay)
                            buffered_kr = [((max_diff - i) * achieved_skr) / len(diff_delay) if method == 'concatenate'
                                           else (max_diff - i) * achieved_skr for i in diff_delay]
                        else:
                            achieved_skr = final_skr
                            buffered_kr = [0, 0, 0]
                    else:
                        achieved_skr = 0
                        buffered_kr = [0, 0, 0]

                # qkd_blocked = self._reserve_skr(paths[0], 1000, i)
                # p = 1
                # while qkd_blocked and p < 3:
                #     qkd_blocked = self._reserve_skr(paths[p], 1000, i)
                #     p += 1
                # qkd_blocked_lst.append(qkd_blocked)

                print(f"Path from {source} to {destination}: {paths} with requested traffic: {traffic}")
            else:
                selected_path_r = 0
                selected_path_t = 0
                print(f"No path found from {source} to {destination} with requested traffic: {traffic}")

            #cum_qkd_blocked += sum(qkd_blocked_lst)
            # df_info.iloc[i, 0] = source
            # df_info.iloc[i, 1] = destination
            # df_info.iloc[i, 2] = traffic
            if buffered_kr and achieved_skr > 0:
                key_secrecy = [1 - (i/achieved_skr) for i in buffered_kr]
                if method == 'concatenate':
                    S = math.prod(key_secrecy)
                    paths_secrecy = 1 - math.prod([1 - (0.99 ** i) for i in NTNs])
                    compromised_secrecy = 1 - (S * paths_secrecy)
                else:
                    compromised_secrecy = math.prod([1 - key_secrecy[i] * (0.99 ** NTNs[i]) for i in range(len(paths))])

                if len(distances) == 3:
                    [dis1, dis2, dis3] = distances
                    [NTN1, NTN2, NTN3] = NTNs
                    [buffered_1, buffered_2, _] = buffered_kr
                elif len(distances) == 2:
                    [dis1, dis2] = distances
                    [NTN1, NTN2] = NTNs
                    buffered_1 = buffered_kr[0]
                    [dis3, NTN3, buffered_2] = ['-', '-', '-']
                else:
                    [dis1] = distances
                    [NTN1] = NTNs
                    [dis2, dis3, NTN2, NTN3, buffered_1, buffered_2] = ['-', '-', '-', '-', 0, '-']

                df_skr.append({
                    'distance_1': dis1,
                    'distance_2': dis2,
                    'distance_3': dis3,
                    'NTN_1': NTN1,
                    'NTN_2': NTN2,
                    'NTN_3': NTN3,
                    'achieved_skr': achieved_skr,
                    'buffered_1': buffered_1,
                    'buffered_2': buffered_2,
                    'Num_of_paths': len(distances),
                    'compromised_secrecy': compromised_secrecy,
                    'blocking': blocking
                })

            if cum_paths_count > 0:
                df_blocking.append(cum_qkd_blocked/cum_paths_count)
            else:
                df_blocking.append(0)

            # df_info.iloc[i, 4] = self.rb_blocking / (i + 1)
            # df_info.iloc[i, 4] = self.tb_blocking
            # df_info.iloc[i, 5] = selected_path_r
            # df_info.iloc[i, 6] = selected_path_t
            # df_info.iloc[i, 7:7+len(qkd_blocked_lst)] = qkd_blocked_lst

        return pd.DataFrame(df_skr), pd.Series(df_blocking)

    def tmp_qkd_ch(self):
        lst = []
        for i in range(13):
            j = i
            while j < 13:
                source = i + 1
                destination = j + 2
                paths, _ = self._find_top_k_paths(source, destination, 3)
                for path in paths:
                    qber, skr, distance = self._qkd_channel(path)
                    lst.append([source, destination, str(path), qber, skr, distance])
                j += 1
        df = pd.DataFrame(lst, columns=['source', 'destination', 'path', 'qber', 'skr', 'distance'])
        return df

    def skr_comparison(self):
        skr_lst = []
        relays_dis = []
        distance = 2500
        relays = [100, 90, 80, 70, 60, 50]
        for i in range(6):
            relays_dis.append(relays[i])
            obj = coexistance(1, 2, relays[i])
            center_freq, slot_widths, signals, baud_rates = obj.channels_init(1000)
            sprs_value_seo, spectral_info_out, power_out = obj.setup_channels(center_freq, slot_widths, signals,
                                                                              baud_rates)
            skr, _, _ = obj.get_sek_qber(sprs_value_seo)
            skr = sum(skr)
            skr_lst.append(skr)

        comparison = pd.DataFrame(
            {'sync_skr': skr_lst,
             'relays_distance': relays_dis
             })
        return comparison

    def multipath_skr_comparison(self):
        source_lst = []
        destination_lst = []
        dis1_lst = []
        dis2_lst = []
        dis3_lst = []
        single_path_lst = []
        double_path_lst = []
        triple_path_lst = []
        first_path_lst = []
        second_path_lst = []
        third_path_lst = []
        second_path_skr_lst = []
        third_path_skr_lst = []
        key_length = 256
        for i in range(14):
            j = i
            while j < 13:
                j += 1
                paths, _ = self._find_top_k_paths(i + 1, j + 1, 3)
                first_path_delay, first_path_skr, dis1 = self._delay_qkd_channel(paths[0], key_length)
                second_path_delay, second_path_skr, dis2 = self._delay_qkd_channel(paths[1], key_length)
                third_path_delay, third_path_skr, dis3 = self._delay_qkd_channel(paths[2], key_length)
                single_path_lst.append(first_path_skr)
                second_path_skr_lst.append(second_path_skr)
                third_path_skr_lst.append(third_path_skr)
                double_path_lst.append(key_length / max([second_path_delay, first_path_delay]))
                triple_path_lst.append(key_length / max([third_path_delay, second_path_delay, first_path_delay]))
                first_path_lst.append(first_path_delay)
                second_path_lst.append(second_path_delay)
                third_path_lst.append(third_path_delay)
                source_lst.append(i + 1)
                destination_lst.append(j + 1)
                dis1_lst.append(dis1)
                dis2_lst.append(dis2)
                dis3_lst.append(dis3)
        comparison = pd.DataFrame(
            {'source': source_lst,
             'destination': destination_lst,
             'single_path_skr': single_path_lst,
             'second-path-skr': second_path_skr_lst,
             'third-path-skr': third_path_skr_lst,
             'double_path_skr': double_path_lst,
             'triple_path_skr': triple_path_lst,
             'first-path-delay': first_path_lst,
             'second-path-delay': second_path_lst,
             'third-path-delay': third_path_lst,
             'distance_path1': dis1_lst,
             'distance_path2': dis2_lst,
             'distance_path3': dis3_lst
             })
        return comparison
