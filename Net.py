import copy
import networkx as nx
import json
import numpy as np
import pandas as pd
from measurement_Framework.coexistance import coexistance
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import minimize


class Net:
    def __init__(self):
        self.G = nx.Graph()
        self._def_edges()
        self._default_capacity = None
        self._connections_capacity = {}
        self._connections_GSNR = {}
        self._def_connection()
        self._connections_buffer()

    def _def_edges(self):
        with open('config/edges.json', 'r') as f:
            self.config_data = json.load(f)
        self.num_of_ch = self.config_data["num_of_channels"]
        for edge in self.config_data['edges']:
            c = np.ceil(edge['distance'] / 100)
            obj = coexistance(edge['source'], edge['target'], edge['distance'] / c)
            center_freq, slot_widths, signals, baud_rates = obj.channels_init()
            sprs_value_seo, spectral_info_out, power_out = obj.setup_channels(center_freq, slot_widths, signals,
                                                                              baud_rates)
            skr, _, _ = obj.get_sek_qber(sprs_value_seo)
            self.G.add_edge(edge['source'], edge['target'], distance=edge['distance'],
                            spectrum=[-1] * self.num_of_ch, connection=[0] * self.num_of_ch,
                            skr=copy.copy(sum(skr)), dedicated_skr=copy.copy(sum(skr)))


    def _connections_buffer(self):
        counter = 0
        for connection, number in self._connections.items():
            if len(connection) == 3:
                if int(connection[0]) < int(connection[2]):
                    conn = connection
            elif len(connection) == 5:
                if int(connection[0:1]) < int(connection[3:4]):
                    conn = connection
            elif len(connection) == 4:
                if connection[2:3].isnumeric():
                    conn = connection
            self._connections_capacity[number] = (
                pd.read_csv('config/connections_info/connection_{}.csv'.format(conn)))
            self._connections_GSNR[number] = (
                pd.read_csv('config/connections_GSNR/connection_{}.csv'.format(conn)))
            counter += 1

    def _return_spectrum_capacities(self, source, destination):
        if source > destination:
            return self._connections_capacity[self._connections['{}-{}'.format(destination, source)]]
        return self._connections_capacity[self._connections['{}-{}'.format(source, destination)]]

    def _return_spectrum_GSNR(self, source, destination):
        if source > destination:
            return self._connections_GSNR[self._connections['{}-{}'.format(destination, source)]]
        return self._connections_GSNR[self._connections['{}-{}'.format(source, destination)]]

    def _def_connection(self):
        self._connections = {}
        number = 1
        for i in range(len(self.G.nodes)):
            for j in range(len(self.G.nodes)):
                if i < j:
                    self._connections["{}-{}".format(i + 1, j + 1)] = number
                    number += 1
                elif i > j:
                    self._connections["{}-{}".format(i + 1, j + 1)] = self._connections["{}-{}"
                    .format(j + 1, i + 1)]
        df = pd.DataFrame(list(self._connections.items()), columns=['connection', 'number'])
        df.to_csv("Outputs/connections.csv")

    def _default_channels(self, path, k):
        df = self._return_spectrum_capacities(path[0], path[-1])
        self._default_capacity = df[['path{}'.format(k + 1)]].iloc[:, 0] * 100
        return self._default_capacity

    def get_GSNR(self, path, k):
        df = self._return_spectrum_GSNR(path[0], path[-1])
        GSNR = df[['path{}'.format(k + 1)]].iloc[:, 0]
        return GSNR

    def _find_1st_empty_channel(self, channels):
        for i in range(len(channels)):
            if channels[i] == 0:
                return i
        return -1

    def _find_existing_capacity(self, path, channel, k):
        existing_capacity_links = []
        for u, v in zip(path[:-1], path[1:]):
            existing_traffic = 0
            if self.G[u][v]['spectrum'][channel] >= 0:
                existing_traffic = int(self.G[u][v]['spectrum'][channel])
            if int(self._default_channels(path, k)[channel]) - existing_traffic > 0:
                existing_capacity_links.append(int(self._default_channels(path, k)[channel]) - existing_traffic)
            else:
                existing_capacity_links.append(0)
        return min(existing_capacity_links)

    def reset_network(self):
        for u, v in self.G.edges():
            self.G[u][v]['spectrum'] = [-1] * self.num_of_ch  # Reset traffic
            self.G[u][v]['connection'] = [0] * self.num_of_ch
            self.G[u][v]['dedicated_skr'] = copy.copy(self.G[u][v]['skr'])

    def allocate_channels(self, path, channel, requested_traffic):
        connection = "{}-{}".format(path[0], path[-1])
        for u, v in zip(path[:-1], path[1:]):
            if self.G[u][v]['connection'][channel] == self._connections[connection]:
                self.G[u][v]['spectrum'][channel] += requested_traffic
            elif self.G[u][v]['connection'][channel] == 0:
                # connect and setup TRx
                self.G[u][v]['spectrum'][channel] = requested_traffic
                self.G[u][v]['connection'][channel] = self._connections[connection]
            else:
                print()

    def establish_qkd_channels(self):

        return

    def find_capacity_channels(self, path, k):
        channels_capacity = []
        connection = "{}-{}".format(path[0], path[-1])
        for ch in range(self.num_of_ch):
            links_capacity = []
            for u, v in zip(path[:-1], path[1:]):
                link_capacity = 0
                if (self.G[u][v]['connection'][ch] == self._connections[connection] or
                        self.G[u][v]['connection'][ch] == 0):
                    link_capacity = self._find_existing_capacity([u, v], ch, k)

                links_capacity.append(link_capacity)
            channels_capacity.append(min(links_capacity))
        return channels_capacity

    def return_network(self):
        return self.G

    def return_edge_data(self):
        edge_data = []
        for (u, v, data) in self.G.edges(data=True):
            edge_data.append((u, v, *data['connection'], *data['spectrum']))
        return edge_data

    def draw_graph_topology(self):
        pos = self._compute_positions()
        nx.draw_networkx_nodes(self.G, pos, node_size=500, node_color='lightblue')

        # Draw the edges with edge lengths (distances)
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), width=2)

        # Draw the labels on the nodes
        nx.draw_networkx_labels(self.G, pos, font_size=12, font_color="black")

        # Draw edge labels showing distances
        edge_labels = nx.get_edge_attributes(self.G, 'distance')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        plt.title("Network Graph with Edge Distances")
        plt.axis('off')  # Hide the axes
        plt.show()

    def _compute_positions(self):
        # Get the list of nodes
        nodes = list(self.G.nodes)
        n = len(nodes)

        # Initialize random positions (2D coordinates) for the nodes
        positions = np.random.rand(n, 2) * 10  # Random initial positions scaled up

        # Extract edge lengths (distances)
        distances = nx.get_edge_attributes(self.G, 'weight')

        # Optimization: minimize the difference between actual and desired distances
        def distance_error(pos_flat):
            pos = pos_flat.reshape((n, 2))
            current_distances = distance_matrix(pos, pos)
            error = 0.0
            for (i, j), target_distance in distances.items():
                idx_i = nodes.index(i)
                idx_j = nodes.index(j)
                error += (current_distances[idx_i, idx_j] - target_distance) ** 2
            return error

        # Use scipy's minimization method to optimize positions

        result = minimize(distance_error, positions.flatten(), method='BFGS')

        # Reshape the optimized flat position array back to 2D
        optimized_positions = result.x.reshape((n, 2))

        # Convert positions to a dict mapping node names to coordinates
        pos_dict = {node: optimized_positions[i] for i, node in enumerate(nodes)}

        return pos_dict

