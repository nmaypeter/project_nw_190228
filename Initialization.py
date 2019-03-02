import random
import os.path
import time


class IniGraph:
    def __init__(self, data_name):
        ### data_set_name, data_data_path, data_weight_path, data_degree_path: (str)
        self.data_name = data_name
        self.data_data_path = "data/" + data_name + "/data.txt"
        self.data_weight_path = "data/" + data_name + "/weight.txt"
        self.data_degree_path = "data/" + data_name + "/degree.txt"

    def setEdgeWeight(self):
        #  -- set weight on edge --
        fw = open(self.data_weight_path, 'w')
        with open(self.data_data_path) as f:
            for line in f:
                (key, val) = line.split()
                # --- output: first node, second node, weight on the edge within nodes ---
                fw.write(key + "\t" + val + "\t" + "0.1\n")
        fw.close()
        f.close()

    def outputNodeOutDegree(self):
        #  -- count the out-degree --
        ### num_node: (int) the number of nodes in data
        fw = open(self.data_degree_path, 'w')
        with open(self.data_data_path) as f:
            num_node = 0
            out_degree_list = []
            for line in f:
                (node1, node2) = line.split()
                num_node = max(num_node, int(node1), int(node2))
                out_degree_list.append(node1)

        for i in range(0, num_node + 1):
            fw.write(str(i) + "\t" + str(out_degree_list.count(str(i))) + "\n")
        fw.close()
        f.close()

    def getNodeOutDegree(self, i_node):
        #  -- get the out-degree --
        deg = 0
        with open(self.data_degree_path) as f:
            for line in f:
                (node, degree) = line.split()
                if node == i_node:
                    deg = int(degree)
                    break
        f.close()

        return deg

    def constructSeedCostDict(self):
        # -- calculate the cost for each seed --
        ### s_cost_dict: (dict) the set of cost for each seed
        ### s_cost_dict[ii]: (float2) the degree of ii's seed
        ### num_node: (int) the number of nodes in data
        ### max_deg: (int) the maximum degree in data
        s_cost_dict = {}
        with open(self.data_degree_path) as f:
            num_node, max_deg = 0, 0
            seed_cost_list = []
            for line in f:
                (node, degree) = line.split()
                num_node = max(num_node, int(node))
                max_deg = max(max_deg, int(degree))
                seed_cost_list.append([node, degree])

            for i in range(num_node + 1):
                s_cost_dict[str(i)] = round(int(seed_cost_list[i][1]) / max_deg, 2)
        f.close()

        return s_cost_dict

    def constructGraphDict(self):
        # -- build graph --
        ### graph: (dict) the graph
        ### graph[node1]: (dict) the set of node1's receivers
        ### graph[node1][node2]: (str) the weight one the edge of node1 to node2
        graph = {}
        with open(self.data_weight_path) as f:
            for line in f:
                (node1, node2, wei) = line.split()
                if node1 in graph:
                    graph[node1][node2] = str(wei)
                else:
                    graph[node1] = {node2: str(wei)}
        f.close()
        return graph

    def getNumNode(self):
        #  -- get the num_node --
        ### num_node: (int) the number of nodes in data
        num_node = 0
        with open(self.data_data_path) as f:
            for line in f:
                (node1, node2) = line.split()
                num_node = max(int(node1), int(node2), num_node)
        f.close()
        print('num_node = ' + str(round(num_node, 2)))

        return num_node

    def getNumEdge(self):
        # -- get the num_edge --
        num_edge = 0
        with open(self.data_weight_path) as f:
            for line in f:
                num_edge += 1
        f.close()
        print('num_edge = ' + str(round(num_edge, 2)))

        return num_edge

    def getMaxDegree(self):
        # -- get the max_deg --
        ### max_deg: (int) the maximum degree in data
        with open(self.data_degree_path) as f:
            max_deg = 0
            for line in f:
                (node, degree) = line.split()
                max_deg = max(max_deg, int(degree))
        f.close()
        print('max_deg = ' + str(round(max_deg, 2)))

        return max_deg


class IniWallet:
    def __init__(self, data_name):
        ### data_set_name, data_degree_path: (str)
        self.data_name = data_name
        self.data_degree_path = "data/" + data_name + "/degree.txt"

    def setNodeWallet(self, prod_name, upper):
        # -- set node's personal budget (wallet) --
        fw = open("data/" + self.data_name + "/wallet_r" + list(prod_name)[list(prod_name).index('r') + 1] +
                  "p" + list(prod_name)[list(prod_name).index('p') + 1] +
                  "n" + list(prod_name)[list(prod_name).index('n') + 1] + ".txt", 'w')
        with open(self.data_degree_path) as f:
            for line in f:
                (key, val) = line.split()
                fw.write(key + "\t" + str(round(random.uniform(0, upper), 2)) + "\n")
        fw.close()
        f.close()

    def getWalletList(self, prod_name):
        # -- get wallet_list from file --
        w_list = []
        with open("data/" + self.data_name + "/wallet_r" + list(prod_name)[list(prod_name).index('r') + 1] +
                  "p" + list(prod_name)[list(prod_name).index('p') + 1] +
                  "n" + list(prod_name)[list(prod_name).index('n') + 1] + ".txt") as f:
            for line in f:
                (node, wallet) = line.split()
                w_list.append(float(wallet))
        f.close()

        return w_list

    def getTotalWallet(self, prod_name):
        # -- get total_wallet from file --
        total_w = 0.0
        with open("data/" + self.data_name + "/wallet_r" + list(prod_name)[list(prod_name).index('r') + 1] +
                  "p" + list(prod_name)[list(prod_name).index('p') + 1] +
                  "n" + list(prod_name)[list(prod_name).index('n') + 1] + ".txt") as f:
            for line in f:
                (node, wallet) = line.split()
                total_w += float(wallet)
        f.close()
        print(prod_name + ' = total wallet = ' + str(round(total_w, 2)))

        return total_w


class IniProduct:
    def __init__(self, prod_name):
        ### prod_name: (str)
        ### num_ratio, num_price: (int)
        self.prod_name = prod_name
        self.num_ratio = int(list(prod_name)[list(prod_name).index('r') + 1])
        self.num_price = int(list(prod_name)[list(prod_name).index('p') + 1])

    def setProductListWithSRRandMFP(self):
        # -- set the product with single random ratios and multiple fix interval prices
        # -- the difference between each price has to be greater than 1 / number_price --
        ### dp: (int) the definition of price
        ### prod_list: (list) the set to record output products
        ### prod_list[num]: (list) [num's profit, num's cost, num's ratio, num's price]
        ### prod_list[num][]: (float2)
        dp = 1
        prod_list = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.num_price)]
        while bool(dp):
            dp = min(0, dp)
            prod_list = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.num_price)]
            bias_price = round(random.uniform(0, 1 / self.num_price), 2)
            prod_ratio = round(random.uniform(0, 2), 2)
            for k in range(self.num_price):
                prod_list[k][3] = round(bias_price * (k + 1), 2)
                prod_list[k][0] = round(prod_list[k][3] * (prod_ratio / (1 + prod_ratio)), 2)
                prod_list[k][1] = round(prod_list[k][3] * (1 / (1 + prod_ratio)), 2)
                if prod_list[k][1] == 0:
                    dp += 1
                    continue
                prod_list[k][2] = round(prod_list[k][0] / prod_list[k][1], 2)
                if prod_list[k][0] < 0.05 or prod_list[k][1] < 0.05 or prod_list[k][3] > 1 or prod_list[k][0] + \
                        prod_list[k][1] != prod_list[k][3]:
                    dp += 1
                    continue
            for k in range(len(prod_list) - 1):
                if abs(prod_list[k + 1][2] - prod_list[k][2]) > 0.05:
                    dp += 1
                    continue

        n = 1
        file_path = "product/r1p" + str(self.num_price) + "n" + str(n) + ".txt"
        while os.path.exists(file_path):
            file_path = "product/r1p" + str(self.num_price) + "n" + str(n) + ".txt"
            n += 1
        fw = open(file_path, 'w')
        for p, c, r, pr in prod_list:
            fw.write(str(p) + " " + str(c) + " " + str(r) + " " + str(pr) + "\n")
        fw.close()

    def getProductList(self):
        # -- get product list from file
        ### prod_list: (list) [profit, cost, price]
        prod_list = []
        with open("product/" + self.prod_name + ".txt") as f:
            for line in f:
                (p, c, r, pr) = line.split()
                prod_list.append([float(p), float(c), round(float(p) + float(c), 2)])

        return prod_list

    def getTotalPrice(self):
        # -- get total_price from file
        ### total_price: (float2) the sum of prices
        total_price = 0.0
        with open("product/" + self.prod_name + ".txt") as f:
            for line in f:
                (p, c, r, pr) = line.split()
                total_price += float(pr)
        print('total_price = ' + str(round(total_price, 2)))

        return round(total_price, 2)


if __name__ == "__main__":
    start_time = time.time()
    data_set_name = "NetPHY_undirected"
    product_name = "r1p3n2"

    iniG = IniGraph(data_set_name)
    iniW = IniWallet(data_set_name)
    iniP = IniProduct(product_name)

    iniG.setEdgeWeight()
    iniG.outputNodeOutDegree()
    # number_node = iniG.getNumNode()
    number_edge = iniG.getNumEdge()
    max_degree = iniG.getMaxDegree()

    # iniP.setProductListWithSRRandMFP()
    # product_list = iniP.getProductList()
    # sum_price = iniP.getTotalPrice()
    # iniW.setNodeWallet(product_name, sum_price)

    # seed_cost_dict = iniG.constructSeedCostDict()
    # graph_dict = iniG.constructGraphDict()
    # product_list = iniP.getProductList()
    # wallet_list = iniW.getWalletList(product_name)
    total_wallet = iniW.getTotalWallet(product_name)

    how_long = round(time.time() - start_time, 4)
    print("total time: " + str(how_long) + "sec")

    ### -- sum_price --
    ### -- r1p3n1, r1p3n2 = 1.44 --
    ### -- r1p3n1a, r1p3n2a = 1.32 --
    ### -- r1p3n1b, r1p3n2b = 1.68 --

    ### -- num_node --
    ### -- email_undirected = 1133 --
    ### -- dnc_email_directed = 2029 --
    ### -- email_Eu_core_directed = 1004 --
    ### -- WikiVote_directed = 7115 --
    ### -- NetPHY_undirected = 37149 --

    ### -- num_edge --
    ### -- email_undirected = 10902 --
    ### -- dnc_email_directed = 5598 --
    ### -- email_Eu_core_directed = 25571 --
    ### -- WikiVote_directed = 201524 --
    ### -- NetPHY_undirected = 348322 --

    ### -- max_degree --
    ### -- email_undirected = 71 --
    ### -- dnc_email_directed = 331 --
    ### -- email_Eu_core_directed = 334 --
    ### -- WikiVote_directed = 1065 --
    ### -- NetPHY_undirected = 178 --

    ### -- total wallet --
    ### -- email_undirected: r1p3n1 = 833.93 --
    ### -- email_undirected: r1p3n2 = 817.32 --
    ### -- dnc_email_directed: r1p3n1 = 1451.4 --
    ### -- dnc_email_directed: r1p3n2 = 1449.16 --
    ### -- email_Eu_core_directed: r1p3n1 = 729.0 --
    ### -- email_Eu_core_directed: r1p3n2 = 712.76 --
    ### -- WikiVote_directed: r1p3n1 = 5995.88 --
    ### -- WikiVote_directed: r1p3n2 = 5998.45 --
    ### -- NetPHY_undirected - r1p3n1 = total_wallet = 26920.52 --
    ### -- NetPHY_undirected - r1p3n2 = total_wallet = 26734.2 --
