from Diffusion_NormalIC import *


class SeedSelectionHD:
    def __init__(self, g_dict, s_c_dict, prod_list, total_bud):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### total_bud: (int) the budget to select seed
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.total_budget = total_bud
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)

    def constructDegreeDict(self, data_name):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        with open(IniGraph(data_name).data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    if deg in d_dict:
                        d_dict[deg].add((k, i))
                    else:
                        d_dict[deg] = {(k, i)}
        f.close()

        return d_dict

    def getHighDegreeNode(self, d_dict, cur_bud):
        # -- get the node with highest degree --
        mep = [-1, '-1']
        max_degree = -1
        while mep[1] == '-1':
            while max_degree == -1:
                for deg in list(d_dict.keys()):
                    if int(deg) > max_degree:
                        max_degree = int(deg)

                if max_degree == -1:
                    return mep, d_dict

                if d_dict[str(max_degree)] == set():
                    del d_dict[str(max_degree)]
                    max_degree = -1

            if d_dict[str(max_degree)] == set():
                del d_dict[str(max_degree)]
                max_degree = -1
                continue

            mep = choice(list(d_dict[str(max_degree)]))
            d_dict[str(max_degree)].remove(mep)
            mep = list(mep)

            if self.seed_cost_dict[mep[1]] + cur_bud > self.total_budget:
                mep[1] = '-1'

        return mep, d_dict


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    total_budget = 1
    pp_strategy = 1
    whether_passing_information_without_purchasing = bool(0)
    eva_monte_carlo = 100

    iniG = IniGraph(data_set_name)
    iniW = IniWallet(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()
    sshd = SeedSelectionHD(graph_dict, seed_cost_dict, product_list, total_budget)

    ### result: (list) [profit, budget, seed number per product, customer number per product, seed set] in this execution_time
    result = []
    avg_profit, avg_budget = 0.0, 0.0
    avg_num_k_seed, avg_num_k_pn = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
    profit_k_list, budget_k_list = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

    # -- initialization for each sample_number --
    ### now_budget: (float) the budget in this execution_time
    now_budget = 0.0
    ### seed_set: (list) the seed set
    seed_set = [set() for _ in range(num_product)]

    degree_dict = sshd.constructDegreeDict(data_set_name)
    mep_g, degree_dict = sshd.getHighDegreeNode(degree_dict, now_budget)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # -- main --
    while now_budget <= total_budget and mep_i_node != '-1':
        seed_set[mep_k_prod].add(mep_i_node)

        budget_k_list[mep_k_prod] += seed_cost_dict[mep_i_node]
        now_budget += seed_cost_dict[mep_i_node]

        mep_g, degree_dict = sshd.getHighDegreeNode(degree_dict, now_budget)
        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    eva = Evaluation(graph_dict, seed_cost_dict, product_list, pp_strategy, whether_passing_information_without_purchasing)
    wallet_list = iniW.getWalletList(product_name)
    personal_prob_list = eva.setPersonalProbList(wallet_list)

    pro_acc, pro_k_list_acc, pnn_k_list_acc = 0.0, [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    for _ in range(eva_monte_carlo):
        pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
        pro_acc += pro
        for kk in range(num_product):
            pro_k_list_acc[kk] += pro_k_list[kk]
            pnn_k_list_acc[kk] += pnn_k_list[kk]
    pro_acc = round(pro_acc / eva_monte_carlo, 4)
    for kk in range(num_product):
        profit_k_list[kk] += round(pro_k_list_acc[kk] / eva_monte_carlo, 4)
        pnn_k_list_acc[kk] = round(pnn_k_list_acc[kk] / eva_monte_carlo, 2)
    now_budget = round(now_budget, 2)

    # -- result --
    now_num_k_seed = [len(kk) for kk in seed_set]
    result.append([pro_acc, now_budget, now_num_k_seed, pnn_k_list_acc, seed_set])
    avg_profit += pro_acc
    avg_budget += now_budget
    for kk in range(num_product):
        budget_k_list[kk] = round(budget_k_list[kk], 2)
        avg_num_k_seed[kk] += now_num_k_seed[kk]
        avg_num_k_pn[kk] += pnn_k_list_acc[kk]
    how_long = round(time.time() - start_time, 2)
    print("\nresult")
    print(result)
    print("\npro_k_list, budget_k_list")
    print(profit_k_list, budget_k_list)
    print("total time: " + str(how_long) + "sec")
