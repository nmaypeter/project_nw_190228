from Diffusion_NormalIC import *


class SeedSelectionIP:
    def __init__(self, g_dict, s_c_dict, prod_list, total_bud, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### total_bud: (int) the budget to select seed
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.total_budget = total_bud
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.monte = monte

    def executeIterativePrune(self, A_set, B_set):
        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)
        A_set_minus, B_set_minus = [set(self.graph_dict) for _ in range(self.num_product)], [set(self.graph_dict) for _ in range(self.num_product)]
        for k in range(self.num_product):
            A_set_minus[k] = A_set_minus[k] - A_set[k]
            B_set_minus[k] = B_set_minus[k] - B_set[k]
            print(len(A_set_minus[k]), len(B_set_minus[k]))
        A_set_new, B_set_new = [set() for _ in range(self.num_product)], [set() for _ in range(self.num_product)]
        for k in range(self.num_product):
            print('b')
            if B_set[k] == set(self.graph_dict):
                A_set_new[k] = set()
            else:
                for i in A_set_minus[k]:
                    s_set_t = copy.deepcopy(B_set)
                    s_set_t[k].remove(i)
                    ep = 0.0
                    for _ in range(self.monte):
                        ep += diff_ss.getExpectedProfit(k, i, s_set_t)
                    if ep > 0:
                        A_set_new[k].add(i)

            print('a')
            if A_set[k] == set(self.graph_dict):
                B_set_new[k] = set()
            else:
                for i in B_set_minus[k]:
                    ep = 0.0
                    for _ in range(self.monte):
                        ep += diff_ss.getExpectedProfit(k, i, A_set)
                    if ep <= 0:
                        B_set_new[k].add(i)

        for k in range(self.num_product):
            print(len(A_set[k]), len(A_set_new[k]), len(B_set[k]), len(B_set_new[k]))

        return A_set, A_set_new, B_set, B_set_new


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    total_budget = 10
    pp_strategy = 1
    whether_passing_information_without_purchasing = bool(0)
    monte_carlo, eva_monte_carlo = 10, 100

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
    ssip = SeedSelectionIP(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)
    diff = Diffusion(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)

    # -- initialization for each sample_number --
    now_budget = 0.0

    A_seed_set, B_seed_set = [set() for _ in range(num_product)], [set(graph_dict.keys()) for _ in range(num_product)]
    A_seed_set, A_seed_set_new, B_seed_set, B_seed_set_new = ssip.executeIterativePrune(A_seed_set, B_seed_set)
    print(round(time.time() - start_time, 4))
    while A_seed_set != A_seed_set_new or B_seed_set != B_seed_set_new:
        A_seed_set, A_seed_set_new, B_seed_set, B_seed_set_new = ssip.executeIterativePrune(A_seed_set_new, B_seed_set_new)
        print(round(time.time() - start_time, 4))

    seed_set = A_seed_set_new
    eva = Evaluation(graph_dict, seed_cost_dict, product_list, pp_strategy, whether_passing_information_without_purchasing)
    wallet_list = iniW.getWalletList(product_name)
    personal_prob_list = eva.setPersonalProbList(wallet_list)

    sample_pro_acc, sample_bud_acc = 0.0, 0.0
    sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

    for _ in range(eva_monte_carlo):
        pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
        sample_pro_acc += pro
        for kk in range(num_product):
            sample_pro_k_acc[kk] += pro_k_list[kk]
            sample_pnn_k_acc[kk] += pnn_k_list[kk]
    sample_pro_acc = round(sample_pro_acc / eva_monte_carlo, 4)
    for kk in range(num_product):
        sample_pro_k_acc[kk] = round(sample_pro_k_acc[kk] / eva_monte_carlo, 4)
        sample_pnn_k_acc[kk] = round(sample_pnn_k_acc[kk] / eva_monte_carlo, 2)
        sample_sn_k_acc[kk] = len(seed_set[kk])
        for sample_seed in seed_set[kk]:
            sample_bud_acc += seed_cost_dict[sample_seed]
            sample_bud_k_acc[kk] += seed_cost_dict[sample_seed]
            sample_bud_acc = round(sample_bud_acc, 2)
            sample_bud_k_acc[kk] = round(sample_bud_k_acc[kk], 2)

    print("seed set: " + str(seed_set))
    print("profit: " + str(sample_pro_acc))
    print("budget: " + str(sample_bud_acc))
    print("seed number: " + str(sample_sn_k_acc))
    print("purchasing node number: " + str(sample_pnn_k_acc))
    print("ratio profit: " + str(sample_pro_k_acc))
    print("ratio budget: " + str(sample_bud_k_acc))
    print("total time: " + str(round(time.time() - start_time, 2)) + "sec")