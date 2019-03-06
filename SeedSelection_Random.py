from Diffusion_NormalIC import *


class SeedSelectionR:
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

    def selectRandomSeed(self, nb_set):
        # -- select a seed for a random product randomly
        ### product_num: (list) record the product number with possible seeds
        product_num = [k for k in range(self.num_product)]
        mep = [-1, '-1']
        while mep[0] == -1:
            mep[0] = choice(product_num)
            if len(nb_set[mep[0]]) == 0:
                product_num.remove(mep[0])
                mep[0] = -1
            if len(product_num) == 0:
                return mep
        mep[1] = choice(list(nb_set[mep[0]]))

        return mep

    def updateNotbanSet(self, nb_set, cur_bud):
        # -- remove the impossible seeds --
        b_set = [set() for _ in range(self.num_product)]
        for k in range(self.num_product):
            for i in nb_set[k]:
                # -- the cost of seed cannot exceed the budget --
                if self.seed_cost_dict[i] + cur_bud > self.total_budget:
                    b_set[k].add(i)

        # -- remove the impossible seeds fom nban_set
        for k in range(self.num_product):
            for i in b_set[k]:
                if i in nb_set[k]:
                    nb_set[k].remove(i)
        return nb_set


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    bud = 10
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
    ssr = SeedSelectionR(graph_dict, seed_cost_dict, product_list, bud)

    # -- initialization for each sample_number --
    ### now_budget: (float) the budget in this execution_time
    now_budget = 0.0
    ### seed_set: (list) the seed set
    seed_set = [set() for _ in range(num_product)]

    nban_set = [{ii for ii in graph_dict} for _ in range(num_product)]
    nban_set = ssr.updateNotbanSet(nban_set, 0)
    mep_g = ssr.selectRandomSeed(nban_set)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # -- main --
    while now_budget <= bud and mep_i_node != '-1':
        seed_set[mep_k_prod].add(mep_i_node)
        now_budget += seed_cost_dict[mep_i_node]

        nban_set = ssr.updateNotbanSet(nban_set, now_budget)
        mep_g = ssr.selectRandomSeed(nban_set)
        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

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