from Diffusion_NormalIC import *


class SeedSelectionTO:
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

    def generateTopOrderSequence(self, data_name):
        celf_seq = [[-1, '-1', 0.0, 0]]
        mg_seq, deg_seq, prod_seq = [], [], []

        iniG_ss = IniGraph(data_name)
        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        for k in range(self.num_product):
            if self.product_list[k][2] not in prod_seq:
                prod_seq.append(self.product_list[k][2])
            for i in set(self.graph_dict.keys()):
                # -- the cost of seed cannot exceed the budget --
                if self.seed_cost_dict[i] > self.total_budget:
                    continue

                s_set = [set() for _ in range(self.num_product)]
                s_set[k].add(i)
                ep = 0.0
                for _ in range(self.monte):
                    ep += diff_ss.getSeedSetProfit(s_set)
                ep = round(ep / self.monte, 4)
                mg = round(ep, 4)
                if mg not in mg_seq:
                    mg_seq.append(mg)

                deg = iniG_ss.getNodeOutDegree(str(i))
                if deg not in deg_seq:
                    deg_seq.append(deg)

                celf_ep = [k, i, mg, deg]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

        celf_seq.remove([-1, '-1', 0.0, 0])
        mg_seq.sort(reverse=True)
        deg_seq.sort(reverse=True)
        prod_seq.sort(reverse=True)

        order_seq = [[-1, '-1', 0, 0, 0, 0]]
        for celf_item in celf_seq:
            celf_order = mg_seq.index(celf_item[2]) + 1
            deg_order = deg_seq.index(celf_item[3]) + 1
            prod_order = prod_seq.index(self.product_list[celf_item[0]][2]) + 1
            total_order = celf_order * prod_order + deg_order
            order = [celf_item[0], celf_item[1], celf_order, deg_order, prod_order, total_order]
            order_seq.append(order)
            for order_item in order_seq:
                if order[5] <= order_item[5]:
                    order_seq.insert(order_seq.index(order_item), order)
                    order_seq.pop()
                    break

        order_seq.remove([-1, '-1', 0, 0, 0, 0])
        order_seq.append([-1, '-1', len(mg_seq) + 1, len(deg_seq) + 1, self.num_product + 1, len(celf_seq) + 1])

        return order_seq


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
    ssto = SeedSelectionTO(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)
    diff = Diffusion(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)

    # -- initialization for each sample_number --
    ### now_budget: (float) the budget in this execution_time
    now_budget = 0.0
    ### seed_set: (list) the seed set
    seed_set = [set() for _ in range(num_product)]

    order_sequence = ssto.generateTopOrderSequence(data_set_name)
    mep_g = order_sequence.pop(0)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    while now_budget <= total_budget and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > total_budget:
            mep_g = order_sequence.pop(0)
            mep_k_prod, mep_i_node = mep_g[0], mep_g[1]
            if mep_i_node != '-1':
                break
            continue

        seed_set[mep_k_prod].add(mep_i_node)
        now_budget += seed_cost_dict[mep_i_node]

        mep_g = order_sequence.pop(0)
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
