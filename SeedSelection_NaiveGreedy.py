from Diffusion_NormalIC import *


class SeedSelectionNG:
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

    def generateCelfSequence(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        celf_seq = [[-1, '-1', 0.0, 0]]

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        for k in range(self.num_product):
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

                celf_ep = [k, i, mg, 0]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

        return celf_seq


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
    ssng = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)
    diff = Diffusion(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)

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

    celf_sequence = ssng.generateCelfSequence()
    mep_g = celf_sequence.pop(0)
    mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

    while now_budget <= total_budget and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > total_budget:
            mep_g = celf_sequence.pop(0)
            mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]
            continue

        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
        if mep_flag == seed_set_length:
            seed_set[mep_k_prod].add(mep_i_node)

            budget_k_list[mep_k_prod] += seed_cost_dict[mep_i_node]
            now_budget += seed_cost_dict[mep_i_node]
        else:
            ep_g = 0.0
            for _ in range(monte_carlo):
                ep_g += diff.getSeedSetProfit(seed_set)
            ep_g = round(ep_g / monte_carlo, 4)

            ep1_g = 0.0
            for _ in range(monte_carlo):
                ep1_g += diff.getExpectedProfit(mep_k_prod, mep_i_node, seed_set)
            ep1_g = round(ep1_g / monte_carlo, 4)
            mep_mg = round(ep1_g - ep_g, 4)

            seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
            mep_flag = seed_set_length

            celf_ep_g = [mep_k_prod, mep_i_node, mep_mg, mep_flag]
            celf_sequence.append(celf_ep_g)
            for celf_item_g in celf_sequence:
                if celf_ep_g[2] >= celf_item_g[2]:
                    celf_sequence.insert(celf_sequence.index(celf_item_g), celf_ep_g)
                    celf_sequence.pop()
                    break

        mep_g = celf_sequence.pop(0)
        mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

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
    now_profit = round(pro_acc / eva_monte_carlo, 4)
    for kk in range(num_product):
        profit_k_list[kk] += round(pro_k_list_acc[kk] / eva_monte_carlo, 4)
        pnn_k_list_acc[kk] = round(pnn_k_list_acc[kk] / eva_monte_carlo, 2)
    now_budget = round(now_budget, 2)

    # -- result --
    now_num_k_seed = [len(kk) for kk in seed_set]
    result.append([now_profit, now_budget, now_num_k_seed, pnn_k_list_acc, seed_set])
    avg_profit += now_profit
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