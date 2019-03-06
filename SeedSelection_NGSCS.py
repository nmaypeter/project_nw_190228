from Diffusion_NormalIC import *
import numpy as np
from scipy import stats


class SeedSelectionNGSCS:
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
        celf_seq = [[[-1, '-1', 0.0, 0]] for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        for i in set(self.graph_dict.keys()):
            # -- the cost of seed cannot exceed the budget --
            if self.seed_cost_dict[i] > self.total_budget:
                continue

            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diff_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)

            for k in range(self.num_product):
                mg = round(ep * self.product_list[k][0] / self.product_list[0][0], 4)
                mg_score = round(mg, 4)

                celf_ep = [k, i, mg_score, 0]
                celf_seq[k].append(celf_ep)
                for celf_item in celf_seq[k]:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq[k].insert(celf_seq[k].index(celf_item), celf_ep)
                        celf_seq[k].pop()
                        break

        return celf_seq


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    bud = 10
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
    ssngscs = SeedSelectionNGSCS(graph_dict, seed_cost_dict, product_list, bud, monte_carlo)
    diff = Diffusion(graph_dict, seed_cost_dict, product_list, bud, monte_carlo)

    # -- initialization for each sample_number --
    now_budget = 0.0
    seed_set = [set() for _ in range(num_product)]

    celf_sequence = ssngscs.generateCelfSequence()
    mep_celf = [-1, 0.0]
    for kk in range(num_product):
        if celf_sequence[kk][0][2] > mep_celf[1]:
            mep_celf = [kk, celf_sequence[kk][0][2]]
    mep_g = celf_sequence[mep_celf[0]].pop(0)
    mep_k_prod, mep_i_node, mep_mg_score, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

    while now_budget <= bud and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > bud:
            mep_celf = [-1, 0.0]
            for kk in range(num_product):
                if celf_sequence[kk][0][2] > mep_celf[1]:
                    mep_celf = [kk, celf_sequence[kk][0][2]]
            mep_g = celf_sequence[mep_celf[0]].pop(0)
            mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]
            if mep_i_node != '-1':
                break
            continue

        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
        if mep_flag == seed_set_length:
            seed_set[mep_k_prod].add(mep_i_node)
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
            mep_mg_score = round(mep_mg, 4)

            seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
            mep_flag = seed_set_length

            celf_ep_g = [mep_k_prod, mep_i_node, mep_mg_score, mep_flag]
            celf_sequence[mep_k_prod].append(celf_ep_g)
            for celf_item_g in celf_sequence[mep_k_prod]:
                if celf_ep_g[2] >= celf_item_g[2]:
                    celf_sequence[mep_k_prod].insert(celf_sequence[mep_k_prod].index(celf_item_g), celf_ep_g)
                    celf_sequence[mep_k_prod].pop()
                    break

        mep_celf = [-1, 0.0]
        for kk in range(num_product):
            if celf_sequence[kk][0][2] > mep_celf[1]:
                mep_celf = [kk, celf_sequence[kk][0][2]]
        mep_g = celf_sequence[mep_celf[0]].pop(0)
        mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

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