from Diffusion_NormalIC import *
import operator


class SeedSelectionPMIS:
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

    def getMarginalProfit(self, k_prod, i_node, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- notice: prevent the node from owing no receiver --
        if i_node not in self.graph_dict:
            return round(ep, 4)

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set_t[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, child_depth = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if child_depth >= 3:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, child_depth + 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set_t[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, child_depth = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if child_depth >= 3:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, child_depth + 1])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)

    def generateCelfSequence(self, k_prod):
        # -- calculate expected profit for all combinations of nodes and products --
        celf_seq = [[-1, '-1', 0.0, 0.0]]

        sspmis_ss = SeedSelectionPMIS(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        for i in set(self.graph_dict.keys()):
            # -- the cost of seed cannot exceed the budget --
            if self.seed_cost_dict[i] > self.total_budget:
                continue

            s_set = [set() for _ in range(self.num_product)]
            s_set[k_prod].add(i)
            ep = 0.0

            for _ in range(self.monte):
                ep += sspmis_ss.getMarginalProfit(k_prod, i, s_set)
            ep = round(ep / self.monte, 4)
            mg = round(ep, 4)
            mg_ratio = 0.0
            if self.seed_cost_dict[i] != 0:
                mg_ratio = round(mg / self.seed_cost_dict[i], 4)

            if mg <= 0:
                continue
            celf_ep = [k_prod, i, mg, mg_ratio]
            celf_seq.append(celf_ep)
            for celf_item in celf_seq:
                if celf_ep[3] >= celf_item[3]:
                    celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                    celf_seq.pop()
                    break

        celf_seq.remove([-1, '-1', 0.0, 0.0])
        mep = celf_seq[0]

        return mep, celf_seq

    def getMostValuableSeed(self, s_set, celf_seq_ini, cur_pro, cur_bud):
        # -- calculate expected profit for all combinations of nodes and products --
        celf_seq = [[-1, '-1', 0.0, 0.0]]

        sspmis_ss = SeedSelectionPMIS(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        # -- the cost of seed cannot exceed the budget --
        if self.seed_cost_dict[celf_seq_ini[0][1]] + cur_bud <= self.total_budget and len(celf_seq_ini) >= 2:
            ep_ini_top = 0.0
            for _ in range(self.monte):
                ep_ini_top += sspmis_ss.getMarginalProfit(celf_seq_ini[0][0], celf_seq_ini[0][1], s_set)
            ep_ini_top = round(ep_ini_top / self.monte, 4)
            mg_ini = round(ep_ini_top - cur_pro, 4)
            mg_ini_ratio = 0.0
            if self.seed_cost_dict[celf_seq_ini[0][1]] != 0:
                mg_ini_ratio = round(mg_ini / self.seed_cost_dict[celf_seq_ini[0][1]], 4)

            if mg_ini_ratio >= celf_seq_ini[1][3]:
                mep = [celf_seq_ini[0][0], celf_seq_ini[0][1], mg_ini, mg_ini_ratio]

                return mep, celf_seq_ini

        for celf in celf_seq_ini:
            k_prod, i_node = celf[0], celf[1]
            ep = 0.0

            # -- the cost of seed cannot exceed the budget --
            if self.seed_cost_dict[i_node] + cur_bud > self.total_budget:
                continue

            for _ in range(self.monte):
                ep += sspmis_ss.getMarginalProfit(k_prod, i_node, s_set)
            ep = round(ep / self.monte, 4)
            mg = round(ep - cur_pro, 4)
            mg_ratio = 0.0
            if self.seed_cost_dict[i_node] != 0:
                mg_ratio = round(mg / self.seed_cost_dict[i_node], 4)

            if mg <= 0:
                continue
            celf_ep = [k_prod, i_node, mg, mg_ratio]
            celf_seq.append(celf_ep)
            for celf_item in celf_seq:
                if celf_ep[3] >= celf_item[3]:
                    celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                    celf_seq.pop()
                    break

        celf_seq.remove([-1, '-1', 0.0, 0.0])
        if len(celf_seq) == 0:
            mep = [-1, '-1', 0.0, 0.0]
        else:
            mep = celf_seq[0]

        return mep, celf_seq

    def generateDecomposedResult(self):
        s_mat, p_mat, c_mat = [[] for _ in range(self.num_product)], [[] for _ in range(self.num_product)], [[] for _ in range(self.num_product)]
        sspmis_ss = SeedSelectionPMIS(self.graph_dict, self.seed_cost_dict, self.product_list, self.total_budget, self.monte)

        for k in range(self.num_product):
            s_mat[k].append([set() for _ in range(self.num_product)])
            p_mat[k].append(0.0)
            c_mat[k].append(0.0)

            seed_set_t = [set() for _ in range(self.num_product)]
            cur_profit, cur_budget = 0.0, 0.0

            mep_g, celf_sequence = sspmis_ss.generateCelfSequence(k)
            mep_k_prod, mep_i_node, mep_profit = mep_g[0], mep_g[1], mep_g[2]

            while cur_budget < self.total_budget and mep_i_node != '-1':
                celf_sequence.remove(celf_sequence[0])
                seed_set_t[mep_k_prod].add(mep_i_node)

                cur_profit += mep_profit
                cur_budget += self.seed_cost_dict[mep_i_node]
                s_mat[k].append(copy.deepcopy(seed_set_t))
                p_mat[k].append(cur_profit)
                c_mat[k].append(round(cur_budget, 2))

                if len(celf_sequence) == 0:
                    break
                mep_g, celf_sequence = sspmis_ss.getMostValuableSeed(seed_set_t, celf_sequence, cur_profit, cur_budget)
                mep_k_prod, mep_i_node, mep_profit = mep_g[0], mep_g[1], mep_g[2]

        return s_mat, p_mat, c_mat


if __name__ == "__main__":
    data_set_name = "email_undirected"
    product_name = "r1p3n1"
    total_budget = 1
    pp_strategy = 1
    whether_passing_information_without_purchasing = bool(0)
    monte_carlo, eva_monte_carlo = 10, 100

    iniG = IniGraph(data_set_name)
    iniW = IniWallet(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()
    wallet_list = iniW.getWalletList(product_name)
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()

    sspmis = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, total_budget, monte_carlo)
    eva = Evaluation(graph_dict, seed_cost_dict, product_list, pp_strategy, whether_passing_information_without_purchasing)

    personal_prob_list = eva.setPersonalProbList(wallet_list)

    ### result: (list) [profit, budget, seed number per product, customer number per product, seed set] in this execution_time
    result = []
    avg_profit, avg_budget = 0.0, 0.0
    avg_num_k_seed, avg_num_k_pn = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
    profit_k_list, budget_k_list = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

    # -- initialization for each sample_number --
    ### now_profit, now_budget: (float) the profit and budget in this execution_time
    now_profit, now_budget = 0.0, 0.0
    mep_result = [0.0, [set() for _ in range(num_product)]]

    s_matrix, p_matrix, c_matrix = sspmis.generateDecomposedResult()

    ### bud_index: (list) the using budget index for products
    ### bud_bound_index: (list) the bound budget index for products
    bud_index, bud_bound_index = [len(kk) - 1 for kk in c_matrix], [0 for _ in range(num_product)]
    ### temp_bound_index: (list) the bound to exclude the impossible budget combination s.t. the k-budget is smaller than the temp bound
    temp_bound_index = [0 for _ in range(num_product)]

    while not operator.eq(bud_index, bud_bound_index):
        ### bud_pmis: (float) the budget in this pmis execution
        bud_pmis = 0.0
        for kk in range(num_product):
            bud_pmis += copy.deepcopy(c_matrix)[kk][bud_index[kk]]

        if bud_pmis <= total_budget:
            temp_bound_flag = 1
            for kk in range(num_product):
                if temp_bound_index[kk] > bud_index[kk]:
                    temp_bound_flag = 0
                    break
            if temp_bound_flag:
                temp_bound_index = copy.deepcopy(bud_index)

                # -- pmis execution --
                seed_set = [set() for _ in range(num_product)]
                for kk in range(num_product):
                    seed_set[kk] = copy.deepcopy(s_matrix)[kk][bud_index[kk]][kk]

                pro_acc = 0.0
                for _ in range(eva_monte_carlo):
                    pro_acc += sspmis.getSeedSetProfit(seed_set)
                pro_acc = round(pro_acc / eva_monte_carlo, 4)

                if pro_acc > mep_result[0]:
                    mep_result = [pro_acc, seed_set]

        pointer = num_product - 1
        while bud_index[pointer] == bud_bound_index[pointer]:
            bud_index[pointer] = len(c_matrix[pointer]) - 1
            pointer -= 1
        bud_index[pointer] -= 1

    pro_acc, pro_k_list_acc, pnn_k_list_acc = 0.0, [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    seed_set = mep_result[1]
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
        for ii in mep_result[1][kk]:
            budget_k_list[kk] += seed_cost_dict[ii]
            now_budget += seed_cost_dict[ii]
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
