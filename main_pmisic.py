from SeedSelection_PMIS import *

if __name__ == "__main__":
    monte_carlo, eva_monte_carlo = 10, 100
    for data_setting in [1, 2, 3]:
        data_set_name = "email_undirected" * (data_setting == 1) + "dnc_email_directed" * (data_setting == 2) + \
                        "email_Eu_core_directed" * (data_setting == 3) + "WikiVote_directed" * (data_setting == 4) + \
                        "NetPHY_undirected" * (data_setting == 5)
        for pps in [1, 2, 3]:
            for wpiwp in [bool(0), bool(1)]:
                for prod_setting in [1, 2]:
                    for prod_setting2 in [1, 2, 3]:
                        product_name = "r1p3n" + str(prod_setting) + "a" * (prod_setting2 == 2) + "b" * (prod_setting2 == 3)

                        total_budget = 10
                        sample_number, sample_output_number = 10, 10

                        iniG = IniGraph(data_set_name)
                        iniW = IniWallet(data_set_name)
                        iniP = IniProduct(product_name)

                        seed_cost_dict = iniG.constructSeedCostDict()
                        graph_dict = iniG.constructGraphDict()
                        product_list = iniP.getProductList()
                        wallet_list = iniW.getWalletList(product_name)
                        num_node = len(seed_cost_dict)
                        num_product = len(product_list)

                        for bud in range(1, total_budget + 1):
                            start_time = time.time()

                            sspmis_main = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, bud, monte_carlo)
                            eva_main = Evaluation(graph_dict, seed_cost_dict, product_list, pps, wpiwp)

                            personal_prob_list = eva_main.setPersonalProbList(wallet_list)

                            result = []
                            avg_profit, avg_budget = 0.0, 0.0
                            avg_num_k_seed, avg_num_k_pn = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
                            profit_k_list, budget_k_list = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                            for sample_count in range(sample_number):
                                print("pp_strategy = " + str(pps) + ", wpiwp = " + str(wpiwp) + ", data_set_name = " + data_set_name +
                                      ", product_name = " + product_name + ", budget = " + str(bud) + ", sample_count = " + str(sample_count))
                                now_budget = 0.0
                                seed_set = [set() for _ in range(num_product)]
                                mep_result = [0.0, [set() for _ in range(num_product)]]

                                s_matrix, p_matrix, c_matrix = sspmis_main.generateDecomposedResult()

                                bud_index, bud_bound_index = [len(kk) - 1 for kk in c_matrix], [0 for _ in range(num_product)]
                                temp_bound_index = [0 for _ in range(num_product)]

                                while not operator.eq(bud_index, bud_bound_index):
                                    ### bud_pmis: (float) the budget in this pmis execution
                                    bud_pmis = 0.0
                                    for kk in range(num_product):
                                        bud_pmis += copy.deepcopy(c_matrix)[kk][bud_index[kk]]

                                    if bud_pmis <= bud:
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
                                                pro_acc += sspmis_main.getSeedSetProfit(seed_set)
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
                                    pro, pro_k_list, pnn_k_list = eva_main.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
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

                                # -- result --
                                how_long = round(time.time() - start_time, 2)
                                print("total_time: " + str(how_long) + "sec")
                                print(result[sample_count])
                                print("avg_profit = " + str(round(avg_profit / (sample_count + 1), 4)) + ", avg_budget = " + str(round(avg_budget / (sample_count + 1), 4)))
                                print("------------------------------------------")

                                if (sample_count + 1) % sample_output_number == 0:
                                    path1 = "result/mpmisic_pps" + str(pps) + "_wpiwp" * wpiwp
                                    if not os.path.isdir(path1):
                                        os.mkdir(path1)
                                    path = "result/mpmisic_pps" + str(pps) + "_wpiwp" * wpiwp + "/" +\
                                           data_set_name + "_" + product_name
                                    if not os.path.isdir(path):
                                        os.mkdir(path)
                                    fw = open(path + "/" + "b" + str(bud) + "_i" + str(sample_count + 1) + ".txt", 'w')
                                    fw.write("mpmisic, pp_strategy = " + str(pps) + ", total_budget = " + str(bud) + ", wpiwp = " + str(wpiwp) + "\n" +
                                             "data_set_name = " + data_set_name + ", product_name = " + product_name + "\n" +
                                             "total_budget = " + str(bud) + ", sample_count = " + str(sample_count + 1) + "\n" +
                                             "avg_profit = " + str(round(avg_profit / (sample_count + 1), 4)) +
                                             ", avg_budget = " + str(round(avg_budget / (sample_count + 1), 4)) + "\n" +
                                             "total_time = " + str(how_long) + ", avg_time = " + str(round(how_long / (sample_count + 1), 4)) + "\n")
                                    fw.write("\nprofit_ratio =")
                                    for kk in range(num_product):
                                        fw.write(" " + str(round(profit_k_list[kk] / (sample_count + 1), 4)))
                                    fw.write("\nbudget_ratio =")
                                    for kk in range(num_product):
                                        fw.write(" " + str(round(budget_k_list[kk] / (sample_count + 1), 4)))
                                    fw.write("\nseed_number =")
                                    for kk in range(num_product):
                                        fw.write(" " + str(round(avg_num_k_seed[kk] / (sample_count + 1), 4)))
                                    fw.write("\ncustomer_number =")
                                    for kk in range(num_product):
                                        fw.write(" " + str(round(avg_num_k_pn[kk] / (sample_count + 1), 4)))
                                    fw.write("\n")

                                    for t, r in enumerate(result):
                                        fw.write("\n" + str(t) + " " + str(round(r[0], 4)) + " " + str(round(r[1], 4)) + " " + str(r[2]) + " " + str(r[3]) + " " + str(r[4]))
                                    fw.close()