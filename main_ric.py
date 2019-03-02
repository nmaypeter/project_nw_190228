from SeedSelection_Random import *

if __name__ == "__main__":
    eva_monte_carlo = 100
    data_setting_seq = [1, 2, 3]
    wpiwp_seq = [bool(0), bool(1)]
    prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
    pps_seq = [1, 2, 3]
    begin_budget, total_budget = 1, 10
    sample_number = 20
    for data_setting in data_setting_seq:
        data_set_name = "email_undirected" * (data_setting == 1) + "dnc_email_directed" * (data_setting == 2) + "email_Eu_core_directed" * (data_setting == 3) + \
                        "WikiVote_directed" * (data_setting == 4) + "NetPHY_undirected" * (data_setting == 5)
        for wpiwp in wpiwp_seq:
            for prod_setting in prod_setting_seq:
                for prod_setting2 in prod_setting2_seq:
                    product_name = "r1p3n" + str(prod_setting) + "a" * (prod_setting2 == 2) + "b" * (prod_setting2 == 3)

                    iniG = IniGraph(data_set_name)
                    iniW = IniWallet(data_set_name)
                    iniP = IniProduct(product_name)

                    seed_cost_dict = iniG.constructSeedCostDict()
                    graph_dict = iniG.constructGraphDict()
                    product_list = iniP.getProductList()
                    num_node = len(seed_cost_dict)
                    num_product = len(product_list)

                    for bud in range(begin_budget, total_budget + 1):
                        start_time = time.time()
                        ssr_main = SeedSelectionR(graph_dict, seed_cost_dict, product_list, bud)
                        seed_set_sequence = []

                        for sample_count in range(sample_number):
                            print("@ mric seed selection @ data_set_name = " + data_set_name + ", wpiwp = " + str(wpiwp) +
                                  ", product_name = " + product_name + ", budget = " + str(bud) + ", sample_count = " + str(sample_count))
                            now_budget = 0.0
                            seed_set = [set() for _ in range(num_product)]

                            nban_set = [{ii for ii in graph_dict} for _ in range(num_product)]
                            nban_set = ssr_main.updateNotbanSet(nban_set, 0)
                            mep_g = ssr_main.selectRandomSeed(nban_set)
                            mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

                            # -- main --
                            while now_budget <= bud and mep_i_node != '-1':
                                seed_set[mep_k_prod].add(mep_i_node)

                                now_budget += seed_cost_dict[mep_i_node]

                                nban_set = ssr_main.updateNotbanSet(nban_set, now_budget)
                                mep_g = ssr_main.selectRandomSeed(nban_set)
                                mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

                            seed_set_sequence.append(seed_set)
                        ss_time = round(time.time() - start_time, 2)

                        result = [[] for _ in range(len(pps_seq))]
                        for pps in pps_seq:
                            pps_start_time = time.time()
                            avg_pro, avg_bud = 0.0, 0.0
                            avg_sn_k, avg_pnn_k = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
                            avg_pro_k, avg_bud_k = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                            eva_main = Evaluation(graph_dict, seed_cost_dict, product_list, pps, wpiwp)
                            wallet_list = iniW.getWalletList(product_name)
                            personal_prob_list = eva_main.setPersonalProbList(wallet_list)

                            for sample_count, sample_seed_set in enumerate(seed_set_sequence):
                                print("@ mric evaluation @ data_set_name = " + data_set_name + ", wpiwp = " + str(wpiwp) +
                                      ", product_name = " + product_name + ", budget = " + str(bud) + ", pps = " + str(pps) + ", sample_count = " + str(sample_count))
                                sample_pro_acc, sample_bud_acc = 0.0, 0.0
                                sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
                                sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                                for _ in range(eva_monte_carlo):
                                    pro, pro_k_list, pnn_k_list = eva_main.getSeedSetProfit(sample_seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
                                    sample_pro_acc += pro
                                    for kk in range(num_product):
                                        sample_pro_k_acc[kk] += pro_k_list[kk]
                                        sample_pnn_k_acc[kk] += pnn_k_list[kk]
                                sample_pro_acc = round(sample_pro_acc / eva_monte_carlo, 4)
                                for kk in range(num_product):
                                    sample_pro_k_acc[kk] = round(sample_pro_k_acc[kk] / eva_monte_carlo, 4)
                                    sample_pnn_k_acc[kk] = round(sample_pnn_k_acc[kk] / eva_monte_carlo, 2)
                                    sample_sn_k_acc[kk] = len(sample_seed_set[kk])
                                    for sample_seed in sample_seed_set[kk]:
                                        sample_bud_acc += seed_cost_dict[sample_seed]
                                        sample_bud_k_acc[kk] += seed_cost_dict[sample_seed]
                                        sample_bud_acc = round(sample_bud_acc, 2)
                                        sample_bud_k_acc[kk] = round(sample_bud_k_acc[kk], 2)

                                result[pps - 1].append([sample_pro_acc, sample_bud_acc, sample_sn_k_acc, sample_pnn_k_acc, sample_seed_set])
                                avg_pro += sample_pro_acc
                                avg_bud += sample_bud_acc
                                for kk in range(num_product):
                                    avg_sn_k[kk] += sample_sn_k_acc[kk]
                                    avg_pnn_k[kk] += sample_pnn_k_acc[kk]
                                    avg_pro_k[kk] += sample_pro_k_acc[kk]
                                    avg_bud_k[kk] += sample_bud_k_acc[kk]

                                print("total_time: " + str(round(time.time() - start_time, 2)) + "sec")
                                print(result[pps - 1][sample_count])
                                print("avg_profit = " + str(round(avg_pro / (sample_count + 1), 4)) + ", avg_budget = " + str(round(avg_bud / (sample_count + 1), 4)))
                                print("------------------------------------------")

                            avg_pro = round(avg_pro / sample_number, 4)
                            avg_bud = round(avg_bud / sample_number, 2)
                            for kk in range(num_product):
                                avg_sn_k[kk] = round(avg_sn_k[kk] / sample_number, 2)
                                avg_pnn_k[kk] = round(avg_pnn_k[kk] / sample_number, 2)
                                avg_pro_k[kk] = round(avg_pro_k[kk] / sample_number, 4)
                                avg_bud_k[kk] = round(avg_bud_k[kk] / sample_number, 2)

                            pps_time = round(time.time() - pps_start_time, 2)
                            total_time = round(ss_time + pps_time, 2)
                            path1 = "result/mhdic_pps" + str(pps) + "_wpiwp" * wpiwp
                            if not os.path.isdir(path1):
                                os.mkdir(path1)
                            path = "result/mhdic_pps" + str(pps) + "_wpiwp" * wpiwp + "/" + data_set_name + "_" + product_name
                            if not os.path.isdir(path):
                                os.mkdir(path)
                            fw = open(path + "/" + "b" + str(bud) + "_i" + str(sample_number) + ".txt", 'w')
                            fw.write("mhdic, pp_strategy = " + str(pps) + ", total_budget = " + str(bud) + ", wpiwp = " + str(wpiwp) + "\n" +
                                     "data_set_name = " + data_set_name + ", product_name = " + product_name + "\n" +
                                     "total_budget = " + str(bud) + ", sample_count = " + str(sample_number) + "\n" +
                                     "avg_profit = " + str(avg_pro) +
                                     ", avg_budget = " + str(avg_bud) + "\n" +
                                     "total_time = " + str(total_time) + ", avg_time = " + str(round(total_time / sample_number, 4)) + "\n")
                            fw.write("\nprofit_ratio =")
                            for kk in range(num_product):
                                fw.write(" " + str(avg_pro_k[kk]))
                            fw.write("\nbudget_ratio =")
                            for kk in range(num_product):
                                fw.write(" " + str(avg_bud_k[kk]))
                            fw.write("\nseed_number =")
                            for kk in range(num_product):
                                fw.write(" " + str(avg_sn_k[kk]))
                            fw.write("\ncustomer_number =")
                            for kk in range(num_product):
                                fw.write(" " + str(avg_pnn_k[kk]))
                            fw.write("\n")

                            for t, r in enumerate(result[pps - 1]):
                                fw.write("\n" + str(t) + "\t" + str(round(r[0], 4)) + "\t" + str(round(r[1], 4)) + "\t" + str(r[2]) + "\t" + str(r[3]) + "\t" + str(r[4]))
                            fw.close()