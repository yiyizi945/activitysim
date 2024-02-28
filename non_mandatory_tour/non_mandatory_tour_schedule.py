import pandas as pd
import numpy as np
import collections.abc
import warnings
from patsy import dmatrix
# hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import choicemodels
warnings.filterwarnings("ignore")


class Non_Mandatory_Tour_schedule(object):
    def __init__(self, tour_data_path=None, chooser_data_path=None, alternatives_path=None, model_expression=None):
        if tour_data_path is None:      # 标定数据
            tour_data_path = "../data/standard_non_mantatory_tour_schedule_data.csv"
        tour_data = pd.read_csv(tour_data_path)
        if chooser_data_path is None:     # 选择者
            chooser_data_path = "../output/non_mandatory_tour_destination_tour.csv"
        chooser_data = pd.read_csv(chooser_data_path)
        if alternatives_path is None:    # 备选集
            alternatives_path = '../data/tdd_alts.csv'
        alternatives = pd.read_csv(alternatives_path)
        if model_expression is None:    # 模型表达式
            # c tot_tours == 1&is_famale + is_famale*_shopping + is_famale*_othmaint + is_famale*_eatout + auto_ownership&tot_tours == 1 + auto_ownership&tot_tours == 2
            # model_expression = 'tot_tours + shopping + eatout + social'
            model_expression = 'start_lt_6 + start_7 + start_8 + start_9 + start_12_16 + start_18_22 + start_mt_21 + \
                                end_lt_7 + end_6_10 + end_9_13 + end_12_15 + end_15 + end_16 + end_17 + end_18 + end_18_22 + \
                                end_mt_21 + duration_3_6 + duration_5_8 + duration_7_11 + duration_10_14 + duration_13_19 + \
                                is_shopping * start + is_shopping*start + is_shopping*duration + is_shopping*end + \
                                is_medical * start + is_medical*start + is_medical*duration + is_medical*end'
        self.tour_data = tour_data
        self.chooser_data = chooser_data
        self.alternatives = alternatives
        self.model_expression = model_expression

    def alternatives_process(self):
        # 活动表根据原数据添加新列
        # 开始时间是否小于6
        start_lt_6 = self.alternatives['start'].apply(lambda x: True if x < 6 else False)
        self.alternatives['start_lt_6'] = start_lt_6
        start_6 = self.alternatives['start'].apply(lambda x: True if x == 6 else False)
        self.alternatives['start_6'] = start_6
        start_7 = self.alternatives['start'].apply(lambda x: True if x == 7 else False)
        self.alternatives['start_7'] = start_7
        start_8 = self.alternatives['start'].apply(lambda x: True if x == 8 else False)
        self.alternatives['start_8'] = start_8
        start_9 = self.alternatives['start'].apply(lambda x: True if x == 9 else False)
        self.alternatives['start_9'] = start_9
        start_12_16 = self.alternatives['start'].apply(lambda x: True if 12 < x < 16 else False)
        self.alternatives['start_12_16'] = start_12_16
        start_15_19 = self.alternatives['start'].apply(lambda x: True if 15 < x < 19 else False)
        self.alternatives['start_15_19'] = start_15_19
        start_18_22 = self.alternatives['start'].apply(lambda x: True if 18 < x < 22 else False)
        self.alternatives['start_18_22'] = start_18_22
        start_mt_21 = self.alternatives['start'].apply(lambda x: True if 21 < x else False)
        self.alternatives['start_mt_21'] = start_mt_21
        end_lt_7 = self.alternatives['end'].apply(lambda x: True if x < 7 else False)
        self.alternatives['end_lt_7'] = end_lt_7
        end_6_10 = self.alternatives['end'].apply(lambda x: True if 6 < x < 10 else False)
        self.alternatives['end_6_10'] = end_6_10
        end_9_13 = self.alternatives['end'].apply(lambda x: True if 9 < x < 13 else False)
        self.alternatives['end_9_13'] = end_9_13
        end_12_15 = self.alternatives['end'].apply(lambda x: True if 12 < x < 15 else False)
        self.alternatives['end_12_15'] = end_12_15
        end_15 = self.alternatives['end'].apply(lambda x: True if x == 15 else False)
        self.alternatives['end_15'] = end_15
        end_16 = self.alternatives['end'].apply(lambda x: True if x == 16 else False)
        self.alternatives['end_16'] = end_16
        end_17 = self.alternatives['end'].apply(lambda x: True if x == 17 else False)
        self.alternatives['end_17'] = end_17
        end_18 = self.alternatives['end'].apply(lambda x: True if x == 18 else False)
        self.alternatives['end_18'] = end_18
        end_18_22 = self.alternatives['end'].apply(lambda x: True if 18 < x < 22 else False)
        self.alternatives['end_18_22'] = end_18_22
        end_mt_21 = self.alternatives['end'].apply(lambda x: True if x > 21 else False)
        self.alternatives['end_mt_21'] = end_mt_21
        duration_lt_2 = self.alternatives['duration'].apply(lambda x: True if x < 2 else False)
        self.alternatives['duration_lt_2'] = duration_lt_2
        duration_3_6 = self.alternatives['duration'].apply(lambda x: True if 3 < x < 6 else False)
        self.alternatives['duration_3_6'] = duration_3_6
        duration_5_8 = self.alternatives['duration'].apply(lambda x: True if 5 < x < 8 else False)
        self.alternatives['duration_5_8'] = duration_5_8
        duration_7_11 = self.alternatives['duration'].apply(lambda x: True if 7 < x < 11 else False)
        self.alternatives['duration_7_11'] = duration_7_11
        duration_10_14 = self.alternatives['duration'].apply(lambda x: True if 10 < x < 14 else False)
        self.alternatives['duration_10_14'] = duration_10_14
        duration_13_19 = self.alternatives['duration'].apply(lambda x: True if 13 < x < 19 else False)
        self.alternatives['duration_13_19'] = duration_13_19
        return self.alternatives

    def tour_data_process(self):
        # 活动表根据原数据添加新列
        is_shopping = self.tour_data['tour_type'] == 'shopping'
        self.tour_data['is_shopping'] = is_shopping
        is_medical = self.tour_data['tour_type'] == 'medical'
        self.tour_data['is_medical'] = is_medical
        is_social = self.tour_data['tour_type'] == 'social'
        self.tour_data['is_social'] = is_social
        is_eatout = self.tour_data['tour_type'] == 'eatout'
        self.tour_data['is_eatout'] = is_eatout
        return self.tour_data

    def chooser_data_process(self):
        # 合并个人表及家庭表
        # 活动表根据原数据添加新列
        is_shopping = self.chooser_data['tour_type'] == 'shopping'
        self.chooser_data['is_shopping'] = is_shopping
        is_medical = self.chooser_data['tour_type'] == 'medical'
        self.chooser_data['is_medical'] = is_medical
        is_social = self.chooser_data['tour_type'] == 'social'
        self.chooser_data['is_social'] = is_social
        is_eatout = self.chooser_data['tour_type'] == 'eatout'
        self.chooser_data['is_eatout'] = is_eatout
        return self.chooser_data

    def segment_fit(self, name, mct, model_expression):
        print('SPEC_SEGMENTS:', name)
        results = choicemodels.MultinomialLogit(data=mct,
                                                model_expression=model_expression,
                                                observation_id_col='tour_id',
                                                choice_col='chosen',
                                                alternative_id_col='tdd')
        return results

    def segment_cal_pop(self, name, m, mct_simu):
        print('section:', name)
        fitted_model = m.fit()   # 标定的模型
        prop = fitted_model.probabilities(mct_simu)  # 用标定模型去预测选择概率
        return prop

    def df_split(self, df, split_col):
        groups = df.groupby(split_col)
        num_level = len(df[split_col].unique().tolist())  # 判断等级数
        return groups, num_level

    def calibration_process(self):
        # 标定结果
        choosers = self.tour_data   # 标定数据
        choosers = choosers.fillna(0)
        alternatives = self.alternatives
        # print('choosers', choosers)
        choosers.set_index('tour_id', inplace=True)
        # print('alternatives', alternatives)
        alternatives.set_index('tdd', inplace=True)
        model_expression = self.model_expression
        merged_tb = choicemodels.tools.MergedChoiceTable(choosers, alternatives, chosen_alternatives='tdd', sample_size=190)
        # print(merged_tb.to_frame())
        results = choicemodels.MultinomialLogit(data=merged_tb,
                                                model_expression=model_expression,
                                                observation_id_col='tour_id',
                                                choice_col='chosen',
                                                alternative_id_col='tdd')
        print(results.fit())
        self.results = results

    def simulated_selection(self):
        chooser_data = self.chooser_data
        alternatives = self.alternatives
        choosers = chooser_data   # 选择者
        choosers = choosers.fillna(0)
        # choosers = choosers.iloc[:1000, :]
        # choosers.set_index('tour_id', inplace=True)
        mct_simu = choicemodels.tools.MergedChoiceTable(choosers, alternatives, sample_size=50)
        name = 'non_mandatory_tour_scheduling'
        m = self.results
        prop = self.segment_cal_pop(name, m, mct_simu)
        result_df = choicemodels.tools.simulation.monte_carlo_choices(prop)
        print(result_df)
        result_df = pd.concat([choosers, result_df], axis=1)
        print(result_df)
        result_df.reset_index(inplace=True)
        del result_df['obs_id']
        result_df.set_index('tour_id', inplace=True)
        result_df.to_csv('../output/non_mandatory_tour_schedule_tour.csv')
        # df = choosers
        # # 每段的大小
        # chunk_size = 1000
        # # 计算分割后的块数
        # num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
        # # 分割 DataFrame
        # chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        # print(chunks)
        # # 打印每个块的大小
        # result_df = {}
        # for i, chunk in enumerate(chunks):
        #     print(f"Chunk {i + 1}: {len(chunk)} rows")
        #     chunk.set_index('tour_id', inplace=True)
        #     mct_simu = choicemodels.tools.MergedChoiceTable(chunk, alternatives, sample_size=50)
        #     name = 'non_mandatory_tour_scheduling'
        #     m = self.results
        #     prop = self.segment_cal_pop(name, m, mct_simu)
        #     result_df[i] = choicemodels.tools.simulation.monte_carlo_choices(prop)


        # result_df = pd.merge(result_df, choosers, on=['tour_id'])
        # result_df = pd.merge(result_df, alternatives, on=['tdd'])
        # print(result_df)
        # result_df.to_csv('./output/non_mandatory_tour_schedule_tour.csv')

    # def simulated_selection(self):
    #     chooser_data = self.chooser_data
    #     alternatives = self.alternatives
    #     choosers = chooser_data   # 选择者
    #     choosers = choosers.fillna(0)
    #     choosers.set_index('tour_id', inplace=True)
    #     # print('choosers', choosers)
    #     # alternatives.set_index('tdd', inplace=True)
    #     # print('alternatives', alternatives)
    #     df = choosers
    #     split_col = 'tour_type'
    #     spec_segments, num_level = self.df_split(df, split_col)
    #     grouped_list = [(key) for key, group in spec_segments]
    #     print(grouped_list)
    #     for i in range(1, num_level+1):
    #         exec('spec_segments_{} = spec_segments.get_group("{}")'.format(i,  grouped_list[i-1]))
    #     groups = spec_segments
    #     choice_list = {}
    #     choosers_list = {}
    #     alternatives = self.alternatives
    #     for name, group in groups:
    #         choosers = group   # 选择者
    #         choosers = choosers.fillna(0)
    #         mct_simu = choicemodels.tools.MergedChoiceTable(choosers, alternatives, sample_size=190)
    #         print(mct_simu.to_frame())
    #         name = 'non_mandatory_tour_scheduling'
    #         m = self.results
    #         prob = self.segment_cal_pop(name, m, mct_simu)
    #         # result_df = choicemodels.tools.simulation.monte_carlo_choices(prop)
    #         choice_list[i] = choicemodels.tools.simulation.monte_carlo_choices(prob)
    #         choosers_list[i] = choosers
    #     choosers_df = pd.concat([choosers_list[i] for i in choosers_list.keys()])
    #     result_df = pd.concat([choice_list[i] for i in choice_list.keys()])
    #     result_df = pd.merge(result_df, choosers_df, on=['tour_id'])
    #     print(result_df)
    #     result_df.to_csv('./output/non_mandatory_tour_schedule_tour.csv')


if __name__ == '__main__':
    nmts = Non_Mandatory_Tour_schedule()
    nmts.alternatives_process()
    nmts.tour_data_process()
    print('tour_data\n', nmts.tour_data)
    # print('household_data\n', nmtf.household_data)
    # print('person_merged_data\n', nmtf.person_merged_data)
    print('alternatives\n', nmts.alternatives)
    nmts.calibration_process()
    nmts.chooser_data_process()
    nmts.simulated_selection()
