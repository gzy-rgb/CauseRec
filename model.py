import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import random
import sys
from preprocess import *
from sklearn.preprocessing import LabelEncoder
import time


file_name = "result"
class CauseRec(nn.Module):
    def __init__(self, device):
        super(CauseRec, self).__init__()
        self.user_num = 603668
        self.test_user_num = 60366
        self.item_num = 367982
        self.batch_size = 1024
        self.seqence_length = 20
        self.embedding_dim = 64
        self.device = device
        self.sample_size = 5000
        self.negative_sample_num = 300
        self.replace_size = 5
        self.DCT_num = 4
        self.RCT_num = 4
        self.bank_size = 1024
        self.embedding_layer = torch.nn.Embedding(self.item_num + 1, self.embedding_dim)

        self.proposal_num = 20
        self.critical_set_size = int(self.proposal_num / 2)
        self.inessential_set_size = self.proposal_num - self.critical_set_size
        self.W1 = torch.nn.Parameter(data=torch.randn(128, self.embedding_dim), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(self.proposal_num, 128), requires_grad=True)

        self.memory_bank = torch.zeros((self.bank_size, self.embedding_dim))
        self.item_bank = []

        # DNN
        self.dnn_hidden_size = (256, 256, self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, self.dnn_hidden_size[0])
        self.fc2 = nn.Linear(self.dnn_hidden_size[0], self.dnn_hidden_size[1])
        self.fc3 = nn.Linear(self.dnn_hidden_size[1], self.dnn_hidden_size[2])
        self.bias = []
        for i in range(self.batch_size):
            self.bias += [i*self.negative_sample_num]*self.replace_size


    def gen_dict(self, item_profile):
        self.item_embedding_dict = torch.cat([torch.tensor([[0] * self.embedding_dim]).to(self.device),
                                              self.embedding_layer(item_profile.to(self.device))])

    def create_proposal_and_scoring(self, watch_movie, next_item):
        dim0, dim1 = watch_movie.shape
        watch_movie = torch.reshape(watch_movie, (1, dim0 * dim1))
        watch_movie_embedding = self.item_embedding_dict[watch_movie]
        watch_movie_embedding = torch.reshape(watch_movie_embedding, (dim0, dim1, -1))
        proposals_weight = F.softmax(torch.matmul(self.W2, F.tanh(torch.matmul(self.W1, torch.transpose(watch_movie_embedding, 1, 2)))), dim=2)
        watch_proposals = torch.matmul(proposals_weight, watch_movie_embedding)

        next_item_embedding = torch.unsqueeze(self.item_embedding_dict[next_item], 1)
        #cosine相似度作为分数
        item_score = torch.cosine_similarity(next_item_embedding, watch_movie_embedding, dim=2)
        _, ranked_item= torch.sort(item_score)
        item_score = torch.unsqueeze(item_score, 2)
        proposal_score = torch.matmul(proposals_weight, item_score)
        proposal_score = torch.squeeze(proposal_score)
        '''
        item_score = torch.matmul(next_item_embedding, torch.transpose(watch_movie_embedding, 1, 2))
        proposal_score = torch.matmul(proposals_weight, torch.transpose(item_score, 1, 2))
        proposal_score = torch.squeeze(proposal_score)
        '''
        _, ranked_proposal = torch.sort(proposal_score)
        return watch_proposals, ranked_proposal, ranked_item

    def create_embedding(self, watch_movie):
        dim0, dim1 = watch_movie.shape
        watch_movie = torch.reshape(watch_movie, (1, dim0 * dim1))
        watch_movie_embedding = self.item_embedding_dict[watch_movie]
        watch_movie_embedding = torch.reshape(watch_movie_embedding, (dim0, dim1, -1))
        watch_movie_embedding = torch.mean(watch_movie_embedding, dim=1)
        return watch_movie_embedding


    def create_proposal(self, watch_movie):
        dim0, dim1 = watch_movie.shape
        watch_movie = torch.reshape(watch_movie, (1, dim0 * dim1))
        watch_movie_embedding = self.item_embedding_dict[watch_movie]
        watch_movie_embedding = torch.reshape(watch_movie_embedding, (dim0, dim1, -1))
        proposals_weight = F.softmax(torch.matmul(self.W2, F.tanh(torch.matmul(self.W1, torch.transpose(watch_movie_embedding, 1, 2)))), dim=2)
        watch_proposals = torch.matmul(proposals_weight, watch_movie_embedding)
        return watch_proposals
    
    def forward(self, watch_proposals, flag=0):
        if flag == 0:
            watch_proposals = torch.mean(watch_proposals, dim=1)
        x = self.fc1(watch_proposals)
        x = self.fc2(x)
        user_embedding = F.tanh(self.fc3(x))
        return user_embedding

    def sampled_softmax(self, user_embedding, next_item, candidate_set):
        target_embedding = torch.sum(self.item_embedding_dict[next_item] * user_embedding, dim=1).view(len(user_embedding),1)
        product = torch.matmul(user_embedding, torch.transpose(self.item_embedding_dict[candidate_set], 0, 1))
        product = torch.cat([target_embedding, product], dim=1)
        return product

    #对item进行打分
    def scoring(self, watch_proposals, next_item, proposals_weight, watch_movie):
        batch_num = len(next_item)
        next_item_embedding = torch.unsqueeze(self.item_embedding_dict[next_item], 1)  # （，64）
        #similarity = F.cosine_similarity(next_item_embedding, watch_proposals, dim=2)
        proposal_score = torch.matmul(next_item_embedding, torch.transpose(watch_proposals, 1, 2))
        proposal_score = torch.matmul(proposals_weight, torch.transpose(proposal_score, 1, 2))
        proposal_score = torch.squeeze(proposal_score)
        _, ranked_idx = torch.sort(proposal_score,descending=True)
        return ranked_idx


    # 反事实操作
    def DCT(self, watch_proposals, watch_movie, ranked_item, next_item):
        batch_num = len(watch_proposals)

        item_raw_critical_set = torch.zeros((batch_num, self.critical_set_size), dtype=torch.int64).to(self.device)
        proposal_critical_set = torch.zeros((batch_num, self.critical_set_size, self.embedding_dim)).to(self.device)
        #other_item = torch.zeros((batch_num, self.negative_sample_num), dtype=torch.int64).to(self.device)

        for i in range(batch_num):
            item_raw_critical_set[i] = watch_movie[i, ranked_item[i]][0:self.critical_set_size]

        proposal_negative_user_embedding = torch.zeros((self.DCT_num, batch_num, self.embedding_dim)).to(device)

        item_critical_set_feature = torch.reshape(
            self.item_embedding_dict[torch.squeeze(torch.reshape(item_raw_critical_set, (1, -1)))],
            (batch_num, -1, self.embedding_dim))
        L2_item_critical_set = F.normalize(item_critical_set_feature, p=2, dim=2)  # (1024, 10, 64)

        for j in range(self.DCT_num):

            item_critical_set = item_raw_critical_set  # (1024, 10)
            other_item_feature = torch.squeeze(self.item_embedding_dict[self.item_bank])  # (1024, 64)
            L2_other_item_embedding = torch.transpose(F.normalize(other_item_feature, p=2, dim=1), 0, 1)  # (64, 1024)
            item_critical_prob_dis = F.softmax(torch.matmul(L2_item_critical_set, L2_other_item_embedding),
                                               dim=2)  # (1024, 10, 1024)

            item_critical_prob_dis = 1 - item_critical_prob_dis
            dim0, dim1, dim2 = item_critical_prob_dis.shape
            item_critical_prob_dis = torch.reshape(item_critical_prob_dis, (dim0 * dim1, dim2))

            item_critical_set = torch.reshape(item_critical_set, (-1, 1))  # (10240, 1)
            other_item = torch.reshape(torch.tensor(self.item_bank), (-1, 1))
            item_DCT_replacing_pos = torch.multinomial(item_critical_prob_dis, 1)  # (1024*10, 1)
            item_DCT_replaced_pos = []
            for i in range(batch_num):
                item_DCT_replaced_pos += random.sample(
                    range(i * self.critical_set_size, (i + 1) * self.critical_set_size),
                    self.replace_size)

            item_DCT_replacing_pos = torch.squeeze(item_DCT_replacing_pos[item_DCT_replaced_pos])
            item_critical_set[item_DCT_replaced_pos] = other_item[item_DCT_replacing_pos]

            item_critical_set = torch.reshape(item_critical_set, (dim0, dim1))

            #print(j)
            proposal_critical_set_, ranked_proposal, _ = self.create_proposal_and_scoring(item_critical_set, next_item)
            for i in range(batch_num):
                proposal_critical_set[i] = proposal_critical_set_[i, ranked_proposal[i]][0:self.critical_set_size]

            L2_proposal_critical_set = F.normalize(proposal_critical_set, p=2, dim=2)#(128, 10, 64)
            L2_proposal_bank_embedding = torch.transpose(F.normalize(self.memory_bank, p=2, dim=1), 0,
                                                      1)  # (100, 64)
            proposal_critical_prob_dis = F.softmax(torch.matmul(L2_proposal_critical_set, L2_proposal_bank_embedding),
                                          dim=1)  # (128, 10, 100)
            proposal_critical_prob_dis = 1 - proposal_critical_prob_dis
            dim0, dim1, dim2 = proposal_critical_prob_dis.shape
            proposal_critical_prob_dis = torch.reshape(proposal_critical_prob_dis, (dim0 * dim1, dim2))
            #print(critical_prob_dis)
            proposal_critical_set = torch.reshape(proposal_critical_set, (batch_num*self.critical_set_size, -1))#(128*10, 64)
            proposal_DCT_replacing_pos = torch.multinomial(proposal_critical_prob_dis, 1)  # (128*10, 1)
            proposal_DCT_replaced_pos = []
            for i in range(batch_num):
                proposal_DCT_replaced_pos += random.sample(range(i * self.critical_set_size, (i + 1) * self.critical_set_size),
                                                  self.replace_size)

            proposal_DCT_replacing_pos = torch.squeeze(proposal_DCT_replacing_pos[proposal_DCT_replaced_pos])
            proposal_critical_set[proposal_DCT_replaced_pos] = self.memory_bank[proposal_DCT_replacing_pos]
            proposal_critical_set = torch.reshape(proposal_critical_set, (batch_num, self.critical_set_size, -1))
            proposal_negative_user_embedding[j] = self.forward(torch.tensor(proposal_critical_set))

        return proposal_negative_user_embedding

    def RCT(self, watch_proposals, watch_movie, ranked_item, next_item):
        batch_num = len(watch_proposals)
        item_raw_inessential_set = torch.zeros((batch_num, self.critical_set_size), dtype=torch.int64).to(self.device)
        proposal_inessential_set = torch.zeros((batch_num, self.inessential_set_size, self.embedding_dim)).to(self.device)
        # other_item = torch.zeros((batch_num, self.negative_sample_num), dtype=torch.int64).to(self.device)

        for i in range(batch_num):
            item_raw_inessential_set[i] = watch_movie[i, ranked_item[i]][self.critical_set_size:]

        proposal_positive_user_embedding = torch.zeros((self.RCT_num, batch_num, self.embedding_dim)).to(device)

        item_inessential_set_feature = torch.reshape(
            self.item_embedding_dict[torch.squeeze(torch.reshape(item_raw_inessential_set, (1, -1)))],
            (batch_num, -1, self.embedding_dim))
        L2_item_inessential_set = F.normalize(item_inessential_set_feature, p=2, dim=2)  # (1024, 10, 64)

        for j in range(self.RCT_num):
            # print(j)
            item_inessential_set = item_raw_inessential_set  # (1024, 10)
            other_item_feature = torch.squeeze(self.item_embedding_dict[self.item_bank])  # (1024, 64)
            L2_other_item_embedding = torch.transpose(F.normalize(other_item_feature, p=2, dim=1), 0, 1)  # (64, 1024)
            item_inessential_prob_dis = F.softmax(torch.matmul(L2_item_inessential_set, L2_other_item_embedding),
                                                  dim=2)  # (1024, 10, 1024)
            dim0, dim1, dim2 = item_inessential_prob_dis.shape
            item_inessential_prob_dis = torch.reshape(item_inessential_prob_dis, (dim0 * dim1, dim2))

            item_inessential_set = torch.reshape(item_inessential_set, (-1, 1))  # (10240, 1)
            other_item = torch.reshape(torch.tensor(self.item_bank), (-1, 1))
            item_RCT_replacing_pos = torch.multinomial(item_inessential_prob_dis, 1)  # (1024*10, 1)
            item_RCT_replaced_pos = []
            for i in range(batch_num):
                item_RCT_replaced_pos += random.sample(
                    range(i * self.inessential_set_size, (i + 1) * self.inessential_set_size),
                    self.replace_size)

            item_RCT_replacing_pos = torch.squeeze(item_RCT_replacing_pos[item_RCT_replaced_pos])
            item_inessential_set[item_RCT_replaced_pos] = other_item[item_RCT_replacing_pos]

            item_inessential_set = torch.reshape(item_inessential_set, (dim0, dim1))

            proposal_inessential_set_, ranked_proposal, _ = self.create_proposal_and_scoring(item_inessential_set, next_item)
            for i in range(batch_num):
                proposal_inessential_set[i] = proposal_inessential_set_[i, ranked_proposal[i]][self.inessential_set_size:]


            L2_proposal_inessential_set = F.normalize(proposal_inessential_set, p=2, dim=2)  # (128, 10, 64)
            L2_proposal_bank_embedding = torch.transpose(F.normalize(self.memory_bank, p=2, dim=1), 0,
                                                         1)  # (100, 64)
            proposal_inessential_prob_dis = F.softmax(torch.matmul(L2_proposal_inessential_set, L2_proposal_bank_embedding),
                                                   dim=1)  # (128, 10, 100)
            dim0, dim1, dim2 = proposal_inessential_prob_dis.shape
            proposal_inessential_prob_dis = torch.reshape(proposal_inessential_prob_dis, (dim0 * dim1, dim2))
            # print(critical_prob_dis)
            proposal_inessential_set = torch.reshape(proposal_inessential_set,
                                                  (batch_num * self.inessential_set_size, -1))  # (128*10, 64)
            proposal_RCT_replacing_pos = torch.multinomial(proposal_inessential_prob_dis, 1)  # (128*10, 1)
            proposal_RCT_replaced_pos = []
            for i in range(batch_num):
                proposal_RCT_replaced_pos += random.sample(
                    range(i * self.inessential_set_size, (i + 1) * self.inessential_set_size),
                    self.replace_size)

            proposal_RCT_replacing_pos = torch.squeeze(proposal_RCT_replacing_pos[proposal_RCT_replaced_pos])
            proposal_inessential_set[proposal_RCT_replaced_pos] = self.memory_bank[proposal_RCT_replacing_pos]
            proposal_inessential_set = torch.reshape(proposal_inessential_set, (batch_num, self.inessential_set_size, -1))
            proposal_positive_user_embedding[j] = self.forward(torch.tensor(proposal_inessential_set))



        return proposal_positive_user_embedding
    
    def eva(self, pre, ground_truth):
        hit20, recall20, NDCG20, hit50, recall50, NDCG50 = (0, 0, 0, 0, 0, 0)
        epsilon = 0.1 ** 10
        for i in range(len(ground_truth)):
            one_DCG20 , one_recall20, IDCG20 , one_hit20, one_DCG50 , one_recall50, IDCG50 , one_hit50 = (0,0,0,0,0,0,0,0)
            top_20_item = pre[i][0:20].tolist()
            top_50_item = pre[i][0:50].tolist()
            positive_item = ground_truth[i]
            for pos, iid in enumerate(positive_item):
                if iid in top_20_item:
                    one_recall20 += 1
                    one_DCG20 += 1/np.log2(pos+2)
                if iid in top_50_item:
                    one_recall50 += 1
                    one_DCG50 += 1/np.log2(pos+2)

            for pos in range(one_recall20):
                IDCG20 += 1/np.log2(pos+2)
            for pos in range(one_recall50):
                IDCG50 += 1/np.log2(pos+2)

            NDCG20 += one_DCG20 / max(IDCG20, epsilon)
            NDCG50 += one_DCG50 / max(IDCG50, epsilon)
            top_20_item = set(top_20_item)
            top_50_item = set(top_50_item)
            positive_item = set(positive_item)
            if len(top_20_item & positive_item) > 0:
                hit20 += 1
            if len(top_50_item & positive_item) > 0:
                hit50 += 1
            recall20 += len(top_20_item & positive_item) / max(len(positive_item) ,epsilon)
            recall50 += len(top_50_item & positive_item) / max(len(positive_item), epsilon)
            #F1 += 2 * precision * recall / max(precision + recall, epsilon)

        return hit20, recall20, NDCG20, hit50, recall50, NDCG50

if __name__ == '__main__':
    device = torch.device("cuda:2")
    # 载入数据,设置参数
    datapath = "../dataset/book_data/"


    # 加载数据
    ml_train, ml_test, ml_valid = pd.read_csv(datapath + 'book_train.csv'), pd.read_csv(datapath + 'book_test.csv'), pd.read_csv(datapath + 'book_valid.csv')
    user_profile = list(set(ml_train['uid'].tolist()+ml_test['uid'].tolist()+ml_valid['uid'].tolist()))
    item_profile = list(set(ml_train['sid'].tolist()+ml_test['sid'].tolist()+ml_valid['sid'].tolist()))
    all_item = item_profile
    item_profile = torch.tensor( item_profile)
    #all_item = item_profile
    train_set = gen_train_set(ml_train)
    #valid_set = gen_test_set(ml_valid)
    test_set = gen_test_set(ml_test)
    train_model_input, train_label = gen_model_input(train_set, 20)
    #valid_model_input, valid_label = gen_model_input(valid_set, 20)
    test_model_input, test_label = gen_model_input(test_set, 20)
    #print(train_model_input)

    # 优化器
    model = CauseRec(device)
    model = model.to(device)
    lr = 0.003
    weight_decay=1e-07
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cross_entropy_loss = nn.CrossEntropyLoss()
    contrastive_loss = nn.TripletMarginLoss(margin=1, p=2)
    cosine_contrative_loss = nn.CosineEmbeddingLoss(0.5)

    record_length = len(train_model_input['user_id'])
    train_batch_num = (record_length // model.batch_size) + 1
    epochs = 30
    best_sum = 0
    bank_index = 0
    for epoch in range(epochs):
        for i in range(train_batch_num):
            model.gen_dict(item_profile)
            start = i * model.batch_size
            end = min((i + 1) * model.batch_size, record_length)
            # 计算batch
            user_batch = train_model_input['user_id'][start:end]
            next_item = torch.tensor(train_model_input['movie_id'][start:end], dtype=torch.int64).to(device)
            watch_movie = torch.tensor(train_model_input['hist_movie_id'][start:end], dtype=torch.int64).to(device)
            watch_movie_length = torch.tensor(train_model_input['hist_len'][start:end], dtype=torch.int64).to(device)
            target = torch.tensor([0] * len(user_batch), dtype=torch.int64).to(device)
            candidate_set = list(set(item_profile.tolist()) ^ set(next_item.tolist()))
            candidate_set = torch.tensor(random.sample(candidate_set, model.sample_size)).to(device)
            #print(watch_movie)
            watch_proposals, ranked_proposal, ranked_item = model.create_proposal_and_scoring(watch_movie, next_item)
            if epoch == 0 and i == 0 :
                with open(file_name + ".txt", "a") as f:
                    f.write('sample_size: ' + str(model.sample_size) + 'lr: ' + str(lr) + ' , ' + 'weight_decay: ' + str(weight_decay) + '\n')
                watch_proposals = torch.reshape(watch_proposals, (watch_proposals.shape[0]*watch_proposals.shape[1], -1))
                watch_movie = torch.reshape(watch_movie, (watch_movie.shape[0]*watch_movie.shape[1], -1))
                index = random.sample(range(0, len(watch_proposals)), model.bank_size)
                model.memory_bank = watch_proposals[index]
                model.item_bank = watch_movie[index]
            else:
                user_embedding = model.forward(watch_proposals)
                product = model.sampled_softmax(user_embedding, next_item, candidate_set)
                # print(product.shape)
                loss1 = cross_entropy_loss(product, target)

                #DCT
                negative_user_embedding = model.DCT(watch_proposals, watch_movie, ranked_item, next_item)
                #RCT
                positive_user_embedding = model.RCT(watch_proposals,  watch_movie, ranked_item, next_item)

                #loss2 = contrastive_loss(item_embedding_dict[next_item], user_embedding, negative_user_embedding)

                pos_target = torch.tensor([1]*len(user_batch)).to(device)
                neg_target = torch.tensor([-1]*len(user_batch)).to(device)
                loss_pos = cosine_contrative_loss(model.item_embedding_dict[next_item], user_embedding, pos_target) + cosine_contrative_loss(model.item_embedding_dict[next_item], torch.mean(positive_user_embedding, dim=0), pos_target)
                loss_neg = cosine_contrative_loss(model.item_embedding_dict[next_item], torch.mean(negative_user_embedding, dim=0), neg_target) + cosine_contrative_loss(model.item_embedding_dict[next_item], user_embedding, neg_target)
                loss2 = loss_pos + loss_neg
                loss3 = contrastive_loss(user_embedding, torch.mean(positive_user_embedding, dim=0), torch.mean(negative_user_embedding, dim=0))
                #loss3 = model.nce_loss(user_embedding, positive_user_embedding, negative_user_embedding)
                loss = loss1 + loss2 + loss3


            # print('epoch: ' + str(epoch) + ', ' + 'batch: ' + str(i)+', '+'loss: '+str(loss))
            #print(loss1, loss_pos, loss_neg)
                with open(file_name + ".txt", "a") as f:
                    if i != 1 and i % 200 == 1:
                        f.write('epoch: ' + str(epoch) + ', ' + 'batch: ' + str(i) + ', ' + 'loss: ' + str(loss.item())+'\n')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                watch_proposals = torch.reshape(watch_proposals,
                                                (watch_proposals.shape[0] * watch_proposals.shape[1], -1))
                watch_movie = torch.reshape(watch_movie, (watch_movie.shape[0] * watch_movie.shape[1], -1))
                index = random.sample(range(0, len(watch_proposals)), model.bank_size)
                model.memory_bank = watch_proposals[index]
                model.item_bank = watch_movie[index]
            # eva
            if i == train_batch_num-1:
                hit20, recall20, NDCG20, hit50, recall50, NDCG50 = (0, 0, 0, 0, 0, 0)
                with torch.no_grad():
                    test_batch_num = (len(test_model_input['user_id']) // 128) + 1
                    for j in range(test_batch_num):
                        start = j * 128
                        end = min((j + 1) * 128, len(test_model_input['user_id']))
                        #print(test_model_input['movie_id'][start:end])
                        next_item = test_model_input['movie_id'][start:end]
                        watch_movie = torch.tensor(test_model_input['hist_movie_id'][start:end],
                                                   dtype=torch.int64).to(device)
                        watch_movie_length = torch.tensor(test_model_input['hist_len'][start:end],
                                                          dtype=torch.int64).to(device)

                        watch_proposals = model.create_proposal(watch_movie)
                        user_embedding = model.forward(watch_proposals)
                        result = torch.matmul(user_embedding, torch.transpose(model.item_embedding_dict[1:], 0, 1))
                        # print(result)
                        _, pre = torch.sort(result, descending=True)
                        pre += 1
                        result = model.eva(pre, next_item)
                        hit20 += result[0]
                        recall20 += result[1]
                        NDCG20 += result[2]
                        hit50 += result[3]
                        recall50 += result[4]
                        NDCG50 += result[5]
                    hit20 = hit20 / model.test_user_num
                    recall20 = recall20 / model.test_user_num
                    NDCG20 = NDCG20 / model.test_user_num
                    hit50 = hit50 / model.test_user_num
                    recall50 = recall50 / model.test_user_num
                    NDCG50 = NDCG50 / model.test_user_num

                    with open("./test_result/"+ file_name + ".txt", "a") as f:
                        sum = 0
                        #torch.save(model.state_dict(), './best_model/best_baseline.pth')
                        sum = hit20 + recall20 + NDCG20 + hit50 + recall50 + NDCG50
                        if sum > best_sum:
                            best_sum = sum
                            f.write('epoch: ' + str(epoch) + ', ' + 'batch: ' + str(i) + ': ' + '\n')
                            # f.write('hit5: ' + str(hit5) + 'recall_5: ' + str(recall_5) + ' , ' + 'NDCG5: ' + str(NDCG5) + '\n')
                            # f.write('hit10: ' + str(hit10) + 'recall_10: ' + str(recall_10) + ' , ' + 'NDCG10: ' + str(NDCG10) + '\n')
                            f.write('hit20: ' + str(hit20) + 'recall_20: ' + str(recall20) + ' , ' + 'NDCG20: ' + str(
                                NDCG20) + '\n')
                            f.write('hit50: ' + str(hit50) + 'recall_50: ' + str(recall50) + ' , ' + 'NDCG50: ' + str(
                                NDCG50) + '\n')
