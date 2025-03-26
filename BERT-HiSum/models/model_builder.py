import copy

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from crf import CRFLayer
import torch.nn.functional as F
import numpy as np


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


def get_sentence_embedding(sentence_mask, hidden_states, sentence_id):
    mask = (sentence_id == sentence_mask)
    mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sen_emb = torch.sum(hidden_states * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sen_emb.unsqueeze(1)

    
def get_centrality_score(embs):
    matrix = torch.matmul(embs, embs.T)
    # return torch.sum(matrix, 1)
    return F.normalize(torch.sum(matrix, 1), p=2, dim=0)

def get_cluster_id(dist_matrix):
    id_lst = dist_matrix.argmax(1)
    return id_lst

def get_index_lst(c, cluster_id_lst):
    index_lst = []
    for index, idx in enumerate(cluster_id_lst):
        if c == idx:
            index_lst.append(index)
    return index_lst

def get_global_score(clusters):
    return get_centrality_score(clusters)

def get_local_score(cluster_id_lst, embs):
    local_scores = torch.randn(len(cluster_id_lst))
    n = torch.max(cluster_id_lst)
    for c in range(n + 1):
        new_embs = embs[(c == cluster_id_lst)]
        index_lst = get_index_lst(c, cluster_id_lst)
        if len(new_embs) == 0:
            continue
        elif len(new_embs) == 1:
            local_scores[index_lst[0]] = 1.0
        else:
            cen_scores = get_centrality_score(new_embs)
            for index, score in zip(index_lst, cen_scores):
                # print(index, score)
                local_scores[index] = score
    return local_scores



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VarGMM, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)  
        self.fc22 = nn.Linear(hidden_dim, z_dim) 
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        #mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar, input_dim):
    ce = nn.MultiLabelSoftMarginLoss(reduction="sum")
    #BCE = ce(recon_x, x.view(-1, input_dim))
    BCE = ce(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_varGMM(varGMM, data_loader, input_dim, epochs=20, learning_rate=1e-3):
    varGMM.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(varGMM.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data[0].to(device) 
            optimizer.zero_grad()
            recon_batch, mu, logvar = varGMM(data)
            loss = loss_function(recon_batch, data, mu, logvar, input_dim)
            total_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
       # print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}')



def cluster_with_varGMM(varGMM, data, n_clusters=10):
    varGMM.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        data = data.to(device)
        encoded_data = varGMM.encode(data)[0].cpu()

    gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(encoded_data.numpy())
    cluster_centers = torch.tensor(gmm.means_, dtype=torch.float) 

    dist = torch.cdist(encoded_data, cluster_centers)

    return cluster_centers, dist

def get_global_local_centrality_score(sens_emb, varGMM):
    clusters, dist = cluster_with_varGMM(varGMM, sens_emb, len(sens_emb))
    cluster_id_lst = get_cluster_id(dist)
    global_scores = get_global_score(clusters)
    global_scores = F.normalize(global_scores, p=2, dim=0)
    # print(global_scores)
    local_scores = get_local_score(cluster_id_lst, sens_emb)
    # print(local_scores)
    gl_scores = [l.item() * global_scores[index].item() for l, index in zip(local_scores, cluster_id_lst)]

    return gl_scores, global_scores, local_scores



class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert_base_chinese', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert_base_chinese', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask, sentence_mask, tf_isf):
        if(self.finetune):
            top_vec, _ = self.model(x, token_type_ids = segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_type_ids = segs, attention_mask=mask)
        sentence_mask = torch.stack(sentence_mask)[:, :top_vec.size(1)].to('cuda')
        
        tf_isf = torch.stack(tf_isf)[:, :top_vec.size(1)].to('cuda')
        hidden_states = top_vec

        if sentence_mask is None:
            print("sentence_mask None warning!!!")
        else:

            sens = []
            for idx in range(1, torch.max(sentence_mask) + 1):
                sens.append(get_sentence_embedding(sentence_mask, hidden_states, idx))
            sens_embs = torch.cat(sens, dim=1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dataset = TensorDataset(hidden_states)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            input_dim =  hidden_states.size(2)
            hidden_dim =  hidden_states.size(1)
            z_dim = 20
            varGMM = VarGMM(input_dim, hidden_dim, z_dim).to(device)
            train_varGMM(varGMM, data_loader, input_dim, epochs=20, learning_rate=1e-3)
            new_weights = []
            for sens_emb, sens_mask in zip(sens_embs, sentence_mask):

                gl_scores1, global_scores, local_scores = get_global_local_centrality_score(sens_emb, varGMM)
                gl_scores2, global_scores, local_scores = get_global_local_centrality_score(sens_emb, varGMM)
                gl_scores3, global_scores, local_scores = get_global_local_centrality_score(sens_emb, varGMM)

                gl_scores = (np.array(gl_scores1) + np.array(gl_scores2) + np.array(gl_scores3)) / 3 
                gl_scores = gl_scores / (np.linalg.norm(gl_scores) + 1e-10)
                tmp = (sens_mask == 0).float() * 1.0

                for idx in range(1, torch.max(sens_mask) + 1):
                    tmp = tmp + gl_scores[idx-1] * (sens_mask == idx).float()
                
                new_weights.append(tmp.unsqueeze(-1))
            new_weights = torch.stack(new_weights).expand(hidden_states.size())

            alpha = 0.5
            top_vec = alpha * (hidden_states * new_weights) + (1-alpha) * hidden_states
        # LayerAttn can find in before Decoder

        return top_vec


class LayerAttn(nn.Module):
    def __init__(self, hidden_size):
        # super().__init__()
        super(LayerAttn, self).__init__()
        self.hidden_size = hidden_size
        self.attention_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None):
        attention_scores = self.attention_head(hidden_states).squeeze(-1) 
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1).unsqueeze(-1) 
        attended_states = attention_weights * hidden_states 
        attended_states = attended_states.sum(dim=1) 
        return attended_states

class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        self.tag_to_ix = {"B": 1, "I": 2, "O": 0, "<START>": 3, "<STOP>": 4}
        self.crf_layer = CRFLayer(self.tag_to_ix)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls, tgt):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
        # CRF
        #crf_scores = self.crf_layer.neg_log_likelihood(sent_scores, tgt)
        #return crf_scores, mask_cls

    def decode(self, src, segs, clss, mask_src, mask_cls, tgt):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        tag_seq = self.crf_layer.forward_test(sent_scores)
        return tag_seq


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.layerAttn = LayerAttn(hidden_size=config.d_model)
        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)



        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls, sentence_mask, tf_isf):
        top_vec = self.bert(src, segs, mask_src, sentence_mask, tf_isf)
        layerAttn_outputs = self.layerAttn(top_vec , attention_mask)
        dec_state = self.decoder.init_decoder_state(src, layerAttn_outputs)
        decoder_outputs, state = self.decoder(tgt[:, :-1], layerAttn_outputs, dec_state)
        return decoder_outputs, None
