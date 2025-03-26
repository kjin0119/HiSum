import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import config


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)



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


import numpy as np
class Encoder(nn.Module):
    def __init__(self, weight):
        super(Encoder, self).__init__()
        #self.embedding = nn.Embedding(config.args.vocab_size, config.emb_dim)
        self.embedding = nn.Embedding.from_pretrained(weight)
        for i in range(weight.shape[0]):
            if weight[i, 0] == 0:
                init_wt_normal(self.embedding.weight[i])

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    def forward(self, input, seq_lens, sentence_mask, tf_isf):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        hidden_states = encoder_outputs
        src_max_len = encoder_outputs.size(1)
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
            encoder_outputs = alpha * (hidden_states * new_weights) + (1-alpha) * hidden_states

        # LayerAttn can find in before Decoder
        
        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x t_k

        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.args.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.args.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        encoder_outputs = self.layerAttn(encoder_outputs, attention_mask)
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage



class PGNModel(object):
    def __init__(self, model_file_path=None, is_eval=False, device=None, embedding=None):
        encoder = Encoder(embedding)
        decoder = Decoder()
        layerAttn = LayerAttn(hidden_size=config.d_model)
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        reduce_state = reduce_state.to(device)
        self.encoder = encoder
        self.layerAttn = layerAttn
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def eval(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()
        self.reduce_state = self.reduce_state.eval()

    def train(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()
        self.reduce_state = self.reduce_state.train()
