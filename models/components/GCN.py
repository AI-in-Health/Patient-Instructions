import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp


def scipy_coo_matrix_to_torch_sparse_tensor(matrix):
	values = matrix.data
	indices = np.vstack((matrix.row, matrix.col))
	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = matrix.shape
	matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
	return matrix


def normalize_adjacent_matrix_symmetrical_freq(adjacent_matrix, return_sparse_tensor=True):
    # https://arxiv.org/pdf/1609.02907.pdf
    # A_hat is the adjacency matrix of the undirected graph G with added self-connection
    # D_hat is the degree matrix based on A_hat
    A_hat = adjacent_matrix          
    D_hat = A_hat.sum(1)
    D_hat_ = np.diag(D_hat ** (-0.5))
    normalized_adjacent_matrix = np.dot(np.dot(D_hat_, A_hat), D_hat_)
    if return_sparse_tensor:
        normalized_adjacent_matrix = sp.coo_matrix(normalized_adjacent_matrix)
        return scipy_coo_matrix_to_torch_sparse_tensor(normalized_adjacent_matrix)
    return nn.Parameter(torch.FloatTensor(normalized_adjacent_matrix), requires_grad=False)


def normalize_adjacent_matrix_symmetrical(adjacent_matrix, return_sparse_tensor=False):
    adjacent_matrix[adjacent_matrix>0] = 1
    for i in range(len(adjacent_matrix)):
        adjacent_matrix[i, i] = 1
    
    return normalize_adjacent_matrix_symmetrical_freq(adjacent_matrix, return_sparse_tensor)


def normalize_adjacent_matrix_freq(adjacent_matrix, return_sparse_tensor=False):
    A_hat = adjacent_matrix          
    D_hat = A_hat.sum(1)
    D_hat_ = np.diag(D_hat ** (-1))
    normalized_adjacent_matrix = np.dot(D_hat_, A_hat)
    if return_sparse_tensor:
        normalized_adjacent_matrix = sp.coo_matrix(normalized_adjacent_matrix)
        return scipy_coo_matrix_to_torch_sparse_tensor(normalized_adjacent_matrix)

    return nn.Parameter(torch.FloatTensor(normalized_adjacent_matrix), requires_grad=False)


def normalize_adjacent_matrix(adjacent_matrix, return_sparse_tensor=False):
    adjacent_matrix[adjacent_matrix>0] = 1
    for i in range(len(adjacent_matrix)):
        adjacent_matrix[i, i] = 1

    return normalize_adjacent_matrix_freq(adjacent_matrix, return_sparse_tensor)


def prepare_adjacent_matrix_given_threshold(path, counts_path, threshold, print_info=True):
    matrix = np.load(path)

    if counts_path is not None:
        diag_counts = np.load(counts_path)
        diag_counts[diag_counts<=threshold] = 0
        degree = diag_counts
    else:
        matrix[matrix<=threshold] = 0
        degree = matrix.sum(0)

    n_total = len(matrix)
    n_activate = len(degree[degree > 0])
    
    if print_info:
        print(threshold, n_total, n_activate)

    ids = []
    new_matrix = []
    for i, d in enumerate(degree):
        if d > 0:
            new_matrix.append(matrix[i, :][np.newaxis, :])
            ids.append(i)
    
    new_matrix = np.concatenate(new_matrix, axis=0)

    new_matrix2 = []
    for i, d in enumerate(degree):
        if d > 0:
            new_matrix2.append(new_matrix[:, i][:, np.newaxis])
    final_matrix = np.concatenate(new_matrix2, axis=1)

    assert final_matrix.shape == (n_activate, n_activate)

    if counts_path is not None:
        index = -1
        for count in diag_counts:
            if count > 0:
                index += 1
                assert final_matrix[index].max() <= count
                final_matrix[index, index] = count
                final_matrix[index] /= count

    return final_matrix, ids


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, 
            dim_input: int, 
            dim_output: int, 
            dropout_rate: float = 0.1,
        ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.constant_(self.net[0].bias, 0)

    def forward(self, input_embs: torch.Tensor, adjacent_matrix: torch.Tensor): 
        #outputs = torch.spmm(adjacent_matrix, input_embs)
        if input_embs is None:
            outputs = adjacent_matrix
        else:
            outputs = torch.mm(adjacent_matrix, input_embs)
        outputs = self.net(outputs)
        return outputs


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, 
            adjacent_matrix: torch.Tensor,
            dim_input: int, 
            dim_hidden: int, 
            dropout_rate: float = 0.5,
            num_layers: int = 1,
        ):
        super().__init__()
        self.adjacent_matrix = adjacent_matrix
        
        layers = []
        for _ in range(num_layers):
            layer = GraphConvolutionalLayer(
                dim_input=dim_input if _ == 0 else dim_hidden,
                dim_output=dim_hidden,
                dropout_rate=dropout_rate,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, input_embs: torch.Tensor):
        if input_embs is not None:
            assert input_embs.dim() == 2            

        hidden_states = input_embs
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.adjacent_matrix)
        
        return hidden_states


class GraphNet(nn.Module):
    def __init__(self, config, ):
        super().__init__()
        adjacent_matrix, ids = prepare_adjacent_matrix_given_threshold(
            config.adjacent_matrix_path,
            counts_path=None if not config.gcn_freq else config.adjacent_matrix_counts_path,
            threshold=config.adjacent_matrix_threshold,
            print_info=True
        )

        print(config.normalize_method)
        func = None
        if config.normalize_method == 'freq':
            func = normalize_adjacent_matrix_freq
        elif config.normalize_method == 'co':
            func = normalize_adjacent_matrix
        elif config.normalize_method == 'sym_freq':
            func = normalize_adjacent_matrix_symmetrical_freq
        elif config.normalize_method == 'sym_co':
            func = normalize_adjacent_matrix_symmetrical
            
        if func is not None:
            adjacent_matrix = func(adjacent_matrix)
        else:
            adjacent_matrix = nn.Parameter(torch.FloatTensor(adjacent_matrix), requires_grad=False)

        if config.gcn_bert_embs_path is not None:
            self.embs = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load(config.gcn_bert_embs_path)),
                freeze=True
            )
        else:
            self.embs = nn.Embedding(len(ids), config.d_model)
            self.embs.weight.data.normal_(mean=0.0, std=1.0)

        dim_input = self.embs.weight.shape[1]

        self.gcn = GraphConvolutionalNetwork(
            adjacent_matrix=adjacent_matrix,
            dim_input=dim_input,
            dim_hidden=config.d_model,
            dropout_rate=config.dropout_rate,
            num_layers=config.gcn_num_layers,
        )
    
    def forward(self):
        if hasattr(self, 'embs'):
            return self.gcn(self.embs.weight)
        else:
            return self.gcn(None)
