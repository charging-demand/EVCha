import os
import numpy as np
import torch
import torch.utils.data
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
from .metrics import *
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power

# keshihua
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker



def get_adjacency_matrix2_new(distance_df_filename, num_of_vertices,threshold,
                         type_='connectivity',id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    import csv
    graph = np.array(pd.read_csv(distance_df_filename, header=None))
    # A = graph.astype(np.float64)
    # # 设置阈值
    # # threshold = 0.7  # 2022Q1 taxi
    # # threshold = 0.6  # 2022Q1 bus
    # threshold = threshold  # 2022Q1 private
    # # 大于阈值的元素设置为 1，小于等于阈值的元素设置为 0
    # A = np.where(graph > threshold, 1, 0)
    A = np.where(graph > threshold, 1.0, 0.0).astype(np.float64)
    return A






import pickle as pk
def pkload(file_path):
    file = open(file_path, 'rb')
    aaa = pk.load(file)
    file.close()
    return aaa

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def process_adj(gat_filename,gcn_filename,DEVICE):
    INPUT_STEP =24
    ADJTYPE = 'doubletransition'

    ########----2022Q1的参数
    TRAIN_NUM = 1258
    VAL_NUM = 403
    TEST_NUM = 403
    ##使用对应的移动性数据
    START_VAL_NUM = 1282
    START_TEST_NUM = 1709  # 1282+427

    # ########----2021Q2填充的参数
    # TRAIN_NUM = 1272
    # VAL_NUM = 408
    # TEST_NUM = 408
    # ##使用对应的移动性数据
    # START_VAL_NUM = 1296
    # START_TEST_NUM = 1728


    ################################################
    # dynamic_gat_adj
    dynamic_gat_adj = []
    dynamic_adjname = ['mobility']
    for dy_adjnale in dynamic_adjname:
        # gat_filename = os.path.join(os.path.dirname(__file__),
        #                         '../data/adj/'+str(DATA_PATH_old)+'/' + str(dy_adjnale) + '_adj_1012_' + vehicle_type + '_region.pk')
        gatfile = pkload(gat_filename)
        gat_values = list(gatfile.values())
        gat_values = np.array(gat_values)
        print('gat_values', gat_values.shape)
        temp = []
        for i in range(gat_values.shape[0]):
            adj = gat_values[i, :, :]
            norm_adj = get_normalized_adj(adj)
            temp.append(norm_adj)
        gat_norm = np.array(temp)
        dynamic_gat_adj.append(gat_norm)
    dynamic_gat_adj = np.array(dynamic_gat_adj).transpose(1, 0, 2, 3)  # [sample,D,N,N]
    print('dynamic_gat_adj shape ', dynamic_gat_adj.shape)
    dynamic_gat_adj = dynamic_gat_adj[8:2144]   ##wsw,2022-01-01到2022-03-30，使用旧的移动性
    # dynamic_gat_adj = dynamic_gat_adj[8:2168]   ##wsw,2021-04-01到2021-06-29，使用旧的移动性
    # dynamic_gat_adj = dynamic_gat_adj[1544:1784]  # wsw#出租车只用3.6-3.16共11天的数据集，1544应该是
    print('dynamic_gat_adj shape ', dynamic_gat_adj.shape)
    # train_dynamic_gat = dynamic_gat_adj[START_INDEX:TRAIN_NUM]
    # val_dynamic_gat = dynamic_gat_adj[START_VAL_NUM:START_VAL_NUM+VAL_NUM]
    # test_dynamic_gat = dynamic_gat_adj[START_TEST_NUM:START_TEST_NUM+TEST_NUM]

    train_dynamic_gat = dynamic_gat_adj[INPUT_STEP-1:TRAIN_NUM+INPUT_STEP-1]
    val_dynamic_gat = dynamic_gat_adj[START_VAL_NUM+INPUT_STEP-1:START_VAL_NUM+VAL_NUM+INPUT_STEP-1]
    test_dynamic_gat = dynamic_gat_adj[START_TEST_NUM+INPUT_STEP-1:START_TEST_NUM+TEST_NUM+INPUT_STEP-1]


    # dynamic_gcn_adj
    dynamic_gcn_adj = []
    dynamic_adjname = ['mobility']
    for dy_adjnale in dynamic_adjname:
        # gcn_filename = os.path.join(os.path.dirname(__file__),
        #                         '../data/adj/'+str(DATA_PATH_old)+'/' + str(dy_adjnale) + '_dynamic_gcn_graph_pk_' + vehicle_type + '_region.pk')
        gcnfile = pkload(gcn_filename)
        gcndata = gcnfile[ADJTYPE]
        dynamic_gcn_adj.append(gcndata)
    dynamic_gcn_adj = np.array(dynamic_gcn_adj).transpose(1, 0, 2, 3, 4)  # [sample,D,K,N,N]
    # dynamic_gcn_adj = dynamic_gcn_adj[1544:1784]  # wsw#出租车只用3.6-3.16共11天的数据集
    # dynamic_gcn_adj = dynamic_gcn_adj[8:2168]  ###wsw,2021-04-01到2021-06-29，使用旧的移动性
    dynamic_gcn_adj = dynamic_gcn_adj[8:2144]   ##wsw,2022-01-01到2022-03-30,使用旧的移动性
    print('dynamic_gcn_adj shape ', dynamic_gcn_adj.shape)
    # train_dynamic_gcn = dynamic_gcn_adj[START_INDEX:TRAIN_NUM]
    # val_dynamic_gcn = dynamic_gcn_adj[START_VAL_NUM:START_VAL_NUM+VAL_NUM]
    # test_dynamic_gcn = dynamic_gcn_adj[START_TEST_NUM:START_TEST_NUM+TEST_NUM]
    train_dynamic_gcn = dynamic_gcn_adj[INPUT_STEP-1:TRAIN_NUM+INPUT_STEP-1]
    val_dynamic_gcn = dynamic_gcn_adj[START_VAL_NUM+INPUT_STEP-1:START_VAL_NUM+VAL_NUM+INPUT_STEP-1]
    test_dynamic_gcn = dynamic_gcn_adj[START_TEST_NUM+INPUT_STEP-1:START_TEST_NUM+TEST_NUM+INPUT_STEP-1]

    train_dynamic_gat = torch.Tensor(train_dynamic_gat).to(DEVICE)
    val_dynamic_gat = torch.Tensor(val_dynamic_gat).to(DEVICE)
    test_dynamic_gat = torch.Tensor(test_dynamic_gat).to(DEVICE)
    print('train_dynamic_gat', train_dynamic_gat.shape, 'val_dynamic_gat.shape',val_dynamic_gat.shape,'test_dynamic_gat,shape', test_dynamic_gat.shape)

    train_dynamic_gcn = torch.Tensor(train_dynamic_gcn).to(DEVICE)
    val_dynamic_gcn = torch.Tensor(val_dynamic_gcn).to(DEVICE)
    test_dynamic_gcn = torch.Tensor(test_dynamic_gcn).to(DEVICE)  # [522,2,2,81,81]
    print('train_dynamic_gcn', train_dynamic_gcn.shape, 'val_dynamic_gcn.shape',val_dynamic_gcn.shape,'test_dynamic_gcn', test_dynamic_gcn.shape)
    return train_dynamic_gat,val_dynamic_gat, test_dynamic_gat, train_dynamic_gcn, val_dynamic_gcn, test_dynamic_gcn


def load_graphdata_channel1_new(graph_signal_matrix_filename,gat_filename,gcn_filename, DEVICE, batch_size):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        print(os.path.join(graph_signal_matrix_filename, mode + '_fill.npz'))
        _ = np.load(os.path.join(graph_signal_matrix_filename, mode + '_fill.npz'))
        data_dict['x_' + mode] = _['x'][:, :, :, 0:1]
        data_dict['y_' + mode] = _['y']
    scaler = StandardScaler(mean=data_dict['x_train'][..., 0].mean(),
                            std=data_dict['x_train'][..., 0].std())  # we only see the training data.
    for mode in ['train', 'val', 'test']:
        # continue
        data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
        data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])
    # #--------mean,std-------------
    # mean = data_dict['x_train'][..., 0].mean()
    # std = data_dict['x_train'][..., 0].std()

    train_dynamic_gat, val_dynamic_gat, test_dynamic_gat, train_dynamic_gcn, val_dynamic_gcn, test_dynamic_gcn = process_adj(gat_filename,gcn_filename,DEVICE)
    print(train_dynamic_gat.shape, val_dynamic_gat.shape, test_dynamic_gat.shape, train_dynamic_gcn.shape, val_dynamic_gcn.shape, test_dynamic_gcn.shape)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(np.transpose(data_dict['x_train'], (0, 2, 3, 1))).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(np.transpose(np.squeeze(data_dict['y_train'], axis=-1), (0, 2, 1))).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_dynamic_gat, train_dynamic_gcn, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(np.transpose(data_dict['x_val'], (0, 2, 3, 1))).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(np.transpose(np.squeeze(data_dict['y_val'], axis=-1), (0, 2, 1))).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_dynamic_gat, val_dynamic_gcn, val_target_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(np.transpose(data_dict['x_test'], (0, 2, 3, 1))).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(np.transpose(np.squeeze(data_dict['y_test'], axis=-1), (0, 2, 1))).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_dynamic_gat, test_dynamic_gcn, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return  train_loader,  val_loader,  test_loader, test_target_tensor, scaler


def compute_val_loss_mstgcn(net, val_loader_taxi, val_loader_private, criterion,  epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader_taxi: torch.utils.data.utils.DataLoader
    :param val_loader_bus: torch.utils.data.utils.DataLoader
    :param val_loader_private: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():
        tmp_taxi = []  # 记录了所有batch的taxi loss
        tmp_private = []  # 记录了所有batch的private loss

        val_loader_length_taxi = len(val_loader_taxi)  # nb of batch for taxi
        val_loader_length_private = len(val_loader_private)  # nb of batch for private

        for batch_index, (batch_data_taxi,  batch_data_private) in enumerate(
                zip(val_loader_taxi,  val_loader_private)):
            encoder_inputs_taxi, dynamic_gat_taxi, dynamic_gcn_taxi, labels_taxi = batch_data_taxi
            encoder_inputs_private, dynamic_gat_private, dynamic_gcn_private, labels_private = batch_data_private

            outputs_taxi,  outputs_private = net(
                [encoder_inputs_taxi,  encoder_inputs_private],
                [dynamic_gat_taxi, dynamic_gat_private],
                [dynamic_gcn_taxi, dynamic_gcn_private]
            )

            loss_taxi = criterion(outputs_taxi, labels_taxi)  # 计算taxi误差
            loss_private = criterion(outputs_private, labels_private)  # 计算private误差

            tmp_taxi.append(loss_taxi.item())
            tmp_private.append(loss_private.item())

            if batch_index % 100 == 0:
                print(
                    f'validation batch {batch_index + 1} / {val_loader_length_taxi}, taxi loss: {loss_taxi.item():.2f}')
                print(
                    f'validation batch {batch_index + 1} / {val_loader_length_private}, private loss: {loss_private.item():.2f}')

            if (limit is not None) and batch_index >= limit:
                break

        validation_loss_taxi = sum(tmp_taxi) / len(tmp_taxi)
        validation_loss_private = sum(tmp_private) / len(tmp_private)

        validation_loss = 0.8*validation_loss_taxi + 0.2*validation_loss_private

    return validation_loss



def evaluate_metrics_per_dimension(YS, YS_pred):
    num_dimensions = YS.shape[1]
    metrics_per_dimension = {
        'RMSE': [],
        'MAE': [],
        'MAPE': []
    }
    for dim in range(num_dimensions):
        YS_dim = YS[:, dim]
        YS_pred_dim = YS_pred[:, dim]
        RMSE1, MAE1, MAPE1 = metric(YS_dim,YS_pred_dim)
        metrics_per_dimension['RMSE'].append(round(RMSE1, 3))
        metrics_per_dimension['MAE'].append(round(MAE1, 3))
        metrics_per_dimension['MAPE'].append(round(MAPE1, 3))
    return metrics_per_dimension



def predict_and_save_results_mstgcn(net, data_loader_taxi, data_loader_private,
                                    data_target_tensor_taxi,  data_target_tensor_private,
                                    scaler_taxi,  scaler_private,
                                    global_step, params_path, type,logger):
    '''
    :param net: nn.Module
    :param data_loader_taxi: torch.utils.data.utils.DataLoader
    :param data_loader_bus: torch.utils.data.utils.DataLoader
    :param data_loader_private: torch.utils.data.utils.DataLoader
    :param data_target_tensor_taxi: tensor
    :param data_target_tensor_bus: tensor
    :param data_target_tensor_private: tensor
    :param scaler_taxi: 标准化缩放器
    :param scaler_bus: 标准化缩放器
    :param scaler_private: 标准化缩放器
    :param global_step: int
    :param params_path: the path for saving the results
    :param type: string
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor_taxi = data_target_tensor_taxi.cpu().numpy()
        data_target_tensor_private = data_target_tensor_private.cpu().numpy()
        loader_length = len(data_loader_taxi)  # nb of batch
        prediction_taxi = []  # 存储所有batch的output
        prediction_private = []  # 存储所有batch的output

        for batch_index, (batch_data_taxi, batch_data_private) in enumerate(
                    zip(data_loader_taxi,  data_loader_private)):
            encoder_inputs_taxi, dynamic_gat_taxi, dynamic_gcn_taxi, labels_taxi = batch_data_taxi
            encoder_inputs_private, dynamic_gat_private, dynamic_gcn_private, labels_private = batch_data_private

            outputs_taxi, outputs_private = net(
                    [encoder_inputs_taxi,  encoder_inputs_private],
                    [dynamic_gat_taxi,  dynamic_gat_private],
                    [dynamic_gcn_taxi,  dynamic_gcn_private]
            )

            prediction_taxi.append(outputs_taxi.detach().cpu().numpy())
            prediction_private.append(outputs_private.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))
                logger.info(f'predicting data set batch {batch_index + 1} / {loader_length}')

        prediction_taxi = np.concatenate(prediction_taxi, 0)  # (batch, T', 1)
        prediction_private = np.concatenate(prediction_private, 0)  # (batch, T', 1)

        ###########taxi
        # 计算误差
        excel_list = []
        data_target_tensor_taxi = scaler_taxi.inverse_transform(data_target_tensor_taxi)
        prediction_taxi = scaler_taxi.inverse_transform(prediction_taxi)
        output_filename_taxi = os.path.join(params_path, f'output_epoch_{global_step}_{type}_taxi')
        np.savez(output_filename_taxi, prediction=prediction_taxi, data_target_tensor=data_target_tensor_taxi)

        # 使用 metric 计算 RMSE、MAE 和 MAPE
        rmse_taxi , mae_taxi, mape_taxi = metric(prediction_taxi, data_target_tensor_taxi)
        metrics_YS1 = evaluate_metrics_per_dimension(data_target_tensor_taxi, prediction_taxi)
        print("taxi每个维度的RMSE:", metrics_YS1['RMSE'])
        print("taxi每个维度的MAE:", metrics_YS1['MAE'])
        print("taxi每个维度的MAPE:", metrics_YS1['MAPE'])
        logger.info(f"taxi每个维度的RMSE:{metrics_YS1['RMSE']}")
        logger.info(f"taxi每个维度的MAE:{metrics_YS1['MAE']}" )
        logger.info(f"taxi每个维度的MAPE:{metrics_YS1['MAPE']}" )

        print('taxi all RMSE: %.4f' % (rmse_taxi))
        print('taxi all MAE: %.4f' % (mae_taxi))
        print('taxi all MAPE: %.4f' % (mape_taxi))
        logger.info('taxi all RMSE: %.4f' % (rmse_taxi))
        logger.info('taxi all MAE: %.4f' % (mae_taxi))
        logger.info('taxi all MAPE: %.4f' % (mape_taxi))
        excel_list.extend([mae_taxi, rmse_taxi, mape_taxi])
        # print(excel_list)



        # 计算误差
        excel_list = []
        data_target_tensor_private = scaler_private.inverse_transform(data_target_tensor_private)
        prediction_private = scaler_private.inverse_transform(prediction_private)
        output_filename_private = os.path.join(params_path, f'output_epoch_{global_step}_{type}_private')
        np.savez(output_filename_private, prediction=prediction_private, data_target_tensor=data_target_tensor_private)
        # 使用 metric 计算 RMSE、MAE 和 MAPE
        rmse_private , mae_private, mape_private = metric(prediction_private, data_target_tensor_private)
        metrics_YS1 = evaluate_metrics_per_dimension(data_target_tensor_private, prediction_private)
        print("private每个维度的RMSE:", metrics_YS1['RMSE'])
        print("private每个维度的MAE:", metrics_YS1['MAE'])
        print("private每个维度的MAPE:", metrics_YS1['MAPE'])
        logger.info(f"private每个维度的RMSE:{metrics_YS1['RMSE']}")
        logger.info(f"private每个维度的MAE:{metrics_YS1['MAE']}" )
        logger.info(f"private每个维度的MAPE:{metrics_YS1['MAPE']}" )

        print('private all RMSE: %.4f' % (rmse_private))
        print('private all MAE: %.4f' % (mae_private))
        print('private all MAPE: %.4f' % (mape_private))
        logger.info('private all RMSE: %.4f' % (rmse_private))
        logger.info('private all MAE: %.4f' % (mae_private))
        logger.info('private all MAPE: %.4f' % (mape_private))
        excel_list.extend([mae_private, rmse_private, mape_private])
        # print(excel_list)





class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

