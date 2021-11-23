from networkx.algorithms.similarity import _n_choose_k
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import networkx.algorithms.community as nxcom
from community import community_louvain

def main(randseed):
    nodes = pd.read_csv('Proj_Data/node.csv', index_col=0)
    edges = pd.read_csv('Proj_Data/edges_with_qkv.csv', index_col=0)

    relation_df = edges[['node1', 'node2', 'q']].copy()

    relation = np.array(relation_df)

    G = nx.Graph()
    G.add_nodes_from(np.array(nodes.index))
    G.add_weighted_edges_from(relation)  # add weight from flow

    pos0 = nodes.loc[:, ['Long', 'Lat']]
    pos0 = np.array(pos0)

    vnode = pos0
    npos = dict(zip(np.array(nodes.index), vnode))  # 获取节点与坐标之间的映射关系，用字典表示

    partition = community_louvain.best_partition(G, resolution=50, weight='weight', random_state=randseed)

    par_df = pd.DataFrame(partition, index=[0]).T
    par_df.columns=['cls']

    nodes['cls'] = par_df['cls']
    nodes['q0'] = nodes['q']*12
    nodes['k0'] = nodes['q0']/nodes['v']

    data_new = pd.read_csv('./Proj_Data/2019-10-21_with_cord.csv', index_col=0)
    data_new = data_new.loc[data_new['Lane type']=='ML']

    data_new['cls'] = ''
    for i in nodes.index:
        ID = nodes.loc[i, 'ID']
        cls = nodes.loc[i, 'cls']
        data_new.loc[data_new['ID']==ID, 'cls'] = cls
    data_new['q0'] = data_new['q'] * 12
    data_new['k0'] = data_new['q0'] / data_new['Avg v']

    c = 0
    c_set = []
    color_set = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig_mfd = plt.figure()
    ax_mfd = fig_mfd.add_subplot(111)
    fig_net = plt.figure(figsize=[15,10])
    ax_net = fig_net.add_subplot(111)

    for i in edges.index:
        node1 = edges.loc[i, 'node1']
        node2 = edges.loc[i, 'node2']
        ax_net.plot([nodes.loc[node1, 'Long'], nodes.loc[node2, 'Long']], [nodes.loc[node1, 'Lat'], nodes.loc[node2, 'Lat']], 'black', lw=0.5)

    for i in range(len(nodes['cls'].drop_duplicates())):
        data_cls = data_new.loc[data_new['cls']==i].sort_values(by=['ID', 'Time'])
        q_cls = data_cls['q'].values
        if q_cls.reshape(-1, 288).shape[0] <= 1:
            continue
        q_cls_avg = q_cls.reshape(-1, 288).mean(axis=0)
        k_cls = data_cls['Avg k'].values
        k_cls_avg = k_cls.reshape(-1, 288).mean(axis=0)
        ax_mfd.scatter(k_cls_avg, q_cls_avg, s=.5, c=color_set[c])
        ax_mfd.set_xlabel('Occupancy', fontsize=12)
        ax_mfd.set_ylabel('Flow/[veh/5 min]', fontsize=12)
        
        lng = nodes.loc[nodes['cls']==i, 'Long']
        lat = nodes.loc[nodes['cls']==i, 'Lat']
        ax_net.scatter(lng, lat, s=5, c=color_set[c])
        ax_net.set_xlabel('Longitude', fontsize=12)
        ax_net.set_ylabel('Latitude', fontsize=12)
        c+=1
        c_set.append(i)


    # a = 3
    NSk_avg = 0
    NSk_set = []
    for a in c_set:
        NSk = 0
        for c in c_set:
            NSk += 2*nodes.loc[nodes['cls']==a, 'q'].std()**2/(nodes.loc[nodes['cls']==a, 'q'].std()**2+nodes.loc[nodes['cls']==c, 'q'].std()**2+(nodes.loc[nodes['cls']==a, 'q'].mean()-nodes.loc[nodes['cls']==c, 'q'].mean())**2)
        NSk_avg += NSk/len(c_set)
        NSk_set.append(NSk/len(c_set))
    NSk_avg = NSk_avg/len(c_set)

    TV = 0
    for c in c_set:
        TV += nodes.loc[nodes['cls']==c, 'q'].__len__()*nodes.loc[nodes['cls']==c, 'q'].std()**2



    fig_mfd.savefig('./Louvain_res/%i_fig_mfd_NSkavg_%.4f_TV_%.0f.png'%(randseed, NSk_avg, TV), dpi=500)
    fig_net.savefig('./Louvain_res/%i_fig_net_NSkavg_%.4f_TV_%.0f.png'%(randseed, NSk_avg, TV), dpi=500)

    pd.DataFrame(NSk_set).to_csv('./Louvain_res/%i_NSkset_NSkavg_%.4f_TV_%.0f.csv'%(randseed, NSk_avg, TV))

    return NSk_avg, TV, pd.DataFrame(NSk_set)

if __name__ == "__main__":
    NSk_avg_set = []
    TV_set = []
    res_df = pd.DataFrame([], columns=['NSk_avg', 'TV'])

    for randseed in range(100):
        NSk_avg, TV, _ = main(randseed)
        res_df.loc[randseed, 'NSk_avg'] = NSk_avg
        res_df.loc[randseed, 'TV'] = TV
    
    res_df.to_csv('./Louvain_res/res_df.csv')