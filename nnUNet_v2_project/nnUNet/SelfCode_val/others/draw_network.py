from graphviz import Digraph
import matplotlib.pyplot as plt
from PIL import Image

# 創建一個有向圖
dot = Digraph(comment='2D U-Net Topology', node_attr={'fontname': 'Helvetica,Arial,sans-serif'},
              edge_attr={'fontname': 'Helvetica,Arial,sans-serif'})
dot.attr(rankdir='TB', size='8,8', dpi='300')

# 定義參數
patch_size = (512, 512)
batch_size = 10
median_shape = ('NA', 512, 512)
target_spacing = ('Na', 1, 1)

# 添加參數節點
dot.node('Params', f'Params\nPatch Size: {patch_size}\nBatch Size: {batch_size}\nMedian Shape: {median_shape}\nTarget Spacing: {target_spacing}', shape='box')

# 定義 U-Net 架構層數
num_layers = 5
feature_maps = [32 * (2**i) for i in range(num_layers)]

# 編碼器部分
encoder_nodes = []
for i in range(num_layers):
    node_name = f'Encoder_{i}'
    dot.node(node_name, f'Encoder Layer {i}\nFeature Maps: {feature_maps[i]}', shape='box')
    if i == 0:
        dot.edge('Params', node_name)
    else:
        dot.edge(encoder_nodes[i-1], node_name)
    encoder_nodes.append(node_name)

# 瓶頸部分
bottleneck_node = 'Bottleneck'
dot.node(bottleneck_node, f'Bottleneck\nFeature Maps: {feature_maps[-1] * 2}', shape='box')
dot.edge(encoder_nodes[-1], bottleneck_node)

# 解碼器部分
decoder_nodes = []
for i in range(num_layers - 1, -1, -1):
    node_name = f'Decoder_{i}'
    dot.node(node_name, f'Decoder Layer {i}\nFeature Maps: {feature_maps[i]}', shape='box')
    if i == num_layers - 1:
        dot.edge(bottleneck_node, node_name)
    else:
        dot.edge(decoder_nodes[-1], node_name)
    # 添加跳躍連接
    dot.edge(encoder_nodes[i], node_name, style='dashed')
    decoder_nodes.append(node_name)

# 輸出層
output_node = 'Output'
dot.node(output_node, 'Output\nFeature Maps: 1', shape='box')
dot.edge(decoder_nodes[-1], output_node)

# 渲染圖形
img_path = '2d_unet_topology'
dot.render(img_path, format='png', cleanup=True, view=False)

# 使用 matplotlib 顯示圖形
img = Image.open(f'{img_path}.png')
plt.rcParams['figure.dpi'] = 300
plt.imshow(img)
plt.axis('off')
plt.show()