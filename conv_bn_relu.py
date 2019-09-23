# coding: utf-8
import numpy as np

def conv_with_bias(x, w, b, pad, stride):
    out = None
    N, C, H, W = x.shape
    F, C, HH, WW= w.shape
    #X = np.pad(x, ((0,0), (0, 0), (pad, pad),(pad, pad)), 'constant')
    X = padding(x, pad)
    Hn = 1 + int((H + 2 * pad - HH) / stride)
    Wn = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hn, Wn))
    for n in range(N):
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    # 一个stide读取的feature map展成一行
                    data = X[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW].reshape(1, -1)
                    # 一个stride的filter展成一列
                    filt = w[m].reshape(-1, 1)
                    out[n, m, i, j] = data.dot(filt) + b[m]
    return out

def bn(x, gamma, beta):
    mean = np.mean(x, axis=0) # mean in dimN
    var = np.mean((x-mean)**2, axis=0)
    xhat = (x - mean) / var
    out = gamma * xhat + beta
    
    return out

def padding(x, pad):
    # x: h w c
    # feature_map: fo fh fw fi
    size_x = x.shape[2] #输入矩阵尺寸
    size = size_x + pad*2 # padding后尺寸
    if x.ndim == 4: # 每个元素是3维的，x的0维是mini-batch
        # 初始化同维全0矩阵
        padding = np.zeros((x.shape[0],x.shape[1], size,size))
        # 中间以x填充
        padding[:,:,pad: pad + size_x, pad: pad + size_x] = x
    return padding

def relu_scale(x, alpha, beta):
    relu_out = np.maximum(x, 0)
    relu_scale_out = np.zeros(relu_out.shape)
    N,C,H,W = relu_out.shape
    for n in range(N):
        for m in range(C):
            for i in range(H):
                for j in range(W):
                    relu_scale_out[n,m,i,j] = relu_out[n,m,i,j] * alpha + beta
    return relu_scale_out

if __name__ == "__main__":
    # 0. param
    #img = np.ones((1, 3, 224, 224)) # NCHW
    np.random.seed(1234)
    img = np.random.rand(2, 3, 224, 224) # NCHW
    N,C,H,W = img.shape
    for n in range(N):
        for m in range(C):
            for i in range(H):
                for j in range(W):
                    img[n, m, i, j] = m * img[n, m, i, j]
    w = np.ones((64, 3, 7, 7)) # Co Ci Fx Fy
    b = np.ones((64)) # C
    # 1. conv
    conv_out = conv_with_bias(img, w, b, 1, 1)
    print(conv_out, conv_out.shape)
    # 2. bn
    bn_out = bn(conv_out, 1, 0) # gamma:1 beta:0 BN结果不变化
    print(bn_out, bn_out.shape)
    # 3. relu+scale
    relu_scale_out = relu_scale(bn_out, 0.01, 0.0005)
    print(relu_scale_out, relu_scale_out.shape)