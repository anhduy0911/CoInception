import matplotlib.pyplot as plt
import numpy as np
import torch
from modules.coinception import CoInception
import seaborn as sns
import os

def visualize_trajectory():
    T = 20
    L = 1000
    N = 1
    np.random.seed(2)
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float32')

    seg_len = L // 3
    data[:, :seg_len] = data[:, :seg_len] + np.random.randn(N, seg_len) * 0.05
    data[:, seg_len:seg_len*2] = data[:, seg_len:seg_len*2]*3 + np.random.randn(N, seg_len) * 0.1
    data[:, seg_len*2:] = data[:, seg_len*2:] + np.random.randn(N, L - 2*seg_len) * 0.15
    
    config = dict(
        batch_size=8,
        lr=0.001,
        output_dims=10,
        max_train_length=3000
    )

    data = data[:,:,np.newaxis]
    model = CoInception(
        input_len=data.shape[1],
        input_dims=data.shape[-1],
        device=torch.device('cuda'),
        **config
    )
    if not os.path.exists('training/vis/model.pkl'):
        # train model
        loss_log = model.fit(
            data,
            n_epochs=None,
            n_iters=None,
            verbose=True
        )
        run_dir = 'training/vis'
        os.makedirs(run_dir, exist_ok=True)
        model.save(f'{run_dir}/model.pkl')
    else:
        model.load('training/vis/model.pkl')

    rep = model.encode(data)

    # visualize
    fig, axs = plt.subplots(2, 1, figsize=(7, 3), sharex=True, gridspec_kw={'height_ratios': [1.5, 2]})
    axs[0].plot(np.arange(L), data[0,:,0].T)
    axs[0].set_ylabel('Amp',fontsize=15)
    axs[0].set_yticks(np.arange(-2, 3, 2))
    axs[0].set_yticklabels(np.arange(-2, 3, 2), fontsize=10)

    sns.heatmap(rep.squeeze().T, cbar=False, ax=axs[1], xticklabels=False)
    axs[1].set_ylabel('Dim', fontsize=15)
    axs[1].set_xticks(np.arange(0, L, 100))
    axs[1].set_xticklabels(np.arange(0, L, 100), fontsize=11)

    plt.xlabel('Time Step', fontsize=15)
    fig.tight_layout()
    plt.savefig('trajectory.pdf', format='pdf', dpi=300, bbox_inches='tight')


def vis_low_pass_filter():
    import pywt

    signal = np.random.randn(1, 1, 1000) * 10
    signal[:, :, :100] = 0
    signal[:, :, 900:] = 0
    def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
        thresh = thresh*np.nanmax(signal)
        coeff = pywt.wavedec(signal, wavelet, mode="per" )
        coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
        reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
        return reconstructed_signal

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(signal[0,0], color="b", alpha=0.5, label='original signal')
    rec = lowpassfilter(signal, 0.2)
    ax.plot(rec[0,0], 'k', label='DWT smoothing}', linewidth=2)
    ax.legend()
    ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
    ax.set_ylabel('Signal Amplitude', fontsize=16)
    ax.set_xlabel('Sample No', fontsize=16)
    plt.savefig('test.png')

if __name__ == '__main__':
    visualize_trajectory()
    # vis_low_pass_filter()