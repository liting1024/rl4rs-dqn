import torch
import matplotlib.pyplot as plt

def read_train_data():
    train_path = 'dataset/train.pt'
    train = torch.load(train_path)

def read_best_pt(pt_path):
    model = torch.load(pt_path)
    rewards = model['episode_rewards'][:-1]
    return rewards

def draw_multi_line(xs, ys, label):
    for x, y, l in zip(xs, ys, label):
        plt.plot(x, y, label=l)

    plt.title('Train')
    plt.xlabel('episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig('rw.png', transparent=True, dpi=300)
    plt.show()

def draw_episode_rewards():
    output = [
        # '../output_bg', 'base_output',  # 这里的base_output是在本地第一次训得极少
        'base_model',
        # 'bilstm7+linear7',
        # 'lstm7',
        'bilstm64+linear7'
    ]
    # output = 'output'  # debug
    pt_path = [f'{output}/best_checkpoint.pt' for output in output]

    ys = [read_best_pt(path) for path in pt_path]
    xs = [list(range(len(y))) for y in ys]

    bilstm7 = read_best_pt('bilstm7+linear7/best_checkpoint.pt')
    ys.append(bilstm7)
    xs.append(list(range(268121, 268121 + len(bilstm7))))
    lstm7 = read_best_pt('lstm7/best_checkpoint.pt')
    ys.append(lstm7)
    xs.append(list(range(289130, 289130+len(lstm7))))
    # plt.plot(xs, ys, label=['baseline', 'BiLSTM7', 'LSTM7', 'BiLSTM64'])
    # plt.show()
    draw_multi_line(xs, ys, ['baseline', 'BiLSTM64', 'BiLSTM7','LSTM7'])


if __name__ == '__main__':
    # read_train_data()
    # read_best_pt()
    draw_episode_rewards()
    pass