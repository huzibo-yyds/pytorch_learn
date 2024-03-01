from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("../log/8_logs")


# 注意同一tag下的数据全在一个图中
for i in range(100):
    writer.add_scalar("y=x^2",i*i*i,i)

writer.close()

"""tensorboard
    tensorBoard是一个由TensorFlow提供的可视化工具
    
    本例子是在tensorboard中画一个坐标图
    
    如何打开tensorboard，在conda命令行中输入以下命令：
        tensorboard --logdir=log
"""