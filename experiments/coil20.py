from .default import Experiment

coil20 = Experiment(
    arch='alexnet',
    hidden_dim=1024,
    verbose=True,
    log_dir='./logs/mytest',
    device='cuda',
    extra_record=True,
    opt='adam',
    epochs=100,
    lr=1e-4,
    batch_size=36,
    cluster_hidden_dim=512,
    ds_name='coil-20',
    img_size=128,
    input_channels=[1, 1, 1],
    views=3,
    clustering_loss_type='ddc',
    num_cluster=20,
    fusion_act='relu',
    use_bn=True,
    contrastive_type='simclr',
    projection_layers=2,
    projection_dim=512,
    contrastive_lambda=0.01,
    temperature=0.1,
    seed=312,
   
)
