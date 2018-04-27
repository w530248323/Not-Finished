import torch
from torch import nn
from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet, pseudonet


def generate_model(opt):

    global model
    assert opt.model in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'pseudonet']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)

    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]
        from models.wide_resnet import get_fine_tuning_parameters
        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)

    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters
        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)

    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]
        from models.pre_act_resnet import get_fine_tuning_parameters
        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)

    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]
        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)

    elif opt.model == 'pseudonet':
        assert opt.model_depth in [15, 31, 63, 131, 199]
        if opt.model_depth == 15:
            model = pseudonet.P3D63(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        if opt.model_depth == 31:
            model = pseudonet.P3D63(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        if opt.model_depth == 63:
            model = pseudonet.P3D63(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 131:
            model = pseudonet.P3D131(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 199:
            model = pseudonet.P3D199(
                num_classes=opt.n_classes,
                sample_height=opt.sample_height,
                sample_width=opt.sample_width,
                sample_duration=opt.sample_duration)

    # gpus = [int(i) for i in opt.gpus.split(',')]
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model = model.cuda()

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        assert opt.arch == pretrain['arch']

        model.load_state_dict(pretrain['state_dict'])

        if opt.model == 'densenet':
            model.module.classifier = nn.Linear(
                model.module.classifier.in_features, opt.n_finetune_classes)
            model.module.classifier = model.module.classifier.cuda()
        else:
            model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model
