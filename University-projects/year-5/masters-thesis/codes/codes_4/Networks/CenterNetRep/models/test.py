from networks import resnet_dcn


net = resnet_dcn.get_pose_net(18, {'hm' : 24, 'dep' : 1, 'reg' : 2}, 64)
a = torch.rand(32, 3, 256, 128)
b = net(a)
