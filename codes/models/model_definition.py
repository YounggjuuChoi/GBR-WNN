import models.archs.GBRWNN_arch as GBRWNN_arch

# Model Definition
def define_model(opt):
    opt_net = opt['network']
    which_model = opt_net['which_model']

    ### VSR ------------------------------------------------------------------------------------------------------------
    # GBR-WNN
    if which_model == 'GBRWNN':
        net = GBRWNN_arch.GBRWNN(
            nf=opt_net['nf'],
            nframes=opt_net['nframes'],
            RBs=opt_net['RBs'],
            scale=opt_net['scale'])
    ### ----------------------------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError('Model [{:s}] not recognized'.format(which_model))

    return net