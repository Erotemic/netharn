                # iaa.Flipud(p=.5),
                # iaa.Affine(
                #     # scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                #     # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                #     rotate=(-3.6, 3.6),
                #     # rotate=(-15, 15),
                #     # shear=(-7, 7),
                #     # order=[0, 1, 3],
                #     order=1,
                #     # cval=(0, 255),
                #     cval=127,
                #     mode=ia.ALL,
                #     backend='cv2',
                # ),

                # iaa.AddToHueAndSaturation((-20, 20)),
                # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                # iaa.AddToHueAndSaturation((-15, 15)),
                # iaa.ContrastNormalization((0.75, 1.5))
                # iaa.ContrastNormalization((0.75, 1.5), per_channel=0.5),

    # def __len__(self):
    #     # hack
    #     if 'train' in self.split:
    #         return 100
    #     else:
    #         return super().__len__()

    # def initialize(harn):
    #     super().initialize()
    #     harn.datasets['train']._augmenter = harn.datasets['train'].augmenter
    #     if harn.epoch <= 0:
    #         # disable augmenter for the first epoch
    #         harn.datasets['train'].augmenter = None
