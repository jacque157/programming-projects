from DatasetLoader import *
from Transforms import *
from Tester import RelativePoseTester


CHANNELS = 3
CLASSES = 24
SEGMENTATION = True
SUBSET = 'ALL'
DATASET_PATH = os.path.join('..', 'codes', 'Dataset')
MINMAX_PATH = os.path.join(DATASET_PATH, 'min_max.npy')
MEAN_PATH = os.path.join(DATASET_PATH, 'mean.npy')

if __name__ == '__main__':
    device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Visualising on {device}')
    min_, max_ = np.load(MINMAX_PATH) #find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(MEAN_PATH)

    transform = transforms.Compose([ZeroCenter(mean),
                                    Rescale(min_, max_, -1, 1),
                                    RootAlignedPose()])
    transform = None

    ACCAD_dataset = Poses3D(DATASET_PATH, 
                            'ACCAD', 
                            subset=SUBSET, 
                            transform=transform,
                            protocol=(0, 1), 
                            include_segmentation=SEGMENTATION,
                            shuffle_seed=None)
    
    CMU_dataset = Poses3D(DATASET_PATH,
                        'CMU',
                        subset=SUBSET,
                        transform=transform,
                        protocol=(0, 1), 
                        include_segmentation=SEGMENTATION,
                        shuffle_seed=None)
    
    EKUT_dataset = Poses3D(DATASET_PATH,
                            'EKUT',
                            subset=SUBSET,
                            transform=transform,
                            protocol=(0, 1), 
                            include_segmentation=SEGMENTATION,
                            shuffle_seed=None)
    
    Eyes_Japan_dataset = Poses3D(DATASET_PATH,
                                'Eyes_Japan',
                                subset=SUBSET,
                                transform=transform,
                                protocol=(0, 1),
                                include_segmentation=SEGMENTATION,
                                shuffle_seed=None)
    
    dataloader = DatasetLoader([CMU_dataset, 
                                EKUT_dataset, 
                                ACCAD_dataset, 
                                Eyes_Japan_dataset], 
                                batch_size=32, 
                                transforms=None, #ToDevice(device),
                                shuffle=False)
    
    for sample in dataloader:
        skeletons = sample['key_points']
        images = sample['sequences']
        segmentations = sample['segmentations']

        for skeleton, image, segmentation in zip(skeletons, images, segmentations):
            ax = plot_body(image)
            plot_skeleton(skeleton, ax)
            plt.show()

            plot_annotation(image, segmentation)
            plt.show()

            f, axs = plt.subplots(3, 1)
            min_ = np.min(image)
            max_ = np.max(image)
            for i in range(3):
                img_i = image[:, :, i]
                axs[i].imshow(img_i, cmap='gray', vmin=min_, vmax=max_)
                axs[i].axis('off')

            plt.axis('off')
            plt.show()
