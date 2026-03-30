import torch 
import numpy as np
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2

def make_augmentation_dict(lib_name):
    if lib_name in ['album', 'albumentations']:
        return make_augmentation_dict_album2()
    return None

def check_module(module):
    s = str(module)
    if s == "None":
        return "None"
    if 'nrtk' in s:
        return corrupt_func_nrtk_reduced
    elif 'albumentations' in s:
        return corrupt_func_album_reduced
    elif 'imagecorruptions' in s:
        return corrupt_func_imagecorrupt_reduced
    elif 'augly' in s:
        return corrupt_func_augly_reduced
    else:
        raise ValueError('not valid library name')

# def corrupt_func_album_reduced_0(images_np, aug_func, aug_params):
#     rseed = 42
#     if 'random_seed' in aug_params:
#         rseed = aug_params['random_seed']
#         aug_params = {k:v for k,v in aug_params.items() if k != 'random_seed'}
#     transform = A.Compose([aug_func(**aug_params), ToTensorV2()])
#     transform.set_random_seed(rseed)
#     corrupted_imgs = np.array([ transform(image=img)['image'].permute(1, 2, 0).numpy() for img in images_np ])
#     return corrupted_imgs

def corrupt_func_album_reduced(images_np, aug_func, aug_params):
    rseed = aug_params.get("random_seed", 42)
    aug_params = {k:v for k,v in aug_params.items() if k != "random_seed"}

    corrupted = []
    for i, img in enumerate(images_np):
        transform = A.Compose([aug_func(**aug_params), ToTensorV2()])
        transform.set_random_seed(rseed + i)   # advance seed per image
        corrupted.append(
            transform(image=img)['image'].permute(1,2,0).numpy()
        )

    return np.array(corrupted)

def corrupt_func_augly_reduced(images_np, aug_func, aug_params): 
    corrupted_imgs = np.array([aug_np_wrapper(img, aug_func, **aug_params) for img in images_np])
    return corrupted_imgs

def corrupt_func_nrtk_reduced(images_np, aug_func, aug_params):
    perturber = aug_func(**aug_params)
    corrupted_imgs = np.array([ perturber(img)[0] for img in images_np ])
    return corrupted_imgs

def corrupt_func_imagecorrupt_reduced(images_np, aug_func, aug_params):
    corrupted_imgs = np.array([ corrupt(img.astype(np.uint8), corruption_name=aug_params['corrname'], severity=aug_params['severity']) for img in images_np ])
    return corrupted_imgs

def make_augmentation_dict_album():
    augmentations_album, aug_names_album = get_album_augmentations_list()
    augmentations_album2 = []
    for a in augmentations_album:
        td = {"None": "None"}
        aug_func = a[0]
        for severity in range(1,6):
            aug_params = {k: (v(severity) if callable(v) else v) for k, v in a[1].items()}
            td[f"severity_{severity}"] = aug_params
        augmentations_album2.append((aug_func, td))
    d = {}
    for aug_tuple, aug_name in zip(augmentations_album2, aug_names_album):
        # print(aug_tuple[1], aug_tuple[0])
        new_aug = Augmentation(aug_name, aug_tuple[1], aug_tuple[0])
        d[aug_name] = new_aug 
    return d 

def get_album_augmentations_list():
    """
    Get the augmentation list and string for albumentations.

    Args:
        None.

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
    augmentations_album = [
        (A.RandomRain, {'slant_range': (0, 30), 
                        'drop_length': lambda s: 2*s, 
                        'drop_width': lambda s: s, 
                        'drop_color': (200, 200, 200), 
                        'blur_value': lambda s: 3 + s, 
                        'brightness_coefficient': lambda s: 1 - 0.1*s, 
                        'rain_type': 'drizzle', 
                        'p': 1.0}),
        (A.RandomSnow, {'snow_point_range': lambda s: (0.3+0.1 * s, 0.5 + 0.1* s), 
                        'brightness_coeff': lambda s: 1.0+0.35*s, 
                        'p': 1.0}),
        (A.ColorJitter, {'brightness': lambda s: (0.85+0.4*s, 1+0.4*s),
                        # 'contrast': (1,1),
                        # 'saturation': (1,1),
                        # 'hue': (0,0),
                        'p':1}),
        (A.GaussianBlur, {'blur_limit': lambda s: (9 + 6*s, 13 + 6*s), 
                        'sigma_limit':lambda s: (0.25 + 0.25*s, 1.0 + 0.25*s),
                        'p': 1.0}),
        # (A.GlassBlur, {'sigma': lambda s: s*2, 'p': 1.0}),
        # (A.Defocus, {'radius': lambda s: (3*s,3*s+1), 'alias_blur': lambda s: 2*s, 'p': 1.0}),
        # (A.MotionBlur, {'blur_limit': lambda s: (5+8*s, 7+12*s), 'p': 1.0}),
        # (A.ZoomBlur, {'max_factor': lambda s: 1 + 0.25 * s, 'p': 1.0}),
        (A.Affine, {'shear': lambda s: (-10*s, 10*s), 'p': 1.0}),
        (A.Affine, {'translate_percent': lambda s: (-0.1*s, 0.1*s), 'p': 1.0}),
        (A.Affine, {'scale': lambda s: (1/(1+0.75*s) , (1+0.75*s) ), 'p': 1.0}),
        (A.ColorJitter, {'contrast': lambda s:  (1+0.75*s,1+0.75*s),'p':1}),
        (A.GaussNoise, {'std_range':  lambda s: (0,0.05*s), 'mean_range': lambda s:  (0,0.05*s),'p':1}),
        (A.Perspective, {'scale':  lambda s: 0.3*s,'p':1}),
        # (A.Erasing, {'scale': lambda s: (0.10*s, 0.10*s), 'ratio': (0.5, 2),'p':1}),
        (A.CoarseDropout, {'hole_height_range': lambda s: (0.1*s, 0.1*s),
                           'hole_width_range': lambda s: (0.1*s, 0.1*s), 
                           'p': 1.0}),
        (A.ImageCompression, {'quality_range': lambda s: (100-19*s, 100-19*s)}),
        (A.Rotate, {'limit': lambda s: 25*s,'p':1}),
    ]
    aug_names_album = [
        "Random Rain", 
        "Random Snow", 
        "Brightness", 
        "Gaussian Blur", 
        # "Glass Blur", 
        # "Defocus Blur", 
        # "Motion Blur", 
        # "Zoom Blur",
        "Shear", 
        "Translate",
        "Scale",
        "Contrast",
        "Gaussian Noise",
        "Perspective",
        "Erasing",
        "Compression",
        "Rotation"
    ]
    return augmentations_album, aug_names_album

def get_augmentation_dict_album_header():
    rng = np.random.default_rng(42)
    d = {
        "None":
        { 
            "None": ("None", "None")
        },
        "Erasing":
        {
            f"scale_{0.1*x:.2f}": (A.CoarseDropout, {
                'num_holes_range': (1,1),
                'hole_height_range': (0.1*x, 0.1*x),
                'hole_width_range': (0.1*x, 0.1*x),
                'p':1
            }) for x in range(1,6+1)
        }, 
        "Rain":
        {
            f"rain_type_{x}": (A.RandomRain, {'slant_range': (-15, 15), 
                        'blur_value': 5, 
                        'rain_type': x, 
                        'p': 1.0}) for x in ['drizzle', 'heavy', 'torrential']
        },
        "Rotate":
        {
            f"rotate_{10*x}": (A.Rotate, {'limit': 10*x,'p':1}) for x in range(1,9+1)
        },
        "GaussianNoise":
        {
            f"std_{0.05*x:.2f}": (A.GaussNoise, {'std_range':  (0.05*x, 0.05*x), 'p':1}) for x in range(1,8+1)
        },
        "BrightnessUp":
        {
            f"bright_{1+0.25*x:.2f}": (A.ColorJitter, {'brightness': (1+0.25*x, 1+0.25*x), 'p':1}) for x in range(1,8+1)
        },
        "BrightnessDown":
        {
            f"bright_{1-0.1*x:.2f}": (A.ColorJitter, {'brightness': (1-0.1*x, 1-0.1*x), 'p':1}) for x in range(1,8+1)
        },
        "GaussianBlur":
        {
            f"sigma_{0.5+1*x:.2f}": (A.GaussianBlur, {'sigma_limit': (0.5+1*x, 0.5+1*x), 'p':1}) for x in range(1,6+1)
        },
        "Contrast":
        {
            f"contrast_{1+1*x:.2f}": (A.ColorJitter, {'contrast': (1+1*x,1+1*x),'p':1}) for x in range(1,8+1)
        },
        "ScaleUp":
        {
            f"scale_{1+0.5*x:.2f}": (A.Affine, {'scale': ((1+0.5*x) , (1+0.5*x) ), 'p': 1.0}) for x in range(1,8+1)
        },
        "ScaleDown":
        {
            f"scale_{1/(1+0.5*x):.2f}": (A.Affine, {'scale': (1/(1+0.5*x) , 1/(1+0.5*x) ), 'p': 1.0}) for x in range(1,8+1)
        },
        "Translate":
        {
            f"translate%_{0.15*x:.2f}": (A.Affine, {'translate_percent': (-0.15*x, 0.15*x) , 'p': 1.0}) for x in range(1,6+1)
        },
        "Shear":
        {
            f"shear_{8*x:.2f}": (A.Affine, {'shear': (-8*x , 8*x) , 'p': 1.0}) for x in range(1,8+1)
        },
        "Perspective":
        {
            f"perspective_{0.5*x:.2f}": (A.Perspective, {'scale': 0.5*x,'p':1}) for x in range(1,8+1)
        },       
        "Compression":
        {
            f"quality_range_{85-x*10:.2f}": (A.ImageCompression, {'quality_range': (85-x*10, 85-x*10),'p':1}) for x in range(1,8+1)
        },
    }

    return d

DETERMINISTIC = {"None", "BrightnessUp", "BrightnessDown", "GaussianBlur", "ScaleUp", "ScaleDown", "Compression"}

import ast

def parse_parameters(s):
    s = s.strip()

    # Case 1: tuples present
    if "(" in s:
        try:
            return ast.literal_eval(f"[{s}]")
        except (ValueError, SyntaxError):
            pass

    # Case 2: try numeric list
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        # Case 3: fallback to strings
        return [x.strip() for x in s.split(",")]

def custom_parameter_change(aug_dict, aug_name, param_name, parameters_in_string):
    parameters_in_num = parse_parameters(parameters_in_string)
    param_dict = {}
    if aug_name == "Erasing" and param_name == "scale":
        for tup in parameters_in_num:
            temp = {
                'num_holes_range': (1,1),
                'hole_height_range': tup,
                'hole_width_range': tup,
                'p':1.0
            }  
            param_dict[f"{param_name}_{tup[0]}"] = temp
    else: 
        for param in parameters_in_num:
            temp = {
                param_name: param,
                'p':1.0
            }  
            val = param if type(param) != tuple else param[0]
            param_dict[f"{param_name}_{val}"] = temp

    for key in aug_dict:
        if key == aug_name:
            aug_class = aug_dict[key]
            aug_func = aug_class.aug_func ; rand_seed = aug_class.random_seed
            new_aug_class = Augmentation(aug_name, param_dict, aug_func)
            if rand_seed is not None: 
                aug_class.set_seed(rand_seed)
            aug_dict[key] = new_aug_class

    return aug_dict

def make_augmentation_dict_album2():
    old_d = get_augmentation_dict_album_header()
    d = {}
    for k,v in old_d.items():
        param_dict = {k1:v1[1] for k1,v1 in v.items()}
        for k1,v1 in param_dict.items():
            if k1 != "None":
                print(k1, v1)
                v1['random_seed'] = 42
        aug_func = v[list(v.keys())[0]][0]
        print(aug_func , "aug_func")
        # print(aug_tuple[1], aug_tuple[0])
        new_aug = Augmentation(k, param_dict, aug_func)
        d[k] = new_aug 
    return d 

def make_augmentation_dict_imagecorrupt():
    augmentations = [
        (corrupt, {'corrname': 'gaussian_noise'}),
        (corrupt, {'corrname': 'fog'}),
        (corrupt, {'corrname': 'brightness'}),
        (corrupt, {'corrname': 'zoom_blur'}),
    ]
    aug_str = ['gaussian_noise', 'fog', 'brightness', 'zoom_blur']
    augmentations2 = []
    for a in augmentations:
        td = {}
        aug_func = a[0]
        for severity in range(6):
            aug_params = a[1]
            td[f"severity_{severity}"] = aug_params
        augmentations_album2.append((aug_func, td))
    d = {}
    for aug_tuple, aug_name in zip(augmentations_album2, aug_names_album):
        # print(aug_tuple[1], aug_tuple[0])
        new_aug = Augmentation(aug_name, aug_tuple[1], aug_tuple[0])
        d[aug_name] = new_aug 
    return d

class Augmentation:
    def __init__(self, name, param_dict, aug_func):
        self.name = name 
        self.severities = list(param_dict.keys())
        if len(set(self.severities)) == 1 and len(self.severities) != 1:
            self.severities = [f"{x}_{i}" for i,x in enumerate(self.severities)]
            self.param_dict = {f"{x}_{i}":v for i,(x,v) in enumerate(param_dict.items())}
        else:
            self.param_dict = param_dict

        self.corrupt_func = check_module(aug_func)
        self.aug_func = aug_func
        self.random_seed = None
        self.deterministic = True if name in DETERMINISTIC else False 

    # def set_seed_0(self, x):
    #     self.random_seed =  x
    #     if self.name != "None":
    #         for k,v in self.param_dict.items():
    #             v['random_seed'] = x

    def set_seed(self, x):
        self.random_seed = x
        if self.name != "None":
            for k in self.param_dict:
                self.param_dict[k] = {
                    **self.param_dict[k],
                    "random_seed": x
                }

    def determine_severity(self, severity_idx):
        if type(severity_idx) == int:
            # print(f"Index is integer value {severity_idx}")
            all_severities = ["None"] + self.severities 
            severity = all_severities[severity_idx]
            # print(f"Which corresponds to value {severity}")
        else:
            # print(f"Severity is directly referenced as {severity_idx}")
            severity = severity_idx
        return severity

    def corr_func_one_img(self, img, severity_idx):
        if type(severity_idx) == int:
            severity = self.severities[severity_idx]
        else:
            severity = severity_idx
        if self.name not in ["None", None]:
            corrupted_image = self.corrupt_func([img], self.aug_func, self.param_dict[severity])
        else:
            corrupted_image = [img]

        return corrupted_image[0]

    def corr_func_arr(self, arr, severity_idx):
        if type(severity_idx) == int:
            print(self.severities, "SEV", severity_idx)
            severity = self.severities[severity_idx]
        else:
            severity = severity_idx
        if self.name not in ["None", None]:
            corrupted_images = self.corrupt_func(arr, self.aug_func, self.param_dict[severity])
        else:
            corrupted_images = arr

        return corrupted_images
    
    def corr_func_dataloader(self, testloader, severity_idx):
        severity = self.determine_severity(severity_idx)

        if self.name in ["None", None] or severity == None:
            return testloader

        dataset = CorruptedDataset(
            testloader.dataset,
            self.corr_func_arr,
            severity_idx,
        )

        pin_memory = torch.cuda.is_available()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=testloader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, corr_func, severity_idx):
        self.dataset = dataset
        self.corr_func = corr_func
        self.severity_idx = severity_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image_np = (
            image.mul(255)
            .byte()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Add batch dimension
        image_np = image_np[None, ...]

        corrupted = self.corr_func(image_np, self.severity_idx)

        # Remove batch dimension
        corrupted = corrupted[0]

        # Handle (H, W, C, 1)
        if corrupted.ndim == 4 and corrupted.shape[-1] == 1:
            corrupted = corrupted[..., 0]

        corrupted = torch.from_numpy(
            corrupted.transpose(2, 0, 1)
        ).float().div_(255.0)

        return corrupted, label

