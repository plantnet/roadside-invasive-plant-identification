#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides code to access datasets (images and tensors folder)

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

import os
import re
from PIL import Image as PIL_Image

import torch
from torch.nn.functional import one_hot
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from lib_utils_files import load_torch_tensor


def create_liste_augmentations_from_model_infos(model_infos, create_augment_specs):

    liste_augmentations = []

    # Cas particulier pour le tiling:
    # Pas de data augmentation, car on prend en entrée des deep features
    if (("identity" in create_augment_specs.keys()) and 
        (create_augment_specs["identity"] == True)):
        if (model_infos["model_name"] == "tiling"):
            liste_augmentations.append({"name" : "identity"})


    # Cas général ou ce sont des images en entrée (vamis, ou inférence simple)
    if (("totensor" in create_augment_specs.keys()) and 
        (create_augment_specs["totensor"] == True)):
        liste_augmentations.append(
        {
            "name" : "totensor"
        })

    if (("resize" in create_augment_specs.keys()) and 
        (create_augment_specs["resize"] == True)):
        height = model_infos["model_input_size"]
        width  = model_infos["model_input_size"]
        if ("model_input_size_xy_vamis" in model_infos.keys()):
            if (model_infos["model_input_size_xy_vamis"] != None):
                height = model_infos["model_input_size_xy_vamis"][0]
                width  = model_infos["model_input_size_xy_vamis"][1]

        liste_augmentations.append(
        {
            "name"       : "resize",
            "img_height" : height, 
            "img_width"  : width
        })

    if (("normalize" in create_augment_specs.keys()) and 
        (create_augment_specs["normalize"] == True)):
        liste_augmentations.append(
        {
            "name"       : "normalize",
            "mean"       : model_infos["img_mean"],
            "std"        : model_infos["img_std"],
        })

    return liste_augmentations


# Construction des data augmentations
def get_augmentations(liste_augmentations):

    print("Appel à get_augmentations()")
    # Afficher les versions des principales librairies utilisées
    print("Version de pytorch:", torch.__version__)
    print("Version de torchvision:", torchvision.__version__)

    transform_stack = []
    print("get_augmentations: liste_augmentations = ")
    print(liste_augmentations)

    for data_aug in liste_augmentations:
        if ("name" in data_aug.keys()):
            transform = None
            match (data_aug["name"]):

                case "identity":
                    transform = torch.nn.Identity()

                case "totensor":
                    transform = transforms.ToTensor()

                case "resize":
                    img_height = data_aug["img_height"]
                    img_width  = data_aug["img_width"]
                    transform  = transforms.Resize((img_height, img_width))

                case "normalize":
                    mean = data_aug["mean"]
                    std  = data_aug["std"]
                    transform = transforms.Normalize(mean, std)

                case "randomverticalflip":
                    transform = transforms.RandomVerticalFlip()

                case "randomhorizontalflip":
                    transform = transforms.RandomHorizontalFlip()

                case "colorjitter":
                    randombrightness = data_aug["randombrightness"]
                    randomcontrast = data_aug["randomcontrast"]
                    randomsaturation = data_aug["randomsaturation"]
                    transform = transforms.ColorJitter(
                        brightness = randombrightness,
                        contrast = randomcontrast,
                        saturation = randomsaturation)

                case "centercrop":
                    img_height = data_aug["img_height"]
                    img_width = data_aug["img_width"]
                    transform = transforms.CenterCrop((img_height, img_width))

                case "randomrotation":
                    randomrotation_factor = data_aug["randomrotation_factor"]
                    transform = transforms.RandomRotation(
                        degrees = 360.0 * randomrotation_factor,
                        interpolation = InterpolationMode.BILINEAR)

                case _:
                    print("get_augmentations > Error: Unrecognized data augmentation:", data_aug["name"])
                    exit(-1)

            transform_stack.append(transform)

    augmentations_transforms = transforms.Compose(transform_stack)
    #print("debug3:", augmentations_transforms)
    #exit()
    return augmentations_transforms



def get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):

    return torch.utils.data.DataLoader(
                dataset = dataset,
                batch_size = batch_size,
                shuffle = shuffle,
                num_workers = num_workers,
                pin_memory = pin_memory)



class simple_dataset(Dataset):

    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform

        self.all_imgs = sorted(list(os.listdir(dataset_folder)), key = str)
        self.total_imgs = len(self.all_imgs)
        
    def __getitem__(self, index):
        img_loc = os.path.join(self.dataset_folder, self.all_imgs[index])
        image = PIL_Image.open(img_loc)
        #image = PIL_Image.open(img_loc).convert("RGB")
        x = self.transform(image)
        return x, img_loc
    
    def __len__(self):
        return self.total_imgs
        

class Dataset_generic_imgs_labels_paths(Dataset):

    def __init__(self,
            transform,
            classes_list,
            caching_imgs                    = None,
            caching_labels                  = None,
            dataset_folder                  = "",      # For path reconstruction
            caching_relative_img_path       = None,
            bool_return_data_images         = True,
            bool_true_tensors_false_images  = False,
            bool_return_relative_img_names  = False,
            bool_return_short_img_names     = False,
            device                          = None):

        self.transform                 = transform
        self.classes_list              = classes_list    # Pour conserver cette liste, et donner du sens aux labels
        self.caching_imgs              = caching_imgs
        self.caching_labels            = caching_labels

        self.dataset_folder            = dataset_folder
        self.caching_relative_img_path = caching_relative_img_path

        self.bool_return_data_images        = bool_return_data_images
        self.bool_true_tensors_false_images = bool_true_tensors_false_images
        self.bool_return_short_img_names    = bool_return_short_img_names
        self.bool_return_relative_img_names = bool_return_relative_img_names
        self.device                         = device
    
        if ((caching_imgs == None) and (caching_relative_img_path == None)):
            print("Dataset_generic_imgs_labels_paths: Error : At least one of these must be provided: caching_imgs or caching_relative_img_path")
            exit(-1)

        if (bool_return_short_img_names and (caching_relative_img_path == None)):
            print("Dataset_generic_imgs_labels_paths: Error: Can't return short_img_name without caching_relative_img_path.")


        n_images = -1
        if (caching_imgs != None):
            if (bool_true_tensors_false_images):
                n_images = caching_imgs.shape[0]
            else:
                n_images = len(caching_imgs)

        if (caching_relative_img_path != None):
            n_images_from_caching_relative_img_path = len(caching_relative_img_path)

            if (n_images == -1):
                n_images = n_images_from_caching_relative_img_path
            else:
                if (n_images != n_images_from_caching_relative_img_path):
                    print("Dataset_generic_imgs_labels_paths: Error > data must have same size (1) - caching_relative_img_path")
                    exit(-1)

        if (caching_labels != None):
            n_images_from_caching_labels = caching_labels.shape[0]
            if (n_images != n_images_from_caching_labels):
                print("Dataset_generic_imgs_labels_paths: Error > data must have same size (2) - caching_labels")
                exit(-1)

        self.n_images   = n_images
        
    def __getitem__(self, index):
        #print("__getitem__:", index)

        # Tuple de retour: Ne doit contenir que des champs 'collatables'
        tuple_return = ()

        # Obtenir les données de l'image ou du tenseur
        if (self.bool_return_data_images):
            if (self.caching_imgs != None):
                img_data             = self.caching_imgs[index]
            else:
                # Chargement de l'image ou du tenseur
                img_path             = os.path.join(self.dataset_folder, self.caching_relative_img_path[index])
                #print("debug v256:", img_path)
                if (self.bool_true_tensors_false_images):
                    img_data = load_torch_tensor(img_path, device = self.device)
                else:
                    img_data = PIL_Image.open(img_path)

            transformed_img_data = self.transform(img_data)

            tuple_return         = tuple_return + (transformed_img_data, )
            

        if (self.caching_labels != None):
            label        = self.caching_labels[index]
            tuple_return = tuple_return + (label, ) 

        # Obtenir le nom du fichier
        if (self.bool_return_relative_img_names):
            #print("debug v257:", self.caching_relative_img_path[index])
            tuple_return         = tuple_return + (self.caching_relative_img_path[index], ) 
        
        
        if (self.bool_return_short_img_names):
            img_path             = self.caching_relative_img_path[index]
            short_img_path       = img_path.split("/")[-1]
            tuple_return         = tuple_return + (short_img_path, ) 
        
        #print("debug v258:", tuple_return)
        #exit()
        
        return tuple_return

    
    def __len__(self):
        return self.n_images


# Un dossier contenant (ou pas) des sous-dossiers (arborescence arbitraire)
# Les images peuvent être disséminées dans les sous-dossiers
# Le Dataset construit va servir tous les fichiers images/tenseurs (tous les fichiers qui ne sont pas des sous-dossiers)
def datasets_generic_ArbitraryFolder(
            dataset_folder,
            bool_return_data_images         = True,
            bool_true_tensors_false_images  = False,
            bool_caching_images             = False,
            bool_return_relative_img_names  = False,
            bool_return_short_img_names     = False,
            transform                       = torch.nn.Identity(),
            device                          = None):

    if (False):      
        print("Debug 260: Appel à datasets_generic_ArbitraryFolder")
        print(dataset_folder,
                bool_return_data_images,
                bool_true_tensors_false_images,
                bool_caching_images,
                bool_return_relative_img_names,
                bool_return_short_img_names,
                transform,
                device)
        exit()

    classes_list = []   # Pas de notion de 'classe' ici

    #caching_relative_img_path = sorted(list(os.listdir(dataset_folder)), key = str)
    #caching_relative_img_path = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_folder) for file in files]
    caching_relative_img_path = [os.path.join(os.path.relpath(root, dataset_folder), file) for root, dirs, files in os.walk(dataset_folder) for file in files]

    caching_relative_img_path = list(sorted(caching_relative_img_path, key = str))

    if (False):    
        print(dataset_folder)
        print(caching_relative_img_path)
        print(len(caching_relative_img_path))
        exit()
    
    n_images = len(caching_relative_img_path)
    if (n_images == 0):
        print("Erreur: Pas d'images/de tenseurs dans le dossier: " + dataset_folder)
        exit(-1)

    caching_imgs = None
    if (bool_caching_images):
        f_path_first_tensor = caching_relative_img_path[0]
        first_tensor = load_torch_tensor(f_path_first_tensor, device = device)
        n_views, n_features = first_tensor.shape
        print("n_views:", n_views, ", n_features:", n_features)
        print("Empreinte mémoire - caching des deep features (Go): ", (4 * n_images * n_views * n_features / 1024 / 1024 / 1024))

        caching_imgs = torch.zeros(n_images, n_views, n_features, device = device)
        image_index = 0
        for relative_class_img_path in caching_relative_img_path:
            full_class_img_path = os.path.join(dataset_folder, relative_class_img_path)
            caching_imgs[image_index] = load_torch_tensor(full_class_img_path, device = device)
            image_index = image_index + 1
        
    return Dataset_generic_imgs_labels_paths(
        transform                       = transform,
        classes_list                    = classes_list,
        caching_imgs                    = caching_imgs,
        caching_labels                  = None,
        dataset_folder                  = dataset_folder,
        caching_relative_img_path       = caching_relative_img_path,
        bool_return_data_images         = bool_return_data_images,
        bool_true_tensors_false_images  = bool_true_tensors_false_images,
        bool_return_relative_img_names  = bool_return_relative_img_names,
        bool_return_short_img_names     = bool_return_short_img_names,
        device                          = device)




# Un seul dossier contenant les images ou tenseurs:
# Utilisé pour le multilabel, pour lequel l'arborescence avec "1 classe = 1 dossier" n'est pas possible
def datasets_generic_SingleFolder(
            dataset_folder,
            bool_return_data_images         = True,
            bool_true_tensors_false_images  = False,
            bool_caching_images             = False,
            bool_caching_labels             = False,
            bool_return_relative_img_names  = False,
            bool_return_short_img_names     = False,
            transform                       = torch.nn.Identity(),
            device                          = None):
            
    dummy_class_name = "dummy_class"
    dummy_class_index = 0

    classes_list = [dummy_class_name]

    caching_relative_img_path = sorted(list(os.listdir(dataset_folder)), key = str)
    n_images = len(caching_relative_img_path)
    if (n_images == 0):
        print("Erreur: Pas d'images/de tenseurs dans le dossier: " + dataset_folder)
        exit(-1)

    caching_imgs = None
    if (bool_caching_images):
        f_path_first_tensor = caching_relative_img_path[0]
        first_tensor = load_torch_tensor(f_path_first_tensor, device = device)
        n_views, n_features = first_tensor.shape
        print("n_views:", n_views, ", n_features:", n_features)
        print("Empreinte mémoire - caching des deep features (Go): ", (4 * n_images * n_views * n_features / 1024 / 1024 / 1024))

        caching_imgs = torch.zeros(n_images, n_views, n_features, device = device)
        image_index = 0
        for relative_class_img_path in caching_relative_img_path:
            full_class_img_path = os.path.join(dataset_folder, relative_class_img_path)
            caching_imgs[image_index] = load_torch_tensor(full_class_img_path, device = device)
            image_index = image_index + 1
        
    caching_labels = None
    if (bool_caching_labels):
        caching_labels   = torch.full(size = (n_images,), fill_value = dummy_class_index, dtype = torch.int64, device = device)

    return Dataset_generic_imgs_labels_paths(
        transform                       = transform,
        classes_list                    = classes_list,
        caching_imgs                    = caching_imgs,
        caching_labels                  = caching_labels,
        dataset_folder                  = dataset_folder,
        caching_relative_img_path       = caching_relative_img_path,
        bool_return_data_images         = bool_return_data_images,
        bool_true_tensors_false_images  = bool_true_tensors_false_images,
        bool_return_relative_img_names  = bool_return_relative_img_names,
        bool_return_short_img_names     = bool_return_short_img_names,
        device                          = device)

       

# Fonction qui généralise le comportement de "datasets.ImageFolder"
# Fonctionnalités supplémentaires:
# * Dossier de tenseurs au lieu d'images (par exemple: Deep features - CLS tokens)
# * Possibilité de caching des images(tenseurs) en RAM.
# * Caching des noms des images
# Output du data_loader: Des batchs de la forme: (transformed_img_data, label, short_img_path)
# Les champs label et short_img_path sont optionnels.
def datasets_generic_ImageFolder(
            dataset_folder,
            bool_return_data_images         = True,
            bool_true_tensors_false_images  = False,
            bool_caching_images             = False,
            bool_caching_labels             = False,
            bool_return_relative_img_names  = False,
            bool_return_short_img_names     = False,
            bool_one_hot_encoding           = False,
            bool_one_hot_remove_last_col    = False,    # For BCE remove the last class, i.e. the 'no_species' class
            transform                       = torch.nn.Identity(),
            device                          = None):

    if (False):
        print("debug:", dataset_folder,
                        bool_true_tensors_false_images,
                        bool_caching_images,
                        bool_caching_labels,
                        bool_return_short_img_names,
                        transform,
                        device)
                        
    if ((not (bool_true_tensors_false_images)) and bool_caching_images):
        print("datasets_generic_ImageFolder - Erreur: Caching d'images non implémenté (Seul le caching de tenseurs est implémenté).")
        exit(-1)
    # A partir d'ici, si du caching d'image est demandé, c'est nécessairement du caching de tenseur.
                        
    # Cas particulier ou il n'y a pas de sous-dossier:
    # Toutes les images sont dans le dossier principal => on ne connait pas la ground true => classe 'dummy'
    has_sub_folders = any(os.path.isdir(os.path.join(dataset_folder, d)) for d in os.listdir(dataset_folder))
    if (not has_sub_folders):
        return datasets_generic_SingleFolder(
            dataset_folder,
            bool_return_data_images,
            bool_true_tensors_false_images,
            bool_caching_images,
            bool_caching_labels,
            bool_return_relative_img_names,
            bool_return_short_img_names,
            transform,
            device)

    

    # Cas standard: Des images, Sans caching, Sans conservation du chemin
    # On utilise la fonction usuelle "datasets.ImageFolder()".
    if ((not (bool_true_tensors_false_images)) and \
        (not (bool_caching_images)) and \
        (not (bool_return_short_img_names))):
            return datasets.ImageFolder(dataset_folder, transform = transform)



    if (device == None):
        print("Error: datasets_ImageFolder_with_or_without_caching > Requires a device")
        exit(-1)
        
    # 1) Premier passage: Pour connaitre ces valeurs (n_images, n_views, n_features)
    #    afin d'allouer le tenseur de deep features, avant de le remplir
    classes_list = sorted(list(os.listdir(dataset_folder)), key = str)
    n_class = len(classes_list)
    image_index = 0
    f_path_first_tensor = None
    for class_entry in classes_list:
        class_full_path = os.path.join(dataset_folder, class_entry)
        if os.path.isdir(class_full_path):
            class_img_path_list = os.listdir(class_full_path)
            image_index = image_index + len(class_img_path_list)
            if ((f_path_first_tensor == None) and (len(class_img_path_list) > 0)):
                f_path_first_tensor = os.path.join(class_full_path, class_img_path_list[0])

    n_images = image_index

    print("Dataset en création avec :")
    print("Dataset_folder: ", dataset_folder)
    print("n_images:", n_images, ", n_class:", n_class)
    print("Liste des especes:", classes_list)

    if (bool_caching_images):
        if (f_path_first_tensor == None):
            print("datasets_ImageFolder_with_caching > Error: All classes are empty")
            exit(-1)

        first_tensor = load_torch_tensor(f_path_first_tensor, device = device)
        n_views, n_features = first_tensor.shape

        print("n_views:", n_views, ", n_features:", n_features)
        print("Empreinte mémoire - caching des deep features (Go): ", (4 * n_images * n_views * n_features / 1024 / 1024 / 1024))


    # 2) Allouer les tenseurs de sortie
    caching_imgs = None
    if (bool_caching_images):
        caching_imgs = torch.zeros(n_images, n_views, n_features, device = device)
    else:
        caching_imgs = None

    if (bool_caching_labels):
        caching_labels   = torch.zeros(n_images, dtype = torch.int64, device = device)    # Index de la bonne classe
    else:
        caching_labels = None

    caching_relative_img_path = []


    # 3) Second passage: Remplissage du gros tenseur de features nouvellement créé
    image_index = 0
    for index_class in range(n_class):
        class_relative_path = classes_list[index_class]
        class_full_path = os.path.join(dataset_folder, class_relative_path)
        if os.path.isdir(class_full_path):
            class_img_path_list = sorted(list(os.listdir(class_full_path)), key = str)
            relative_class_img_path_list = []
            for class_img_path in class_img_path_list:
                relative_class_img_path = os.path.join(class_relative_path, class_img_path)
                relative_class_img_path_list.append(relative_class_img_path)

            if (bool_caching_labels):
                caching_labels[image_index:(image_index + len(class_img_path_list))] = int(index_class)       # Index de la bonne classe

            if (bool_caching_images):
                for relative_class_img_path in relative_class_img_path_list:
                    full_class_img_path = os.path.join(dataset_folder, relative_class_img_path)
                    caching_imgs[image_index] = load_torch_tensor(full_class_img_path, device = device)
                    image_index = image_index + 1
            else:
                image_index = image_index + len(relative_class_img_path_list)

            caching_relative_img_path = caching_relative_img_path + relative_class_img_path_list

    if (bool_one_hot_encoding and bool_caching_labels):
        caching_labels = one_hot(input = caching_labels, num_classes = n_class).float()
        if (bool_one_hot_remove_last_col):
            caching_labels = caching_labels[:, :-1]


    print("Dataset créé")

    if (False):
        print("debug")
        print("transform:", transform)
        print("caching_imgs:", caching_imgs)
        print("caching_labels:", caching_labels)
        print("caching_relative_img_path:", caching_relative_img_path)
        print("bool_true_tensors_false_images:", bool_true_tensors_false_images)
        print("bool_return_short_img_names:", bool_return_short_img_names)
        print("bool_one_hot_encoding:", bool_one_hot_encoding)
        exit()

    #print("Appel à Dataset_generic_imgs_labels_paths")
    return Dataset_generic_imgs_labels_paths(
            transform                       = transform,
            classes_list                    = classes_list,
            caching_imgs                    = caching_imgs,
            caching_labels                  = caching_labels,
            dataset_folder                  = dataset_folder,
            caching_relative_img_path       = caching_relative_img_path,
            bool_return_data_images         = bool_return_data_images,
            bool_true_tensors_false_images  = bool_true_tensors_false_images,
            bool_return_relative_img_names  = bool_return_relative_img_names,
            bool_return_short_img_names      = bool_return_short_img_names,
            device                          = device)




# Get data to build the annotations.csv file (with wrappe_predictions_in_dataframe_with_common_names_and_tensors_list) :
# - The list of classes names
# - The list of images
# - A tensor with one hot encoding of the label/GT:  (0, 0, 0, 1, 0, 0) <-> class_index = 3 (starting with zero)
def get_annotations_data(dataset_folder, dict_classes_folder_to_name = None):

    device = torch.device("cpu")

    # datasets_generic_ImageFolder() returns a dataset class, containing all the relevant info..
    dataset = datasets_generic_ImageFolder(
            dataset_folder,
            bool_return_data_images         = False,  # Don't load the images data when getitem() is called
            bool_true_tensors_false_images  = True,   # To prevent calling the torchvision ImageFolder()
            bool_caching_images             = False,
            bool_caching_labels             = True,   # The Ground Truth
            bool_return_relative_img_names  = False,
            bool_return_short_img_names     = False,
            transform                       = None,
            device                          = device)

    classes_list             = dataset.classes_list
    short_img_names_list     = [img_path.split("/")[-1] for img_path in dataset.caching_relative_img_path]
    caching_labels           = dataset.caching_labels
    
    if (dict_classes_folder_to_name is not None):
        classes_list_names = []
        for class_folder in classes_list:
            if (not (class_folder in dict_classes_folder_to_name.keys())):
                print("Erreur: Classe manquante dans le dictionnaire:", class_folder)
                print("Clefs disponibles:", dict_classes_folder_to_name.keys())
                exit(-1)
            classes_list_names.append(dict_classes_folder_to_name[class_folder])
    else:
        classes_list_names   = classes_list
    
    #print(short_img_names_list[445:455])
    #print(caching_labels[445:455])
    #exit()

    # Trier les données selon le nom des images
    #print(short_img_names_list[:5])
    short_img_names_list, caching_labels = zip(*sorted(zip(short_img_names_list, caching_labels.tolist())))
    short_img_names_list = list(short_img_names_list)
    caching_labels = torch.tensor(caching_labels)
    #print(short_img_names_list[:5])
    #exit()

    num_classes = len(classes_list)
    #print(num_classes)
    #exit()
    caching_labels_one_hot   = one_hot(input = caching_labels, num_classes = len(classes_list))
    
    if (False):
        print("Nombre d'images par classe:")
        for class_index in range(len(classes_list)):
            print(classes_list[class_index], classes_list_names[class_index], " ", torch.sum(caching_labels == class_index).item())
        exit()
    
    return [classes_list_names, short_img_names_list, caching_labels_one_hot]
    
