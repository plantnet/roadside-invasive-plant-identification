#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides standard code for deep learning : train deep learning models, compute predictions over a given dataset

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

import os
import sys
import time
import torch
from torch import nn
from torch.cuda.amp import autocast
from contextlib import contextmanager

def get_device(device_name):
    return torch.device(device_name)

def affiche_tag_heure():
  from datetime import datetime
  import time
  now = datetime.now()
  print("tag heure: " + str(now.strftime("%Y-%m-%d_%H:%M:%S")) + " (" + str(time.time()) + ")")


def print_epoch_like_keras(epoch, n_epochs):
  print('Epoch ' + str(epoch) + '/' + str(n_epochs))


def print_lines_like_keras(n_batch, duration, epoch_loss, epoch_accuracy):
  s_part1 = str(n_batch) + '/' + str(n_batch) + ' - ' + str(int(duration))
  s_part2 = "s - loss: {:.4f} - accuracy: {:.4f}".format(epoch_loss, epoch_accuracy) 
  print(s_part1 + s_part2)


def print_lines_like_keras_avec_validation(n_batch, duration, epoch_loss, epoch_accuracy, val_epoch_loss, val_epoch_accuracy):
  s_part1 = str(n_batch) + '/' + str(n_batch) + ' - ' + str(int(duration))
  s_part2 = "s - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}".format(epoch_loss, epoch_accuracy, val_epoch_loss, val_epoch_accuracy)
  print(s_part1 + s_part2)


# Implémentation de la fonction evaluation avec PyTorch
def model_eval(model, data_loader, criterion, calc_accuracy = False, printPerf = True):

    date_debut = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Bascule le modèle en mode évaluation
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        running_loss = 0.0
        n_species = 6
        running_loss_per_species = torch.zeros(n_species, device = torch.device("cuda"))

        if (calc_accuracy):
          running_corrects = 0.0
        n_verif_imgs = 0
        for inputs, labels in data_loader:
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            outputs = model(inputs)

            #print(outputs)
            #print(labels)
            if (False):
              loss           = criterion(outputs, labels)
              loss_per_species = torch.zeros(n_species, device = torch.device("cuda"))
            else:

              #batch_size = 256
              batch_size = outputs.shape[0]
              loss = 0
              loss_per_species = torch.zeros(n_species, device = torch.device("cuda"))
              import math
              for row_index in range(batch_size):
                for species_index in range( n_species):
                  y = labels[row_index, species_index]
                  x = outputs[row_index, species_index]
                  try:
                    # Implémentation manuelle de la BCE, comme pytorch
                    # cf: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
                    if (x > 0.0):
                      log_x = math.log(x)
                    else:
                      log_x = -100.
                      print("Saturation du log !!")

                    if (x < 1.0):
                      log_1mx = math.log(1. - x)
                    else:
                      log_1mx = -100.
                      print("Saturation du log !!")

                    loss_curr = -1.0 * (y * log_x + (1 - y) * log_1mx)
                  except:
                    print("Erreur dans le calcul manuel de la BCE:", x, y)
                    loss_curr = 0.0
                    exit()
                  loss_per_species[species_index] += loss_curr
                  loss += loss_curr

            running_loss              += loss.item()
            running_loss_per_species  += loss_per_species

            if (calc_accuracy):
              _, predictions = torch.max(outputs, 1)
              running_corrects += torch.sum(predictions == labels.data)
            n_verif_imgs     += inputs.shape[0]

        epoch_loss     = running_loss / len(data_loader.dataset)

        epoch_loss_per_species = torch.zeros(n_species, device = torch.device("cuda"))
        for species_index in range(n_species):
          epoch_loss_per_species[species_index] = running_loss_per_species[species_index].item() / len(data_loader.dataset)

        if (calc_accuracy):
          epoch_accuracy = running_corrects.float()/ len(data_loader.dataset)
        else:
          epoch_accuracy = -1

        date_fin = time.perf_counter()
        duree_epoch = (date_fin - date_debut)

        if (printPerf):
            print_lines_like_keras(
            n_batch = len(data_loader),
            duration = duree_epoch,
            epoch_loss = epoch_loss,
            epoch_accuracy = epoch_accuracy)

        print("epoch_loss_per_species:", epoch_loss_per_species)
        print("n_verif_imgs:", n_verif_imgs)

    return [epoch_loss, epoch_accuracy, duree_epoch]




def model_fit_train_one_epoch(model, data_loader, criterion, optimizer, calc_accuracy = False, printPerf = False):

  date_debut = time.perf_counter()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Bascule le modèle en mode training
  model.train()

  running_loss = 0.0
  if (calc_accuracy):
    running_corrects = 0.0
  for inputs, labels in data_loader:
    #print("model_fit_train_one_epoch iteration")
    #print(inputs)
    #print(labels)
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    #exit()


    # Pass forward: Appel à la fonction de coût
    loss           = criterion(outputs, labels)
    if (calc_accuracy):
      _, predictions = torch.max(outputs, 1)

    #loss = criterion(outputs, labels)

    # Met tous les gradiants de l'optimizer à zéro.
    optimizer.zero_grad()

    # Pass Backward: Calcule tous les gradiants (pour les parametres tels que: requires_grad = True.)
    loss.backward()

    # Modifie la valeur de tous les paramètres
    # En appliquant l'algorithme de rétropropagation du gradiant
    optimizer.step()
    
    #_, predictions = torch.max(outputs, 1)

    #running_loss     += loss.item()
    running_loss     += loss
    if (calc_accuracy):
      running_corrects += torch.sum(predictions == labels.data)
  
  # Modif pour des raisons de performance (utilisation GPU)
  #epoch_loss     = running_loss / len(data_loader.dataset)
  epoch_loss     = running_loss.item() / len(data_loader.dataset)

  if (calc_accuracy):
    epoch_accuracy = running_corrects.float()/ len(data_loader.dataset)
  else:
    epoch_accuracy = -1

  date_fin = time.perf_counter()
  duree_epoch = (date_fin - date_debut)

  if (printPerf):
    print_lines_like_keras(
      n_batch        = len(data_loader),
      duration       = duree_epoch,
      epoch_loss     = epoch_loss,
      epoch_accuracy = epoch_accuracy)

  return [epoch_loss, epoch_accuracy, duree_epoch]



def ReduceLROnPlateau_reduceLR(epoch, optimizer, reduce_lr_factor):
  optimizer_state_dict = optimizer.state_dict()
  lr_curr = optimizer_state_dict["param_groups"][0]["lr"]
  lr_curr = lr_curr * reduce_lr_factor
  optimizer_state_dict["param_groups"][0]["lr"] = lr_curr
  optimizer.load_state_dict(optimizer_state_dict)
  print("Epoch {:05d}: ReduceLROnPlateau reducing learning rate to {:e}".format(epoch, lr_curr))



# Implémentation de la fonction apprentissage avec PyTorch
def model_fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        n_epochs,
        filename_model_weights_out,
        early_stop_patience,
        early_stop_min_delta,
        reduce_lr_patience,
        reduce_lr_factor,
        tf_train_writer = None,
        tf_val_writer = None,
        calc_accuracy = False):

  duree_totale = 0

  if (False):
      print("model_fit > Appel")
      print("Modèle:")
      print(model)

      print("train set - nb images:", len(train_loader.dataset), "; nb batchs:", len(train_loader))
      print("val   set - nb images:", len(val_loader.dataset),   "; nb batchs:", len(val_loader))

      print("criterion:"); print(criterion)
      print("optimizer:"); print(optimizer)
      print("n_epochs:"); print(n_epochs)

      print("filename_model_weights_out:"); print(filename_model_weights_out)
      print("early_stop_patience:");  print(early_stop_patience)
      print("early_stop_min_delta:"); print(early_stop_min_delta)
      print("reduce_lr_patience:");   print(reduce_lr_patience)
      print("reduce_lr_factor:");     print(reduce_lr_factor)

  train_loss_history = []
  train_accuracy_history = []
  val_loss_history = []
  val_accuracy_history = []

  best_val_epoch_loss_checkpoint    = float('inf')
  best_val_epoch_loss_earlystopping = float('inf')

  early_stop_n_epoch = 0
  reduce_lr_n_epoch  = 0

  for epoch in range(1, n_epochs + 1):

    affiche_tag_heure()
    sys.stdout.flush()

    # Apprentissage: Une itération avec l'algo de rétropropagation du gradiant
    [train_epoch_loss, train_epoch_accuracy, train_duree_epoch] = model_fit_train_one_epoch(model, train_loader, criterion, optimizer, calc_accuracy, printPerf = False)


    # Evaluation sur le dataset "val_loader"
    [val_epoch_loss, val_epoch_acc, val_duree_epoch] = model_eval(model, val_loader, criterion, printPerf = False)

    if (tf_train_writer != None):
      tf_train_writer.add_scalar('loss', train_epoch_loss, epoch)
    if (tf_val_writer != None):
      tf_val_writer.add_scalar('loss', val_epoch_loss, epoch)
    
    # Enregistrement de l'historique des métriques
    train_loss_history.append(train_epoch_loss)
    train_accuracy_history.append(train_epoch_accuracy)
    val_loss_history.append(val_epoch_loss)
    val_accuracy_history.append(val_epoch_acc)

    duree_totale += (train_duree_epoch + val_duree_epoch)


    # Affichage des résultats
    print_epoch_like_keras(epoch, n_epochs)
    print_lines_like_keras_avec_validation(
      n_batch            = len(train_loader),
      duration           = train_duree_epoch + val_duree_epoch,
      epoch_loss         = train_epoch_loss,
      epoch_accuracy     = train_epoch_accuracy,
      val_epoch_loss     = val_epoch_loss,
      val_epoch_accuracy = val_epoch_acc)


    # Implémentation des Callback, de façon similaire à Keras

    # La val_loss a baissé
    if (val_epoch_loss < best_val_epoch_loss_checkpoint):
      # Checkpoint -> On enregistre les poids
      #filename_model_weights_out_checkpoint = filename_model_weights_out.replace(".pth", "_checkpoint" + str(epoch) + ".pth")
      filename_model_weights_out_checkpoint = filename_model_weights_out
      torch.save(model.state_dict(), filename_model_weights_out_checkpoint)
      print("Epoch {:05d}: val_loss improved from {:.5f} to {:.5f}, saving model to ".format(epoch, best_val_epoch_loss_checkpoint, val_epoch_loss) + filename_model_weights_out_checkpoint)
      best_val_epoch_loss_checkpoint = val_epoch_loss
      reduce_lr_n_epoch = 0
    else:
      # La val_loss n'a pas baissé.
      print("Epoch {:05d}: val_loss did not improve from {:.5f}".format(epoch, best_val_epoch_loss_checkpoint))
      reduce_lr_n_epoch = reduce_lr_n_epoch + 1
      if (reduce_lr_n_epoch >= reduce_lr_patience):
        ReduceLROnPlateau_reduceLR(epoch, optimizer, reduce_lr_factor)
        reduce_lr_n_epoch = 0


    # Early Stopping: La val_loss a baissé au moins de la quantité "early_stop_min_delta"
    if (val_epoch_loss < best_val_epoch_loss_earlystopping - early_stop_min_delta):
      best_val_epoch_loss_earlystopping = val_epoch_loss
      early_stop_n_epoch = 0
    else:
      early_stop_n_epoch = early_stop_n_epoch + 1
      if (early_stop_n_epoch >= early_stop_patience):
        print("Epoch {:05d}: early stopping".format(epoch))
        break

    # Vider le cache pour forcér l'envoi des prints
    sys.stdout.flush()


  return [train_loss_history,
          train_accuracy_history,
          val_loss_history,
          val_accuracy_history,
          duree_totale]



def perform_learning(
    model,
    train_loader,
    val_loader,
    n_epochs,
    init_learning_rate,
    filename_model_weights_out,

    early_stop_patience,
    reduce_lr_patience,
    early_stop_min_delta = 0.0,
    reduce_lr_factor = 1.0,

    label_smoothing_factor = 0.0,
    weight_decay = 0.0,
    bool_use_BCE_loss = False,
    
    bool_use_tensorboard = False,
    tensorflow_log_dir = "",
    tf_train_xp_name = "",
    tf_val_xp_name = ""):

    # Permet d'éviter que l'environnement CUDA freeze, et donne plus d'indications sur les erreurs CUDA.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    # Test d'une simple évaluation
    print("Evaluation avant apprentissage:")
    if (bool_use_BCE_loss):
        criterion = nn.BCELoss(reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum', label_smoothing = label_smoothing_factor)
    [epoch_loss, epoch_accuracy, duree_epoch] = model_eval(model, val_loader, criterion, printPerf = True)

    print("Début de l'apprentissage:")
    optimizer = torch.optim.Adam(model.parameters(), lr = init_learning_rate, weight_decay = weight_decay)

    tf_train_writer = None
    tf_val_writer = None
    if ((bool_use_tensorboard) and (n_epochs > 0)):

        ## Commandes Tensorboard
        # $ pip install tensorboard         # Installation de tensorboard (prend quelques secondes)
        # $ tensorboard --logdir=runs       # Lancement de tensorboard
        from torch.utils.tensorboard import SummaryWriter
        
        tf_train_writer = SummaryWriter(log_dir = os.path.join(tensorflow_log_dir, tf_train_xp_name))
        tf_val_writer = SummaryWriter(log_dir = os.path.join(tensorflow_log_dir, tf_val_xp_name))


    return model_fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        n_epochs,
        filename_model_weights_out,
        early_stop_patience,
        early_stop_min_delta,
        reduce_lr_patience,
        reduce_lr_factor,
        tf_train_writer = tf_train_writer,
        tf_val_writer = tf_val_writer,
        calc_accuracy = False)


# Remarque: Le dataloader doit servir 2 champs: "inputs, img_names"
def calc_predictions_and_output_tensor_and_images_names(
        model,
        data_loader,
        device,
        use_autocast,
        use_torch_compile,
        verbose = True):

    print("Appel à calc_predictions_and_output_tensor_and_images_names()")
    # Afficher les versions des principales librairies utilisées
    print("Version de python:", sys.version_info)
    print("Version de pytorch:", torch.__version__)


    if (False):
        filename_model_weights_out = "model_bugge.pth"
        state_dict = model.state_dict()
        torch.save(state_dict, filename_model_weights_out)
        print("Sauve le state dict:", filename_model_weights_out)

        model_param = "blocks.4.mlp.fc1.weight"
        print(model_param, ":", state_dict[model_param].shape)
        print(torch.mean(input = state_dict[model_param], dim = 1))
        print(torch.std(input = state_dict[model_param], dim = 1))
        model_param = "blocks.11.attn.qkv.weight"
        print(model_param, ":", state_dict[model_param].shape)
        print(torch.mean(input = state_dict[model_param], dim = 1))
        print(torch.std(input = state_dict[model_param], dim = 1))
        model_param = "head.weight"
        print(model_param, ":", state_dict[model_param].shape)
        print(torch.mean(input = state_dict[model_param], dim = 1))
        print(torch.std(input = state_dict[model_param], dim = 1))
        model_param = "head.bias"
        print(model_param, ":", state_dict[model_param].shape)
        print(state_dict[model_param])


    if (use_autocast):
        print("autocast activé")
    else:
        print("autocast désactivé")
        none_context = contextmanager(lambda: iter([None]))()
    
    # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    # Requires 'triton'  -> $ conda install pytorch::torchtriton
    # or : micromamba install pytorch::torchtriton==2.2.0
    if (use_torch_compile):
        try:
            print("torch compile activé")
            import triton
            print("Version de triton:", triton.__version__)            
            
            model_opt = torch.compile(model)
        except Exception as error:
            print("Warning: Le modèle n'a pas pu être compilé.")
            print(">>>>>>>>")
            print("Erreur:", type(error).__name__, " – ", error)
            print("<<<<<<<<")
            print("Poursuite des calculs sans torch compilation.")
            model_opt = model
    else:
        print("torch compile désactivé")
        model_opt = model

    model_opt.eval()
    model_opt = model_opt.to(device)


    n_images = len(data_loader.dataset)

    print("Début des inférences")

    img_index = 0
    all_img_names = []
    predictions = None
    #aeff = torch.zeros(4, 8, device = device)
    with torch.no_grad():
        with (autocast() if use_autocast else none_context):

            time_begin = None
            #for inputs, labels, img_names in data_loader:
            for inputs, img_names in data_loader:
                inputs  = inputs.to(device)
                if (False):
                    #if (True):
                    print("inputs")
                    print(torch.mean(input = inputs, dim = (2)))
                    print(torch.std(input = inputs, dim = (2)))
                    #print(torch.mean(input = inputs, dim = (2, 3)))
                    #print(torch.std(input = inputs, dim = (2, 3)))

                #labels  = labels.to(device)
                if (True):
                    outputs = model_opt(inputs)
                else:
                    # Remplir avec des valeurs dummy
                    outputs = aeff
                    #time.sleep(1)

                if (False):
                    print(" ")
                    print("inputs")
                    print(inputs.shape)
                    print("outputs")
                    print(outputs.shape)
                    print("img_names[0]", img_names[0])
                    print(" ")

                    img_first = inputs[0]

                    import numpy as np
                    import matplotlib.pyplot as plt
                    print("img_first.shape:", img_first.shape)
                    # Pour vérifications / debug
                    #norm_resized_batch_image = norm_resized_batch_image * 256
                    img_normalized = np.array(img_first.cpu())

                    # transpose from shape of (3,,) to shape of (,,3)
                    img_normalized = img_normalized.transpose(1, 2, 0)
                    print("img_normalized.shape:", img_normalized.shape)
                        
                    # display the normalized image
                    print(img_normalized)
                    plt.imshow(img_normalized)
                    plt.xticks([])
                    plt.yticks([])
                    plt.show()                
                    exit()
                    
                all_img_names = all_img_names + img_names
                n_batch_img = len(img_names)

                if (predictions == None):
                    # Premier batch
                    print("inputs inférence - premier batch")
                    print(torch.mean(input = inputs, dim = (2)))
                    print(torch.std(input = inputs, dim = (2)))

                    n_classes = outputs.shape[1]    # [batch_size, n_classes]
                    predictions = torch.zeros(n_images, n_classes, device = device)

                predictions[img_index:(img_index + n_batch_img)] = outputs
                img_index = img_index + n_batch_img

                time_now = time.time()
                print("Images traitées ", int(10000 * img_index / n_images) / 100, "%, img ", img_index, " sur ", n_images)
                if (False):
                    # Source: https://discuss.pytorch.org/t/how-to-check-the-gpu-memory-being-used/131220
                    #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                    #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                    #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    print(torch.cuda.memory_summary())

                if (time_begin == None):
                    time_begin = time.time()
                    img_index_start = img_index
                else:
                    n_img_per_sec = (img_index - img_index_start) / (time_now - time_begin)#(img_index + 1)

                    temps_restant = int(100 * (n_images - (img_index + 1)) / (n_img_per_sec * 60)) / 100
                    print("temps restant:", temps_restant, "min")

                    n_img_per_sec_display = int(100. * n_img_per_sec ) / 100.
                    print("Nb image par seconde:", n_img_per_sec_display)

                    
                sys.stdout.flush()            

    #print(predictions)
    #print(predictions.shape)
    #exit()
    print("Inférence terminée.")
    
    # Trier les futures lignes du predictions.csv suivant le short name des images
    sorted_all_img_names, permutation = zip(*sorted((s, i) for i, s in enumerate(all_img_names)))
    permutation_indices = torch.tensor(permutation)
    sorted_predictions = predictions[permutation_indices]

    #return [predictions, all_img_names]
    return [sorted_predictions, sorted_all_img_names]
    