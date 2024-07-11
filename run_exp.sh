#!/bin/bash


######################################################################################################################################################
###################################################################### PoseNet #######################################################################
######################################################################################################################################################

# Basement
#python train.py --model Resnet34 --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34_Basement-2024-05-10-00:57:23/49_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt

#python train.py --model MobilenetV3 --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3_Basement-2024-05-10-01:02:58/49_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
## Lower_Level
#python train.py --model Resnet34 --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34_Lower_Level-2024-05-10-01:03:58/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3_Lower_Level-2024-05-10-01:04:55/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
## Level_1
#python train.py --model Resnet34 --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34_Level_1-2024-05-10-13:42:05/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3_Level_1-2024-05-10-13:44:38/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
## Level_2
#python train.py --model Resnet34 --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34_Level_2-2024-05-10-13:45:25/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3_Level_2-2024-05-10-13:46:40/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#
#######################################################################################################################################################
################################################################## Bayesian PoseNet ###################################################################
#######################################################################################################################################################
#
## Basement
#python train.py --model Resnet34 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_Resnet34_Basement-2024-05-10-13:59:15/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_MobilenetV3_Basement-2024-05-10-14:01:05/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
## Lower_Level
#python train.py --model Resnet34 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_Resnet34_Lower_Level-2024-05-10-14:01:39/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_MobilenetV3_Lower_Level-2024-05-10-14:02:44/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
## Level_1
#python train.py --model Resnet34 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_Resnet34_Level_1-2024-05-10-14:04:06/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_MobilenetV3_Level_1-2024-05-10-14:05:03/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
## Level_2
#python train.py --model Resnet34 --dropout_rate 0.1 --bayesian True \
#--pretrained_model ./models_Resnet34_Level_2-2024-05-10-14:13:59/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
python train.py --model MobilenetV3 --dropout_rate 0.1 --bayesian True \
--pretrained_model ./models_MobilenetV3_Level_2-2024-05-10-14:16:00/best_net.pth \
--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#
#
#######################################################################################################################################################
#################################################################### LSTM-PoseNet #####################################################################
#######################################################################################################################################################
## Basement
#python train.py --model Resnet34lstm --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34lstm_Basement-2024-05-10-14:18:29/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
#python train.py --model MobilenetV3lstm --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
## Lower_Level
#python train.py --model Resnet34lstm --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34lstm_Lower_Level-2024-05-10-14:20:29/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
#python train.py --model MobilenetV3lstm --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
## Level_1
#python train.py --model Resnet34lstm --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
#python train.py --model MobilenetV3lstm --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3lstm_Level_1-2024-05-10-14:23:58/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
## Level_2
#python train.py --model Resnet34lstm --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#python train.py --model MobilenetV3lstm --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3lstm_Level_2-2024-05-10-14:25:38/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#
#
#######################################################################################################################################################
################################################################## Learnable PoseNet ##################################################################
#######################################################################################################################################################
#
## Basement
#python train.py --model Resnet34 --dropout_rate 0.1 --sx 0 --sq -5 --learn_beta True \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --sx 0 --sq -5 --learn_beta True \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
## Lower_Level
#python train.py --model Resnet34 --dropout_rate 0.1 --sx 0 --sq -5 --learn_beta True \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --sx 0 --sq -5 --learn_beta True \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
## Level_1
#python train.py --model Resnet34 --dropout_rate 0.1 --sx 3.65 --sq -1.77 --learn_beta True \
#--pretrained_model ./models_Resnet34_Level_1-2024-05-10-14:36:26/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --sx 0 --sq -5 --learn_beta True \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
## Level_2
#python train.py --model Resnet34 --dropout_rate 0.1 --sx 3.65 --sq -1.77 --learn_beta True \
#--pretrained_model ./models_Resnet34_Level_2-2024-05-10-14:38:30/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#python train.py --model MobilenetV3 --dropout_rate 0.1 --sx 0 --sq -5 --learn_beta True \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#
#
#######################################################################################################################################################
################################################################## Hourglass PoseNet ##################################################################
#######################################################################################################################################################
#
## Basement
#python train.py --model Resnet34hourglass --lr 0.001 --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
#python train.py --model MobilenetV3hourglass --lr 0.001 --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3hourglass_Basement-2024-05-10-14:47:12/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt
#
## Lower_Level
#python train.py --model Resnet34hourglass --lr 0.001 --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
#python train.py --model MobilenetV3hourglass --lr 0.001 --dropout_rate 0.1 \
#--pretrained_model ./models_MobilenetV3hourglass_Lower_Level-2024-05-10-14:50:33/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Lower_Level/image_train_all.txt
#
## Level_1
#python train.py --model Resnet34hourglass --lr 0.001 --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34hourglass_Level_1-2024-05-10-14:51:34/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
#python train.py --model MobilenetV3hourglass --lr 0.001 --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_1/image_train_all.txt
#
## Level_2
#python train.py --model Resnet34hourglass --lr 0.001 --dropout_rate 0.1 \
#--pretrained_model ./models_Resnet34hourglass_Level_2-2024-05-10-14:53:58/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#
#python train.py --model MobilenetV3hourglass --lr 0.001 --dropout_rate 0.1 \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2 \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Level_2/image_train_all.txt
#

#python train.py --model Branchresnet34 --use_euler6 True --lr 0.01
#python train.py --lr 0.01 --model BranchmobilenetV3 --use_euler6 True



#python train.py --model Resnet34 --geometric True --lr 0.000001 --num_epochs 400 \
#--pretrained_model ./models_Resnet34_Basement-2024-04-23-01:15:16/best_net.pth \
#--proj_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement \
#--metadata_path /data/juy220/LU\ Student\ Dropbox/Jun\ Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/geometric_data.pkl