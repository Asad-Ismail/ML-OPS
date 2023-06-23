dvc stage add -n train -d train_segmentation.py -d pets_data \
          -o modelCheckpoints/ \
          -p train.epochs \
          -M dvclive/metrics.json \
          python train_segmentation.py