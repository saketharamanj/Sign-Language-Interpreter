# Sign-Language-Interpreter
A convolutional neural network built with PyTorch that classifies images of hand signs into digits (1–9). The model takes 64×64 RGB images, extracts features through three convolutional blocks (32 → 64 → 128 channels, each followed by ReLU and max-pooling), and passes them through a small fully-connected classifier head to predict the digit being signed.

How it works:
1) Loads training/test images via torchvision.datasets.ImageFolder, with each class represented by its own subfolder
2) Trains with cross-entropy loss and the Adam optimizer, using ReduceLROnPlateau to decay the learning rate when validation loss plateaus
3) Applies early stopping (patience of 7 epochs) and checkpoints the best-performing weights to best_model_slr.pth
4) Reports train/validation loss and accuracy each epoch, then evaluates on a held-out test batch at the end

Known limitation
The model has difficulty with numbers whose sign requires two hands rather than one — noted directly in the project's README.

Tech stack: Python, PyTorch, torchvision, NumPy, Matplotlib, OpenCV
