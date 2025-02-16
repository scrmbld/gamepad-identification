# Biometric Identification With Game Controllers

This repository includes code for performing biometric identification on *Super Smash Bros Melee* players using analog stick inputs performed during gameplay. Although biometric identification in games has been done before using mouse and keyboard inputs, I have been unable to find any examples of identifying players on traditional gamepads. This repository demonstrates a complete process from data collection to classification for game controller biometrics.

The main purpose of this experiment was to see if the analog stick on a game controller could be used to draw any useful conclusions about the player (such as their identity). As such, I only used the controller's left analog stick to do the identification. I believe that higher performance could be achieved by a model that also uses the buttons.

The most effective version of the system used a combined 1D Convolutional + LSTM architecture and achieved a top-1 accuracy of around 85% across 16 classes from 180 analog stick movements as input (on average ~10 seconds of gameplay).

## Usage

1. Clone the repository
    - `git clone git@github.com:scrmbld/gamepad-identification.git`
1. Install dependencies (using a virtual environment may be preferable)
    - `pip install tensorflow matplotlib numpy pandas scikit-learn seaborn py-slippi`
1. Download the dataset and place it in the "dataset" directory
    - slp replays: https://drive.google.com/file/d/1U_PeNa1P0tIoG1ZzmKTA2Ya2I_Ola6as/view
    - Controller state CSVs (smaller, enough to run the classifier): https://drive.google.com/file/d/15iWt38Gzxh9shicN2fhsukWBtXGJSOWn/view

The full experiment was a multi-step process, and those steps are split up between different notebooks. The following explains the purpose of each notebook and python file.

- `classifier.ipynb` demonstrates how the classifier is constructed and run
- `classifier.py` contains the actual code for the classifier, as used in some of the other notebooks
- `demo.ipynb` was created for a live demo of the project
- `extract_inputs.ipynb` reads the controller inputs from the replay files
- `inputs.py` contains functions which are useful for reading controller inputs from replays
- `hyperparam_seach.ipynb` was used to test a variety of hyperparameters in order to help refine the model
- `preprocess.ipynb` converts the data into an event-based representation
- `test_model.ipynb` collects more detailed performance data about the model
- `understand_data.ipynb` computes some population statistics (namely, events per second).

In order to run the model, the inputs must first be extracted from the replays by running `extract_inupts.ipynb` (if you downloaded the 'inputs' dataset, then this step has already been done for you). Then, the inputs will need to be preprocessed using `preprocess.ipynb`. Finally, the model in `classifier.ipynb` (or in any other notebook that includes the classifier) will be able to run. Once inputs have been extracted an preprocessed once, these steps do not need to be performed again.

## Data Collection

*Super Smash Bros Melee* was selected due to the existence of an open source replay system developed by the community, called Project Slippi. Much like similar systems in games such as *Doom*, the replay files contain all of the inputs that the players make over the course of a game.

There are existing datasets of *Melee* replays, but they are all intended for training agents to play the game and are not labelled appropriately for biometrics. For this reason I created my own dataset using publicly available replays from *Smash Summit 11* which I labelled manually by cross-referencing replay metadata with information on the tournament's start.gg page (using things like stage selection, game scores, characters, etc.). The resulting dataset contains data from 16 players against a healthy range of opponents with relatively balanced representation. However, it only covers an extremely short time period of just 3 days.

I then used the (now unmaintained) [py-slippi library](https://github.com/hohav/py-slippi) to parse the replays and read the controller state on every frame. I have made both the labelled replay file and controller state datasets available as google drive downloads in [Usage](#Usage).

## Data Preprocessing

I took only the left analog stick X and Y columns from the original controller state data. In most games, analog sticks have their physical position mapped to in game movement, in contrast to mice which map movement to movement. In other words, analog sticks are used to move characters in games by held in a certain position for as long as the player wants to continue moving. This means that the analog stick state data contains many rows which are identical to each other, as the player is holding the analog stick in one place. The information in the controller state data can be represented more compactly by listing all of the changes in position and adding an 'elapsed' column that stores the amount of time since the previous change. The more compact 'Displacement Event' representation drastically improved performance in my experiments. Additionally, displacement events which were more than 60 frames after the previous event had their 'elapsed' value set to 0 as a primitive form of outlier removal (although it may actually be better to clamp them to 60 instead).

## The Model

For the classifier model, I drew inspiration from mouse biometrics. It uses two separate 1D convolutional layers, the outputs of which are then concatenated before being passed to the next set of convolutions. There are two additional convolutions followed by a pair of recurrent layers. Both GRU and LSTM layers were tested, with the latter achieving marginally higher performance (~1% higher accuracy).

This is based on the convolutional model tested in "Adversarial Attacks on Remote User Authentication Using Behavioural Mouse Dynamics" by Tan et al.  
[https://arxiv.org/pdf/1905.11831](https://arxiv.org/pdf/1905.11831)