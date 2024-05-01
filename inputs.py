from slippi import Game

import pandas as pd


def gamesFromPaths(paths):
    """
    Given a list of paths, load them all in as py-slippi games and return that list.
    """
    games = []
    for p in paths:
        games.append(Game(p))
    
    return games


# get the port indices used by the players
def getPorts(game):
    ports = []
    for i in range(0, len(game.start.players)):
        if game.start.players[i] is not None:
            ports.append(i)

    return ports

# determine filenames of extracted input files
def decideFileNames(path):
    # get the names of our 2 files
    names = path.split('/')[-2].split('_')[0:2] # get the 2 player names from the directory name

    # datetime for the game
    gametime = path.split('_')[-1].split('.')[0]

    # name for file w/ p1 inputs
    p1_name = names[0] + '_' + names[1] + '_' + gametime
    # name for file w/ p2 inputs
    p2_name = names[1] + '_' + names[0] + '_' + gametime

    return p1_name, p2_name

# get all of the characters played in the game
def getCharacters(game):
    """Given a py-slippi game object, return a list of the characters used ordered by port."""
    characters = []
    for player in game.start.players:
        if player is not None:
            characters.append(player.character)
    
    return characters


START_FRAME = 64 # game starts on frame 64
COLS = ['joy_x', 'joy_y', 'cstick_x', 'cstick_y', 'z', 'r_dig', 'l_dig', 'a', 'b', 'x', 'y']
BUTTONS = ['Z', 'R', 'L', 'A', 'B', 'X', 'Y'] # names of the buttons in py-slippi

def getFrameInputs(player):
    """
    Get the state of all buttons/analog sticks for a given player.
    """
    # analog stick / c stick
    analog = [player.joystick.x, player.joystick.y, player.cstick.x, player.cstick.y]

    # buttons
    pressed_buttons = []
    # get the names of the buttons currently being pressed
    logical_pressed_names = map(lambda x: x.name, player.buttons.physical.pressed())

    for b in BUTTONS:

        if b in logical_pressed_names:
            pressed_buttons.append(1)
        else:
            pressed_buttons.append(0)

    return analog + pressed_buttons 

def getGameInputs(game):
    """
    Get the inputs of both players for all frames from a given game.
    """

    p1_inputs = pd.DataFrame(columns=COLS)
    p2_inputs = pd.DataFrame(columns=COLS)

    # get the controller port #s of the players
    p1_port, p2_port = getPorts(game)

    for i in range(START_FRAME, len(game.frames)):
        p1 = game.frames[i].ports[p1_port].leader.pre 
        p1_inputs.loc[len(p1_inputs.index)] = getFrameInputs(p1)

        p2 = game.frames[i].ports[p2_port].leader.pre
        p2_inputs.loc[len(p2_inputs.index)] = getFrameInputs(p2)

    return p1_inputs, p2_inputs