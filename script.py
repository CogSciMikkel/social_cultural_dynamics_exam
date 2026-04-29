from psychopy import gui, core, visual, event
import pandas as pd
import os
import random
from datetime import date

if not os.path.exists("logs"):
    os.makedirs("logs")

def check_exit():
    if event.getKeys(keyList=["escape"]):
        win.close()
        core.quit()

# creating gui
dialogue = gui.Dlg(title = "Welcome to our experiment!")

# adding response fields
dialogue.addText("Participant Info")
dialogue.addFixedField("participant_id", label = "Participant ID: ", initial = random.randint(1000,9999))
dialogue.addField("age", label = "Age: ")
dialogue.addField("gender", label = "Gender: ", choices=["Female", "Male", "Other"])
dialogue.addField("mental", label = "Have you been diagnosed with one or more of the following mental health disorders? Anxiety, depression, autism spectrum", choices=["Yes", "No", "Prefer not to say"])

# showing gui
dialogue.show()

# saving responses as variables
if dialogue.OK:
    ID = dialogue.data["participant_id"]
    age = dialogue.data["age"]
    gender = dialogue.data["gender"]
elif dialogue.Cancel:
    core.quit()

# defining window
win = visual.Window(color = "black", fullscr = True)

# random condition assignment
condition = random.choice(["coop", "comp"])

# add a welcome message - press spacebar to continue
welcome = visual.TextStim(win, "Thank you so much for participating in our experiment.\n\nPress space to continue.")
welcome.draw()
win.flip()
event.waitKeys(keyList = ["space"])

# add instructions - which button to click
instruction = visual.TextStim(win, "INSERT INSTRUCTIONS\n\nPress space when you are ready to begin.")
instruction.draw()
win.flip()
event.waitKeys(keyList = ["space"])

start = visual.TextStim(win, "The experiment will begin in 2 seconds.")
start.draw()
win.flip()
core.wait(2)





















finish = visual.TextStim(win, "Thank you for participating!\n\nThe experiment will now end.\nOur love for you will not ♥")
finish.draw()
win.flip()
core.wait(10)

# get time and date for unique logfile name
date = date.today().strftime("%b-%d-%Y")

# create logfile path and name
log_name = "logs/log_{}_{}.csv".format(ID, date)

# saving data to logfile
#log = pd.DataFrame(reaction_times, columns = ["ID", "age", "gender", "condition", "word", "reaction_time"]) # <-- Update this!!
#log.to_csv(log_name, index = False)