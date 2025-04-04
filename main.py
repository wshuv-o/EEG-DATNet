from matplotlib import pyplot as plt
import numpy as np
data = np.load('A07E.npz')  # data contains the data of the subject 1
print(data['edur'])
signal = data['s']
# The index 7 represent the channel C3, for the info of each channel read the original paper.
channelC3 = signal[:, 7]
x = 7  # this is the event number that I want to extract
# Extract the type of the event 7 in this case the type is 768 (in the table this is a Start of a trial event).
etype = data['etyp'].T[0, x]
# This is the position of the event in the raw signal
epos = data['epos'].T[0, x]
edur = data['edur'].T[0, x]  # And this is the duration of this event
# Then I extract the signal related the event selected.
trial = channelC3[epos:epos+edur]
# The selected event type is 768 (Start of a trial) if you see the array of event types ('etype')
# you can observe the next event is 772 (Cue onset tongue) with that you can deduce de class of
# this trial: Tongue Imagery Task.
# Then for know the class of this trial (7) you need to read the type of the inmediate next event
trial_type = data['etyp'].T[0, x+1]
# For know the order of this events, you can see the data['etyp'] array.
# You can plot this event with matplotlib
plt.plot(trial)
plt.show()