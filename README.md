# ThermoSnooker
The files in the submitted zip are:
1.	The python script: SnookerWorksheet.py.
2.	The simulation class module: simulation_module.py
3.	The ball class module: ball_module.py
4.	An extra functions module: func_module.py

SnookerWorksheet.py
is the computing script. It is divided into cells, one for each logical
unit, which I strongly recommend executing separately.
The parameters of the multiple simulations which are implemented are smaller
than the ones implemented in the simulations used (and documented) in the 
computing report. I changed them to make the script “runnable” in a 
reasonable amount of time. This means the results would qualitatively be the
same but might not be so quantitively. 
The script is more automatized and less hardcoded as possible: on the one 
hand, this involved extensive usage of loops, but on the other, if you want
to change parameters, you only have to do so at the beginning of the cells.

simulation_module.py
contains the simulation class with all its modules, and comments 
throughout it. The methods are described by doc strings. 
In the last days it was enhanced to run the simulation and perform some 
analysis with different types of balls. Not all methods were completely 
updated (add_balls).

ball_module.py
contains the ball class with all its modules.
