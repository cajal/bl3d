__version__ = '0.4.0'


import os

# Switch matplotlib backend if display is not set up correctly
cmd = 'python3 -c "import matplotlib.pyplot as plt; plt.figure()" 2> /dev/null'
if os.system(cmd): # if command fails
    print('No display found. Switching matplotlib backend to "Agg"')
    import matplotlib; matplotlib.use('Agg'); del matplotlib
