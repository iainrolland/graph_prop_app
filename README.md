# Graph-based propagation tensor completion app

This interactive application demonstrates the graph-based propagation approach to tensor completion. The app shows two satellite images (Landsat-7) and lets you draw over one of them. Using the graph-based propagation technique the app completes the missing data in real-time in front of you!


# Quick-start

Run the `app.py` to use the application, e.g.

    python app.py
    
# Files
## Data

The images are contained within the numpy arrays: `al_0.npy` and `al_1.npy`.

## Propagation

The main component of the algorithm can be found in the `graph_prop()` function contained within  `diffusion.py`.
 ## Utility functions
 

Within `im_utils.py` and `adj_utils.py` are contained other helper functions.
