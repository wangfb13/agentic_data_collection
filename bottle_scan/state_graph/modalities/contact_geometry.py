"""Reusable two-contact geometry descriptor in sensor pixel coordinates."""
import numpy as np


def contact_geometry(window,spec=None):
    contacts=window['contacts']
    vectors=[np.asarray([side[k] for k in ('u_px','v_px','area_px')],float) for side in contacts]
    if len(vectors)!=2: raise ValueError('This modality requires two contact surfaces')
    return np.concatenate([(vectors[0]+vectors[1])/2,vectors[0]-vectors[1]]).tolist()
