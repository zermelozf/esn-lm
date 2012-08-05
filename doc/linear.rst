Linear Regression
=================

Description
-----------

The linear regression is the conventional readout for echo state networks.
It can be learned in one shot using least square fitting. 
There is no intercept by default so it must be added "manually".
Since the prediction of a linear function is not well normalized and may even be 
less than 0, It is renormalized appropriately.

References
----------

.. automodule:: esnlm.readouts.linear
   :members:
   :undoc-members: