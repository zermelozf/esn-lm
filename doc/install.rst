Installing esn-lm
=================


esn-lm depends on numpy, scipy, scikit-learn and Oger. Furthermore Oger depends on mdp.

An easy way to install these libraries is to use git and github since the default packages proposed
by ubuntu aren't always up to date. 

First create a directory to clone the libraries::

	mkdir python_scilibs
	cd python_scilibs

Then copy-paste these lines in the terminal to clone the different libraries::

	git clone git://github.com/numpy/numpy.git
	git clone git://github.com/scipy/scipy.git
	git clone git://github.com/scikit-learn/scikit-learn.git
	git clone git://github.com/mdp-toolkit/mdp-toolkit.git
	git clone git://github.com/npinto/Oger.git
	git clone git://github.com/zermelozf/esn-lm.git
	
Then use the setup.py files in each of the cloned directories to install the libraries. For example::

	cd ./numpy/
	sudo python setup.py install

You can also script the process::

	for path in './numpy/' './scipy/' './scikit-learn/' './mdp-toolkit/' './Oger/' './esn-lm/'
	do
	cd $path
	sudo python setup.py install
	cd ..
	done



	
sudo python setup.py install


