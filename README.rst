Language models with ESNs 
=========================

Install using::

	cd esnlm/
	sudo python setup.py install
	
Test using::
	
	nosetests --with-doctest --doctest-extension=.rst -v
	
Build the html documentation::

	cd doc/
	make html

or in pdf::
	
	make latexpdf
