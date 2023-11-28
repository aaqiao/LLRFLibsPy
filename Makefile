#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
# Make file for the LLRFLibsPy

default: help
help ::
	@echo "Makefile for LLRFLibsPy"
	@echo "======================================================"
	@echo "available targets:"
	@echo " -> make clean       clean the Python compilation"
	@echo "======================================================"

# remove all compiled data
clean ::
	rm -rf __pycache__
	rm -rf example/__pycache__
	rm -rf src/__pycache__

