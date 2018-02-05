INTEGRATIONTEST := integrationtest

ifeq ($(DVER),2)
	DC ?= dmd-transitional
endif
