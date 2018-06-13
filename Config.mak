INTEGRATIONTEST := integrationtest

MXNET_ENGINE_TYPE ?= NaiveEngine
export MXNET_ENGINE_TYPE

ifeq ($(DVER),2)
	DC ?= dmd-transitional
endif
