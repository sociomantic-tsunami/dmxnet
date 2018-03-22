# Ensure D2 unittests will fail if stomping prevention is triggered
export ASSERT_ON_STOMPING_PREVENTION=1

override DFLAGS += -w
override LDFLAGS += -lmxnet

ifeq ($(DVER),1)
	override DFLAGS += -v2 -v2=-static-arr-params -v2=-volatile
endif

.PHONY: download-mnist
download-mnist: $C/script/download-mnist
	$(call exec,sh $(if $V,,-x) $^,$(MNIST_DATA_DIR),$^)

# extra build dependencies for integration tests
$O/test-mxnet: override LDFLAGS += -lz
$O/test-mxnet: override DFLAGS += -debug=MXNetHandleManualFree

# extra runtime dependencies for integration tests
$O/test-mxnet.stamp: override ITFLAGS += $(MNIST_DATA_DIR)
$O/test-mxnet.stamp: download-mnist

$O/%unittests: override LDFLAGS += -lz
