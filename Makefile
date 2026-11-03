# libkgsl_fake -- the device layer it replaces is libc-level, so it builds for glibc and for bionic
# everyday use:  make            build both .so files
#                make test       run tests/run.sh (works on GPU-less aarch64)
# for the Android/bionic device:  make bionic    (run it inside the device's own image, whose clang
#                                 targets aarch64-linux-android; PROPS to override with BIONIC_PROPS)
# no Linux on hand? use a container: see the README "Quick start" section

UNAME_S := $(shell uname -s)
ifeq ($(filter clean help,$(MAKECMDGOALS)),)
ifneq ($(UNAME_S),Linux)
$(error this must be built on Linux (on macOS use a container: docker run --rm --platform linux/arm64 -v $$PWD:/work -w /work <your arm64 glibc image> make test))
endif
endif

BUILD   := build
CC      ?= clang
CPPFLAGS:= -Iinclude -Isrc -I$(BUILD) -D__user=
CFLAGS  ?= -O2 -g -std=gnu11 -Wall -Wextra -Wno-unused-parameter
FPIC    := -fPIC
LDFLAGS := -shared

FAKE_SRC := src/log.c src/raw.c src/mem.c src/props.c src/device.c src/ion.c src/interpose.c
REC_SRC  := src/rec.c src/raw.c
PROPS    := props/a6xx.properties

BIONIC_BUILD := $(BUILD)/bionic
BIONIC_PROPS ?= props/a5xx.properties

.PHONY: all bionic clean test help
all: $(BUILD)/libkgsl_fake.so $(BUILD)/libkgsl_rec.so $(BUILD)/probe $(BUILD)/tripwire

# bionic's loader holds a non-recursive lock while constructors run, so a lazy PLT call from one of them
# deadlocks: -z now binds every reference before the first hook can be reached, and the ctors are empty
# there anyway (see the __ANDROID__ guards in interpose.c and rec.c).
$(BIONIC_BUILD):
	@mkdir -p $(BIONIC_BUILD)

$(BIONIC_BUILD)/props_table.h: $(BIONIC_PROPS) tools/genprops.py | $(BIONIC_BUILD)
	python3 tools/genprops.py $(BIONIC_PROPS) -o $@

$(BIONIC_BUILD)/libkgsl_fake.so: $(FAKE_SRC) src/fkgsl.h src/raw.h $(BIONIC_BUILD)/props_table.h | $(BIONIC_BUILD)
	$(CC) $(CFLAGS) $(FPIC) -Iinclude -Isrc -I$(BIONIC_BUILD) -D__user= -Wl,-z,now -o $@ $(FAKE_SRC) $(LDFLAGS) -ldl

$(BIONIC_BUILD)/libkgsl_rec.so: $(REC_SRC) src/fkgsl.h src/raw.h include/msm_kgsl.h src/pm4_names.h | $(BIONIC_BUILD)
	$(CC) $(CFLAGS) $(FPIC) -Iinclude -Isrc -I$(BIONIC_BUILD) -D__user= -Wl,-z,now -o $@ $(REC_SRC) $(LDFLAGS) -ldl

bionic: $(BIONIC_BUILD)/libkgsl_fake.so $(BIONIC_BUILD)/libkgsl_rec.so

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/props_table.h: $(PROPS) tools/genprops.py | $(BUILD)
	python3 tools/genprops.py $(PROPS) -o $@

$(BUILD)/libkgsl_fake.so: $(FAKE_SRC) src/fkgsl.h src/raw.h $(BUILD)/props_table.h | $(BUILD)
	$(CC) $(CFLAGS) $(FPIC) $(CPPFLAGS) -o $@ $(FAKE_SRC) $(LDFLAGS) -ldl

$(BUILD)/libkgsl_rec.so: $(REC_SRC) src/fkgsl.h src/raw.h include/msm_kgsl.h src/pm4_names.h | $(BUILD)
	$(CC) $(CFLAGS) $(FPIC) $(CPPFLAGS) -o $@ $(REC_SRC) $(LDFLAGS) -ldl

# the OpenCL probe only builds where the closed-source runtime exists; failure does not block the main targets
$(BUILD)/probe: tests/probe.c | $(BUILD)
	-$(CC) $(CFLAGS) $(CPPFLAGS) -DCL_TARGET_OPENCL_VERSION=120 -o $@ $< -lOpenCL

# leak-guard field test, needs no OpenCL
$(BUILD)/tripwire: tests/tripwire.c include/msm_kgsl.h | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $<

test: all
	sh tests/run.sh

clean:
	@for f in $(BUILD)/libkgsl_fake.so $(BUILD)/libkgsl_rec.so $(BUILD)/probe $(BUILD)/props_table.h \
		$(BIONIC_BUILD)/libkgsl_fake.so $(BIONIC_BUILD)/libkgsl_rec.so $(BIONIC_BUILD)/props_table.h; do \
		python3 -c "import os,sys; [os.unlink(x) for x in sys.argv[1:] if os.path.exists(x)]" $$f; done
	@echo cleaned

help:
	@sed -n '1,12p' Makefile
