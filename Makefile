.PHONY: build verify test check run clean

build:
	swift build

verify:
	swift run apple-fm-verify

# `swift test` needs full Xcode (XCTest). With CommandLineTools only,
# `make verify` runs the same checks via the apple-fm-verify executable.
test: verify

check: build verify

run:
	./serve

clean:
	rm -rf .build
