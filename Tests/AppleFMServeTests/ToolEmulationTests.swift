import XCTest
@testable import AppleFMServe

final class ToolEmulationTests: XCTestCase {
    func testExtractJSONObject() {
        let text = "Sure! {\"content\": \"hi\", \"tool_calls\": []} done"
        XCTAssertEqual(
            extractJSONObject(from: text),
            "{\"content\": \"hi\", \"tool_calls\": []}"
        )
        XCTAssertNil(extractJSONObject(from: "no json here"))
    }

    func testParseToolResponseWithCalls() {
        let text = "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}"
        let parsed = parseToolResponse(text, validNames: ["noop"])
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertEqual(parsed.toolCalls[0].function.name, "noop")
        XCTAssertTrue(parsed.toolCalls[0].function.arguments.contains("\"x\""))
        XCTAssertTrue(parsed.toolCalls[0].id.hasPrefix("call_"))
    }

    func testParseToolResponseIgnoresUnknownFunctions() {
        let text = "{\"content\": \"hello\", \"tool_calls\": [{\"name\": \"unknown_fn\", \"arguments\": {}}]}"
        let parsed = parseToolResponse(text, validNames: ["noop"])
        XCTAssertTrue(parsed.toolCalls.isEmpty)
        XCTAssertEqual(parsed.content, "hello")
    }

    func testParseToolResponseFallsBackToPlainText() {
        let parsed = parseToolResponse("just a normal reply", validNames: ["noop"])
        XCTAssertTrue(parsed.toolCalls.isEmpty)
        XCTAssertEqual(parsed.content, "just a normal reply")
    }

    func testParseToolResponseUnwrapsEmptyEnvelope() {
        let parsed = parseToolResponse(
            "{\"content\": \"hello!\", \"tool_calls\": []}",
            validNames: ["noop"]
        )
        XCTAssertTrue(parsed.toolCalls.isEmpty)
        XCTAssertEqual(parsed.content, "hello!")
        XCTAssertFalse(parsed.recovered)
    }

    func testStrictParseIsNotMarkedRecovered() {
        let parsed = parseToolResponse(
            "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}",
            validNames: ["noop"]
        )
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertFalse(parsed.recovered)
    }

    func testRecoverMalformedEnvelope() {
        // Shape observed live from codex: `"key": value` pairs inside an
        // arguments array are invalid JSON, but delimiters still balance.
        let text = "{\"content\": \"Creating file.\", \"tool_calls\": [{\"name\": \"apply_patch\", \"arguments\": {\"cmd\": [\"apply_patch\", \"*** Begin Patch\", \"workdir\": \"/tmp\", \"tty\": true]}}]}"
        let parsed = parseToolResponse(text, validNames: ["apply_patch"])
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertEqual(parsed.toolCalls[0].function.name, "apply_patch")
        XCTAssertTrue(parsed.toolCalls[0].function.arguments.contains("Begin Patch"))
        XCTAssertEqual(parsed.content, "Creating file.")
        XCTAssertTrue(parsed.recovered)
    }

    func testRecoverReturnsNilWithoutToolCallsKey() {
        XCTAssertNil(recoverToolCalls("just prose, no envelope"))
    }

    func testRecoverEmitsUnknownNamesForHarnessFeedback() {
        // Recovery (unlike strict parsing) does not filter names: emitting
        // the invented call lets the harness report an unknown-tool error
        // back instead of stalling on dead text.
        let text = "{\"content\": \"on it\", \"tool_calls\": [{\"name\": \"invented_fn\", \"arguments\": {\"cmd\": [\"x\", \"bad\": 1]}}]}"
        let parsed = parseToolResponse(text, validNames: ["real_fn"])
        XCTAssertTrue(parsed.recovered)
        XCTAssertEqual(parsed.toolCalls.count, 1)
        XCTAssertEqual(parsed.toolCalls[0].function.name, "invented_fn")
    }

    func testToolChoiceRequiresTools() {
        XCTAssertTrue(toolChoiceRequiresTools(AnyCodable("required")))
        XCTAssertFalse(toolChoiceRequiresTools(AnyCodable("auto")))
        XCTAssertFalse(toolChoiceRequiresTools(AnyCodable("none")))
        XCTAssertFalse(toolChoiceRequiresTools(nil))
        XCTAssertTrue(toolChoiceRequiresTools(AnyCodable([
            "type": "function",
            "function": ["name": "noop"] as [String: Any],
        ] as [String: Any])))
        XCTAssertFalse(toolChoiceRequiresTools(AnyCodable(["type": "none"] as [String: Any])))
    }

    func testToolInstructionsMentionsFunctions() {
        let tools = [ToolDefinition(
            type: "function",
            function: FunctionDefinition(name: "noop", description: "Does nothing", parameters: nil)
        )]
        let text = toolInstructions(for: tools, toolChoice: nil)
        XCTAssertTrue(text.contains("noop"))
        XCTAssertTrue(text.contains("tool_calls"))
    }

    func testToolChoiceNoneDisablesEmulation() throws {
        let data = """
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hi"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": "none"}
        """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionsRequest.self, from: data)
        XCTAssertFalse(shouldEmulateTools(request: request))
    }

    func testToolChoiceRequiredKeepsEmulation() throws {
        let data = """
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hi"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": "required"}
        """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionsRequest.self, from: data)
        XCTAssertTrue(shouldEmulateTools(request: request))
    }

    func testEnvelopeStatus() {
        XCTAssertEqual(envelopeStatus("just prose", validNames: ["noop"]), .none)
        XCTAssertEqual(
            envelopeStatus("{\"content\": \"hi\", \"tool_calls\": []}", validNames: ["noop"]),
            .empty
        )
        XCTAssertEqual(
            envelopeStatus(
                "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {}}]}",
                validNames: ["noop"]
            ),
            .empty
        )
        XCTAssertEqual(
            envelopeStatus(
                "{\"content\": \"x\", \"tool_calls\": [{\"name\": \"ghost\", \"arguments\": {}}]}",
                validNames: ["noop"]
            ),
            .unusable
        )
        // Malformed but recoverable: no warning (recovery logs separately).
        XCTAssertEqual(
            envelopeStatus(
                "{\"content\": \"x\", \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"cmd\": [\"a\", \"k\": 1]}}]}",
                validNames: ["noop"]
            ),
            .empty
        )
        // Malformed beyond recovery: warn.
        XCTAssertEqual(
            envelopeStatus("{\"content\": \"x\", \"tool_calls\": \"oops\"}", validNames: ["noop"]),
            .unusable
        )
    }

    func testRecoverDoesNotStealLaterCallArguments() {
        // First call lacks "arguments"; the second call's must not leak into it.
        // (Outer braces balance; inner array is malformed so strict fails.)
        let text = "{\"content\": null, \"tool_calls\": [{\"name\": \"first\", \"oops\": [1, \"k\": 2]}, {\"name\": \"second\", \"arguments\": {\"x\": 1, \"bad\": [2, \"k\": 3]}}]}"
        let parsed = parseToolResponse(text, validNames: ["first", "second"])
        XCTAssertTrue(parsed.recovered)
        XCTAssertEqual(parsed.toolCalls.count, 2)
        XCTAssertEqual(parsed.toolCalls[0].function.name, "first")
        XCTAssertEqual(parsed.toolCalls[0].function.arguments, "{}")
        XCTAssertEqual(parsed.toolCalls[1].function.name, "second")
        XCTAssertTrue(parsed.toolCalls[1].function.arguments.contains("\"x\""))
    }

    func testParseUnicodeEscapes() {
        // \u0041 = A, \u00e9 = é, surrogate pair = 😀.
        let (plain, _) = parseJSONString(in: "\"\\u0041\\u00e9\"", from: "\"\\u0041\\u00e9\"".startIndex)!
        XCTAssertEqual(plain, "Aé")
        let (emoji, _) = parseJSONString(in: "\"\\ud83d\\ude00\"", from: "\"\\ud83d\\ude00\"".startIndex)!
        XCTAssertEqual(emoji, "😀")
        XCTAssertNil(parseJSONString(in: "\"\\u12\"", from: "\"\\u12\"".startIndex))
        // End-to-end through recovery content extraction.
        let parsed = parseToolResponse(
            "{\"content\": \"caf\\u00e9\", \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"cmd\": [\"x\", \"k\": 1]}}]}",
            validNames: ["noop"]
        )
        XCTAssertTrue(parsed.recovered)
        XCTAssertEqual(parsed.content, "café")
    }
}
