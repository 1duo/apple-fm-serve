import XCTest
@testable import AppleFMServe

final class PromptTests: XCTestCase {
    func testBuildPromptWithInstructionsAndTurns() throws {
        let built = try buildPrompt(messages: [
            ChatMessage(role: "system", content: .text("You are concise.")),
            ChatMessage(role: "user", content: .text("Hello")),
            ChatMessage(role: "assistant", content: .text("Hi")),
            ChatMessage(role: "user", content: .text("Summarize this.")),
        ])
        XCTAssertEqual(built.instructions, "You are concise.")
        XCTAssertTrue(built.prompt.contains("User: Hello"))
        XCTAssertTrue(built.prompt.contains("Assistant: Hi"))
        XCTAssertTrue(built.prompt.hasSuffix("Assistant:"))
    }

    func testBuildPromptFoldsDeveloperIntoInstructions() throws {
        let built = try buildPrompt(messages: [
            ChatMessage(role: "developer", content: .text("Be terse.")),
            ChatMessage(role: "user", content: .text("Hello")),
        ])
        XCTAssertEqual(built.instructions, "Be terse.")
        XCTAssertTrue(built.prompt.contains("User: Hello"))
    }

    func testBuildPromptRejectsImageParts() {
        XCTAssertThrowsError(try buildPrompt(messages: [
            ChatMessage(role: "user", content: .parts([MessagePart(type: "image_url", text: nil, imageURL: ["url": "https://example.com/a.png"])])),
        ])) { error in
            let adapter = error as? AdapterError
            XCTAssertEqual(adapter?.code, "image_not_supported")
        }
    }

    func testBuildPromptRequiresTurns() {
        XCTAssertThrowsError(try buildPrompt(messages: [
            ChatMessage(role: "system", content: .text("only system")),
        ])) { error in
            XCTAssertEqual((error as? AdapterError)?.code, "no_turns")
        }
    }

    func testBuildPromptFormatsToolHistory() throws {
        let built = try buildPrompt(messages: [
            ChatMessage(role: "user", content: .text("do thing")),
            ChatMessage(
                role: "assistant",
                content: .text(""),
                toolCalls: [ToolCall(id: "call_1", function: FunctionCall(name: "noop", arguments: "{\"x\":1}"))],
                toolCallID: nil
            ),
            ChatMessage(role: "tool", content: .text("{\"ok\":true}"), toolCalls: nil, toolCallID: "call_1"),
            ChatMessage(role: "user", content: .text("continue")),
        ])
        XCTAssertTrue(built.prompt.contains("call_1"))
        XCTAssertTrue(built.prompt.contains("Tool[call_1]"))
    }

    func testTextDeltasFromSnapshots() {
        XCTAssertEqual(textDeltasFromSnapshots(["H", "He", "Hello"]), ["H", "e", "llo"])
    }

    func testTextDeltasRecoverOnDivergence() {
        XCTAssertEqual(textDeltasFromSnapshots(["abc", "ab", "xyz"]), ["abc", "ab", "xyz"])
    }

    func testApplyStopSequences() {
        let (text, truncated) = applyStopSequences("hello STOP world", stop: .single("STOP"))
        XCTAssertEqual(text, "hello ")
        XCTAssertTrue(truncated)
        let (untouched, notTruncated) = applyStopSequences("hello world", stop: .single("STOP"))
        XCTAssertEqual(untouched, "hello world")
        XCTAssertFalse(notTruncated)
    }

    func testEstimateTokens() {
        XCTAssertEqual(estimateTokens(""), 0)
        XCTAssertEqual(estimateTokens("hi"), 1)
        XCTAssertEqual(estimateTokens("12345678"), 2)
    }
}
