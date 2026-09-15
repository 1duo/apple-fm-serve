import XCTest
@testable import AppleFMServe

final class OpenAITypesTests: XCTestCase {
    func testDecodesNullContentAndToolHistory() throws {
        let data = """
        {"model": "m", "messages": [
          {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
          {"role": "tool", "tool_call_id": "call_1", "content": "ok"}]}
        """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionsRequest.self, from: data)
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].toolCalls?.first?.id, "call_1")
        XCTAssertEqual(request.messages[1].toolCallID, "call_1")
    }

    func testIgnoresUnknownFields() throws {
        let data = """
        {"model": "m", "messages": [{"role": "user", "content": "hi"}],
         "logprobs": true, "n": 1, "presence_penalty": 0.5, "seed": 42}
        """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionsRequest.self, from: data)
        XCTAssertEqual(request.model, "m")
    }

    func testConfigFromEnvironment() {
        let config = Config.fromEnvironment([
            "APPLE_FM_HOST": "0.0.0.0",
            "APPLE_FM_PORT": "9000",
            "APPLE_FM_MODEL_ID": "apple.fm.system",
            "APPLE_FM_MAX_CONCURRENCY": "8",
        ])
        XCTAssertEqual(config.host, "0.0.0.0")
        XCTAssertEqual(config.port, 9000)
        XCTAssertEqual(config.maxConcurrency, 8)
        XCTAssertNil(config.apiKey)
        XCTAssertFalse(config.logBodies)
    }

    func testModelObjectIncludesCreated() throws {
        let data = try JSONEncoder().encode(ModelObject(id: "apple.fm.system"))
        let text = String(data: data, encoding: .utf8)!
        XCTAssertTrue(text.contains("\"created\""))
        XCTAssertTrue(text.contains("apple.fm.system"))
    }

    func testContentLengthPermitted() {
        XCTAssertTrue(contentLengthPermitted(0))
        XCTAssertTrue(contentLengthPermitted(maxRequestBodyBytes))
        XCTAssertFalse(contentLengthPermitted(maxRequestBodyBytes + 1))
        XCTAssertFalse(contentLengthPermitted(-1))
    }

    func testStatusTextCoversPayloadTooLarge() {
        XCTAssertEqual(HTTPResponse.statusText(for: 413), "Payload Too Large")
    }

    func testToolCallDecodingToleratesSparseHistory() throws {
        // Missing arguments/type/id must not fail the whole message.
        let data = """
        {"model": "m", "messages": [
          {"role": "assistant", "content": null,
           "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f"}}]},
          {"role": "assistant", "content": null,
           "tool_calls": [{"function": {"name": "g", "arguments": {"x": 1}}}]}]}
        """.data(using: .utf8)!
        let request = try JSONDecoder().decode(ChatCompletionsRequest.self, from: data)
        XCTAssertEqual(request.messages[0].toolCalls?.first?.function.arguments, "{}")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.function.name, "g")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.function.arguments, "{\"x\":1}")
        XCTAssertTrue(request.messages[1].toolCalls?.first?.id.hasPrefix("call_") ?? false)
    }

    func testMessagePartBareStringImageURLStillRejected() {
        XCTAssertThrowsError(try buildPrompt(messages: [
            ChatMessage(role: "user", content: .parts([
                MessagePart(type: "image_url", text: nil, imageURL: nil),
            ])),
        ])) { _ in }
        // Bare-string form decodes (rather than dropping the part) so the
        // adapter 400 path is reached instead of silent context loss.
        let data = """
        {"type": "image_url", "image_url": "https://example.com/a.png"}
        """.data(using: .utf8)!
        let part = try! JSONDecoder().decode(MessagePart.self, from: data)
        XCTAssertEqual(part.imageURL?["url"], "https://example.com/a.png")
        XCTAssertThrowsError(try buildPrompt(messages: [
            ChatMessage(role: "user", content: .parts([part])),
        ])) { error in
            XCTAssertEqual((error as? AdapterError)?.code, "image_not_supported")
        }
    }

    func testFunctionDefinitionToleratesSloppyFields() throws {
        let data = """
        {"type": "weird", "function": {"name": "f", "description": 42, "parameters": false}}
        """.data(using: .utf8)!
        let tool = try JSONDecoder().decode(ToolDefinition.self, from: data)
        XCTAssertEqual(tool.function.name, "f")
        XCTAssertNil(tool.function.description)
        XCTAssertNil(tool.function.parameters)
    }
}
