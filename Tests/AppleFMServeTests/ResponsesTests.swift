import XCTest
@testable import AppleFMServe

private func decodeResponses(_ json: String) throws -> ResponsesRequest {
    try JSONDecoder().decode(ResponsesRequest.self, from: Data(json.utf8))
}

private func responsesHandler(
    modelID: String = "apple.fm.system",
    responseText: String = "hello back",
    snapshots: [String] = ["H", "He", "Hello"]
) -> ChatHandler {
    let config = Config(
        host: "127.0.0.1", port: 8000, modelID: modelID, apiKey: nil,
        maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
        useCase: "general", guardrails: "default"
    )
    return ChatHandler(config: config, provider: MockProvider(modelID: modelID, responseText: responseText, snapshots: snapshots))
}

final class ResponsesTests: XCTestCase {
    func testStringInputProducesMessageOutput() async throws {
        let handler = responsesHandler()
        let request = try decodeResponses("{\"model\": \"apple.fm.system\", \"input\": \"hello\"}")
        let response = try await handler.completeResponses(request: request, model: request.model)
        XCTAssertTrue(response.id.hasPrefix("resp_"))
        XCTAssertEqual(response.status, "completed")
        guard case .message(let id, let text) = response.output.first else {
            return XCTFail("expected message output")
        }
        XCTAssertTrue(id.hasPrefix("msg_"))
        XCTAssertEqual(text, "hello back")
        XCTAssertGreaterThan(response.usage?.totalTokens ?? 0, 0)
    }

    func testToolsProduceFunctionCallItems() async throws {
        let handler = responsesHandler(responseText: "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}")
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": "do",
         "tools": [{"type": "function", "name": "noop", "description": "noop"}]}
        """)
        let response = try await handler.completeResponses(request: request, model: request.model)
        XCTAssertEqual(response.output.count, 1)
        guard case .functionCall(let id, let callId, let name, let args) = response.output.first else {
            return XCTFail("expected function_call output")
        }
        XCTAssertTrue(id.hasPrefix("fc_"))
        XCTAssertTrue(callId.hasPrefix("call_"))
        XCTAssertEqual(name, "noop")
        XCTAssertTrue(args.contains("\"x\""))
    }

    func testHistoryMergePreservesCallIDs() throws {
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": [
          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do"}]},
          {"type": "function_call", "call_id": "call_9", "name": "noop", "arguments": "{}"},
          {"type": "function_call_output", "call_id": "call_9", "output": "ok"},
          {"type": "message", "role": "user", "content": "continue"}]}
        """)
        let chat = try request.asChatRequest()
        XCTAssertEqual(chat.messages.first(where: { $0.role == "assistant" })?.toolCalls?.first?.id, "call_9")
        XCTAssertEqual(chat.messages.first(where: { $0.role == "tool" })?.toolCallID, "call_9")
    }

    func testImageBlockRejected() {
        do {
            let request = try decodeResponses("""
            {"model": "apple.fm.system", "input": [
              {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}
            """)
            _ = try request.asChatRequest()
            XCTFail("expected image_not_supported")
        } catch let adapter as AdapterError {
            XCTAssertEqual(adapter.code, "image_not_supported")
        } catch {
            XCTFail("wrong error \(error)")
        }
    }

    func testUnknownModelRejected() async {
        let handler = responsesHandler()
        let request = try! decodeResponses("{\"model\": \"wrong\", \"input\": \"hi\"}")
        do {
            _ = try await handler.completeResponses(request: request, model: request.model)
            XCTFail("expected model_not_found")
        } catch let adapter as AdapterError {
            XCTAssertEqual(adapter.code, "model_not_found")
        } catch {
            XCTFail("wrong error \(error)")
        }
    }

    func testStreamEventSequence() async throws {
        let handler = responsesHandler()
        let request = try decodeResponses("{\"model\": \"apple.fm.system\", \"input\": \"hi\"}")
        let events = try await handler.responsesEvents(request: request, model: request.model)
        let payload = events.joined()
        XCTAssertTrue(payload.contains("\"type\":\"response.created\""))
        XCTAssertTrue(payload.contains("\"type\":\"response.output_text.delta\""))
        XCTAssertTrue(payload.contains("\"type\":\"response.completed\""))
        XCTAssertFalse(payload.contains("[DONE]"))
    }

    func testMaxOutputTokensMapped() throws {
        let request = try decodeResponses("{\"model\": \"apple.fm.system\", \"input\": \"hi\", \"max_output_tokens\": 7}")
        XCTAssertEqual(try request.asChatRequest().effectiveMaxTokens, 7)
    }

    func testInstructionsAcceptBlockArray() throws {
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": "hi",
         "instructions": [{"type": "input_text", "text": "Be terse."}]}
        """)
        XCTAssertEqual(request.instructions, "Be terse.")
        let chat = try request.asChatRequest()
        XCTAssertEqual(chat.messages.first?.role, "system")
    }

    func testSloppyToolsAndBlocksDoNotFail() throws {
        // Missing tool type defaults; numeric block text is tolerated.
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": "hi",
         "tools": [{"name": "noop"}],
         "tool_choice": "none"}
        """)
        XCTAssertEqual(request.tools?.first?.type, "function")
        let items = try decodeResponses("""
        {"model": "apple.fm.system", "input": [
          {"type": "message", "role": "user",
           "content": [{"type": "input_text", "text": 42}]}]}
        """)
        let chat = try items.asChatRequest()
        XCTAssertEqual(chat.messages.count, 1)
    }
}
