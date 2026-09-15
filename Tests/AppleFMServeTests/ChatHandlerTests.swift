import XCTest
@testable import AppleFMServe

private func decodeRequest(_ json: String) throws -> ChatCompletionsRequest {
    try JSONDecoder().decode(ChatCompletionsRequest.self, from: Data(json.utf8))
}

private func testHandler(
    modelID: String = "apple.fm.system",
    responseText: String = "hello back",
    snapshots: [String] = ["H", "He", "Hello"],
    apiKey: String? = nil
) -> ChatHandler {
    let config = Config(
        host: "127.0.0.1", port: 8000, modelID: modelID, apiKey: apiKey,
        maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
        useCase: "general", guardrails: "default"
    )
    let provider = MockProvider(modelID: modelID, responseText: responseText, snapshots: snapshots)
    return ChatHandler(config: config, provider: provider)
}

final class ChatHandlerTests: XCTestCase {
    func testCompleteNonStream() async throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hello"}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.object, "chat.completion")
        XCTAssertEqual(response.choices[0].message.content, "hello back")
        XCTAssertEqual(response.choices[0].finishReason, "stop")
        XCTAssertNotNil(response.usage)
        XCTAssertGreaterThan(response.usage!.totalTokens, 0)
    }

    func testCompleteRejectsUnknownModel() async {
        let handler = testHandler()
        let request = try! decodeRequest("""
        {"model": "wrong.model", "messages": [{"role": "user", "content": "hello"}]}
        """)
        do {
            _ = try await handler.complete(request: request, model: request.model)
            XCTFail("expected model_not_found")
        } catch let adapter as AdapterError {
            XCTAssertEqual(adapter.code, "model_not_found")
            XCTAssertEqual(adapter.statusCode, 404)
        } catch {
            XCTFail("wrong error \(error)")
        }
    }

    func testCompleteEmulatesToolCalls() async throws {
        let handler = testHandler(responseText: "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}")
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "do thing"}],
         "tools": [{"type": "function", "function": {"name": "noop", "description": "noop"}}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].finishReason, "tool_calls")
        XCTAssertEqual(response.choices[0].message.toolCalls?.count, 1)
        XCTAssertEqual(response.choices[0].message.toolCalls?[0].function.name, "noop")
    }

    func testCompleteRetriesOnceWhenToolsRequired() async throws {
        let config = Config(
            host: "127.0.0.1", port: 8000, modelID: "apple.fm.system", apiKey: nil,
            maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
            useCase: "general", guardrails: "default"
        )
        let provider = MockProvider(
            modelID: "apple.fm.system",
            responseTexts: [
                "I will do it in prose.",
                "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {}}]}",
            ]
        )
        let handler = ChatHandler(config: config, provider: provider)
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": "required"}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].finishReason, "tool_calls")
        XCTAssertEqual(provider.generateCallCount, 2)
    }

    func testCompleteDoesNotRetryWhenToolsOptional() async throws {
        let config = Config(
            host: "127.0.0.1", port: 8000, modelID: "apple.fm.system", apiKey: nil,
            maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
            useCase: "general", guardrails: "default"
        )
        let provider = MockProvider(modelID: "apple.fm.system", responseTexts: ["just prose"])
        let handler = ChatHandler(config: config, provider: provider)
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].finishReason, "stop")
        XCTAssertEqual(provider.generateCallCount, 1)
    }

    func testCompleteSurfacesRecoveredCalls() async throws {
        let malformed = "{\"content\": \"on it\", \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"cmd\": [\"x\", \"bad\": 1]}}]}"
        let config = Config(
            host: "127.0.0.1", port: 8000, modelID: "apple.fm.system", apiKey: nil,
            maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
            useCase: "general", guardrails: "default"
        )
        let handler = ChatHandler(config: config, provider: MockProvider(modelID: "apple.fm.system", responseText: malformed))
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].finishReason, "tool_calls")
        XCTAssertEqual(response.choices[0].message.toolCalls?.first?.function.name, "noop")
    }

    func testCompleteStreamRetriesOnceWhenToolsRequired() async throws {
        let config = Config(
            host: "127.0.0.1", port: 8000, modelID: "apple.fm.system", apiKey: nil,
            maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
            useCase: "general", guardrails: "default"
        )
        let provider = MockProvider(
            modelID: "apple.fm.system",
            responseTexts: [
                "prose without envelope",
                "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {}}]}",
            ]
        )
        let handler = ChatHandler(config: config, provider: provider)
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "stream": true,
         "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": "required"}
        """)
        var payload = ""
        for try await chunk in handler.completeStream(request: request, model: request.model) {
            payload += chunk
        }
        XCTAssertTrue(payload.contains("tool_calls"))
        XCTAssertTrue(payload.contains("\"finish_reason\":\"tool_calls\""))
        XCTAssertEqual(provider.generateCallCount, 2)
    }

    func testCompleteToolHistoryAccepted() async throws {
        let handler = testHandler(responseText: "final answer")
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [
          {"role": "user", "content": "do thing"},
          {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "noop", "arguments": "{\\"x\\":1}"}}]},
          {"role": "tool", "tool_call_id": "call_1", "content": "{\\"ok\\":true}"},
          {"role": "user", "content": "continue"}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].message.content, "final answer")
    }

    func testCompleteStream() async throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "stream": true, "messages": [{"role": "user", "content": "hello"}]}
        """)
        var payload = ""
        for try await chunk in handler.completeStream(request: request, model: request.model) {
            payload += chunk
        }
        XCTAssertTrue(payload.contains("data: [DONE]"))
        XCTAssertTrue(payload.contains("\"role\":\"assistant\""))
        XCTAssertTrue(payload.contains("\"content\":\"H\""))
        XCTAssertTrue(payload.contains("\"content\":\"e\""))
        XCTAssertTrue(payload.contains("\"content\":\"llo\""))
    }

    func testStreamOptionsAccepted() async throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "stream": true,
         "stream_options": {"include_usage": true},
         "messages": [{"role": "user", "content": "hello"}]}
        """)
        var payload = ""
        for try await chunk in handler.completeStream(request: request, model: request.model) {
            payload += chunk
        }
        XCTAssertTrue(payload.contains("data: [DONE]"))
        // OpenAI-standard trailing usage-only chunk (empty choices).
        XCTAssertTrue(payload.contains("\"choices\":[]"))
        XCTAssertTrue(payload.contains("\"prompt_tokens\""))
    }

    func testStreamWithoutIncludeUsageHasNoUsage() async throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "stream": true,
         "messages": [{"role": "user", "content": "hello"}]}
        """)
        var payload = ""
        for try await chunk in handler.completeStream(request: request, model: request.model) {
            payload += chunk
        }
        XCTAssertTrue(payload.contains("data: [DONE]"))
        XCTAssertFalse(payload.contains("prompt_tokens"))
    }

    func testNonStreamIncludesRefusalAndUsageDetails() async throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hello"}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        let data = try JSONEncoder().encode(response)
        let text = String(data: data, encoding: .utf8)!
        XCTAssertTrue(text.contains("\"refusal\":null"))
        XCTAssertTrue(text.contains("prompt_tokens_details"))
        XCTAssertTrue(text.contains("completion_tokens_details"))
    }

    func testAuthRequired() {
        let handler = testHandler(apiKey: "secret")
        XCTAssertThrowsError(try handler.checkAuth(authorization: nil)) { error in
            XCTAssertEqual((error as? AdapterError)?.code, "invalid_api_key")
        }
        XCTAssertNoThrow(try handler.checkAuth(authorization: "Bearer secret"))
    }

    func testResponseFormatJSONSchemaAccepted() async throws {
        let handler = testHandler(responseText: "{\"x\": 1}")
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "return json"}],
         "response_format": {"type": "json_schema", "json_schema": {"name": "obj", "schema": {"type": "object"}}}}
        """)
        let prepared = try handler.prepare(request)
        XCTAssertNotNil(prepared.responseSchemaJSON)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].finishReason, "stop")
    }

    func testResponseFormatJSONObjectRejected() async {
        let handler = testHandler()
        let request = try! decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hi"}],
         "response_format": {"type": "json_object"}}
        """)
        XCTAssertThrowsError(try handler.prepare(request)) { error in
            XCTAssertTrue((error as? AdapterError)?.message.contains("json_schema") ?? false)
        }
    }

    func testToolsWinOverResponseFormat() throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system",
         "messages": [{"role": "user", "content": "hi"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "response_format": {"type": "json_schema", "json_schema": {"name": "o", "schema": {"type": "object"}}}}
        """)
        let prepared = try handler.prepare(request)
        XCTAssertTrue(prepared.emulateTools)
        XCTAssertNil(prepared.responseSchemaJSON)
    }

    func testToolsAndToolChoiceAccepted() async throws {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hello"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": {"type": "function", "function": {"name": "noop"}}}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        XCTAssertEqual(response.choices[0].finishReason, "stop")
    }
}
