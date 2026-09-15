import Foundation

@testable import AppleFMServe

// MARK: - Minimal test harness (works with CommandLineTools, no XCTest)

var failures = 0

func check(_ condition: Bool, _ message: String, file: String = #file, line: Int = #line) {
    if !condition {
        failures += 1
        print("FAIL \(file):\(line): \(message)")
    }
}

func decodeRequest(_ json: String) throws -> ChatCompletionsRequest {
    try JSONDecoder().decode(ChatCompletionsRequest.self, from: Data(json.utf8))
}

func testHandler(
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
    return ChatHandler(config: config, provider: MockProvider(modelID: modelID, responseText: responseText, snapshots: snapshots))
}

func runAll() async {
    // Prompt tests
    do {
        let built = try buildPrompt(messages: [
            ChatMessage(role: "system", content: .text("You are concise.")),
            ChatMessage(role: "user", content: .text("Hello")),
            ChatMessage(role: "assistant", content: .text("Hi")),
            ChatMessage(role: "user", content: .text("Summarize this.")),
        ])
        check(built.instructions == "You are concise.", "instructions")
        check(built.prompt.contains("User: Hello"), "prompt user")
        check(built.prompt.hasSuffix("Assistant:"), "prompt suffix")
    } catch {
        check(false, "buildPrompt threw \(error)")
    }

    do {
        _ = try buildPrompt(messages: [
            ChatMessage(role: "user", content: .parts([MessagePart(type: "image_url", text: nil, imageURL: ["url": "x"])])),
        ])
        check(false, "image should throw")
    } catch let adapter as AdapterError {
        check(adapter.code == "image_not_supported", "image code")
    } catch {
        check(false, "wrong image error")
    }

    check(textDeltasFromSnapshots(["H", "He", "Hello"]) == ["H", "e", "llo"], "deltas")
    check(textDeltasFromSnapshots(["abc", "ab", "xyz"]) == ["abc", "ab", "xyz"], "deltas diverge")

    // Tool emulation
    check(extractJSONObject(from: "a {\"x\": 1} b") == "{\"x\": 1}", "extract json")
    let parsed = parseToolResponse(
        "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}",
        validNames: ["noop"]
    )
    check(parsed.toolCalls.count == 1, "tool parse count")
    check(parsed.toolCalls.first?.function.name == "noop", "tool parse name")
    let unwrapped = parseToolResponse("{\"content\": \"hello!\", \"tool_calls\": []}", validNames: ["noop"])
    check(unwrapped.toolCalls.isEmpty && unwrapped.content == "hello!", "empty envelope unwrap")

    // Handler: non-stream
    do {
        let handler = testHandler()
        let request = try decodeRequest("{\"model\": \"apple.fm.system\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}")
        let response = try await handler.complete(request: request, model: request.model)
        check(response.choices[0].message.content == "hello back", "complete content")
        check((response.usage?.totalTokens ?? 0) > 0, "complete usage")
    } catch {
        check(false, "complete threw \(error)")
    }

    // Handler: unknown model
    do {
        let handler = testHandler()
        let request = try decodeRequest("{\"model\": \"wrong\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")
        _ = try await handler.complete(request: request, model: request.model)
        check(false, "unknown model should throw")
    } catch let adapter as AdapterError {
        check(adapter.code == "model_not_found", "model_not_found code")
    } catch {
        check(false, "wrong model error")
    }

    // Handler: tool emulation
    do {
        let handler = testHandler(responseText: "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}")
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}]}
        """)
        let response = try await handler.complete(request: request, model: request.model)
        check(response.choices[0].finishReason == "tool_calls", "tool_calls finish")
    } catch {
        check(false, "tool emulate threw \(error)")
    }

    // Handler: stream
    do {
        let handler = testHandler()
        let request = try decodeRequest("{\"model\": \"apple.fm.system\", \"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")
        var payload = ""
        for try await chunk in handler.completeStream(request: request, model: request.model) {
            payload += chunk
        }
        check(payload.contains("data: [DONE]"), "stream DONE")
        check(payload.contains("\"role\":\"assistant\""), "stream role")
        check(!payload.contains("prompt_tokens"), "stream no usage without flag")
    } catch {
        check(false, "stream threw \(error)")
    }

    // Handler: stream with include_usage ends in usage-only chunk.
    do {
        let handler = testHandler()
        let request = try decodeRequest("{\"model\": \"apple.fm.system\", \"stream\": true, \"stream_options\": {\"include_usage\": true}, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")
        var payload = ""
        for try await chunk in handler.completeStream(request: request, model: request.model) {
            payload += chunk
        }
        check(payload.contains("\"choices\":[]"), "stream usage-only chunk")
        check(payload.contains("\"prompt_tokens\""), "stream usage present")
    } catch {
        check(false, "stream usage threw \(error)")
    }

    // Recovery from malformed envelopes + required-tools retry.
    do {
        let malformed = "{\"content\": \"on it\", \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"cmd\": [\"x\", \"bad\": 1]}}]}"
        let parsed = parseToolResponse(malformed, validNames: ["noop"])
        check(parsed.toolCalls.count == 1 && parsed.recovered, "recover malformed envelope")
        let invented = parseToolResponse(
            "{\"content\": \"on it\", \"tool_calls\": [{\"name\": \"invented_fn\", \"arguments\": {\"cmd\": [\"x\", \"k\": 1]}}]}",
            validNames: ["noop"]
        )
        check(invented.toolCalls.count == 1 && invented.recovered, "recover emits unknown names")
        let twoCalls = parseToolResponse(
            "{\"content\": null, \"tool_calls\": [{\"name\": \"first\", \"oops\": [1, \"k\": 2]}, {\"name\": \"second\", \"arguments\": {\"x\": 1, \"bad\": [2, \"k\": 3]}}]}",
            validNames: ["first", "second"]
        )
        check(
            twoCalls.toolCalls.count == 2 && twoCalls.toolCalls[0].function.arguments == "{}",
            "recover does not steal later arguments"
        )
        check(toolChoiceRequiresTools(AnyCodable("required")), "choice required")
        check(!toolChoiceRequiresTools(AnyCodable("auto")), "choice auto")
        check(!toolChoiceRequiresTools(nil), "choice nil")

        let config = Config(
            host: "127.0.0.1", port: 8000, modelID: "apple.fm.system", apiKey: nil,
            maxConcurrency: 4, requestTimeoutSeconds: 5, preferPCC: false,
            useCase: "general", guardrails: "default"
        )
        let retryProvider = MockProvider(
            modelID: "apple.fm.system",
            responseTexts: ["prose", "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {}}]}"]
        )
        let retryHandler = ChatHandler(config: config, provider: retryProvider)
        let retryRequest = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": "required"}
        """)
        let retryResponse = try await retryHandler.complete(request: retryRequest, model: retryRequest.model)
        check(retryResponse.choices[0].finishReason == "tool_calls", "retry yields tool calls")
        check(retryProvider.generateCallCount == 2, "retry made second call")

        // Stream path retries too.
        let streamRetryProvider = MockProvider(
            modelID: "apple.fm.system",
            responseTexts: ["prose", "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {}}]}"]
        )
        let streamRetryHandler = ChatHandler(config: config, provider: streamRetryProvider)
        let streamRetryRequest = try decodeRequest("""
        {"model": "apple.fm.system", "stream": true, "messages": [{"role": "user", "content": "do"}],
         "tools": [{"type": "function", "function": {"name": "noop"}}],
         "tool_choice": "required"}
        """)
        var streamPayload = ""
        for try await chunk in streamRetryHandler.completeStream(request: streamRetryRequest, model: streamRetryRequest.model) {
            streamPayload += chunk
        }
        check(streamPayload.contains("\"finish_reason\":\"tool_calls\""), "stream retry yields tool calls")
        check(streamRetryProvider.generateCallCount == 2, "stream retry made second call")

        // Envelope classification: empty is quiet, unusable warns.
        check(envelopeStatus("plain prose", validNames: ["noop"]) == .none, "envelope none")
        check(envelopeStatus("{\"content\": \"hi\", \"tool_calls\": []}", validNames: ["noop"]) == .empty, "envelope empty")
        check(
            envelopeStatus("{\"content\": \"x\", \"tool_calls\": [{\"name\": \"ghost\", \"arguments\": {}}]}", validNames: ["noop"]) == .unusable,
            "envelope unusable"
        )
        // Unicode escapes decode (BMP + surrogate pair).
        let jsonABCD = "\"\\u0041\\u00e9\""
        let (decoded, _) = parseJSONString(in: jsonABCD, from: jsonABCD.startIndex)!
        check(decoded == "Aé", "unicode escapes")
        let jsonEmoji = "\"\\ud83d\\ude00\""
        let (emoji, _) = parseJSONString(in: jsonEmoji, from: jsonEmoji.startIndex)!
        check(emoji == "😀", "surrogate pair")

        // Tolerant decoding: sparse tool_calls must not fail the turn.
        do {
            let request = try decodeRequest("""
            {"model": "apple.fm.system", "messages": [
              {"role": "assistant", "content": null,
               "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f"}}]},
              {"role": "assistant", "content": null,
               "tool_calls": [{"function": {"name": "g", "arguments": {"x": 1}}}]}]}
            """)
            check(request.messages[0].toolCalls?.first?.function.arguments == "{}", "tolerant missing args")
            check(request.messages[1].toolCalls?.first?.function.arguments == "{\"x\":1}", "tolerant object args")
            // Bare-string image_url decodes so the 400 path (not silent drop) triggers.
            let built = try buildPrompt(messages: [
                ChatMessage(role: "user", content: .parts([
                    MessagePart(type: "image_url", text: nil, imageURL: ["url": "https://example.com/a.png"]),
                ])),
            ])
            _ = built
            check(false, "bare image should throw")
        } catch let adapter as AdapterError {
            check(adapter.code == "image_not_supported", "bare image 400")
        } catch {
            check(false, "tolerant decode threw \(error)")
        }

        // tools + response_format: schema set aside.
        do {
            let handler = testHandler()
            let request = try decodeRequest("""
            {"model": "apple.fm.system",
             "messages": [{"role": "user", "content": "hi"}],
             "tools": [{"type": "function", "function": {"name": "noop"}}],
             "response_format": {"type": "json_schema", "json_schema": {"name": "o", "schema": {"type": "object"}}}}
            """)
            let prepared = try handler.prepare(request)
            check(prepared.emulateTools && prepared.responseSchemaJSON == nil, "tools win over response_format")
        } catch {
            check(false, "tools+format threw \(error)")
        }

        // max_tokens validation.
        do {
            let handler = testHandler()
            let request = try decodeRequest("""
            {"model": "apple.fm.system", "max_tokens": 0,
             "messages": [{"role": "user", "content": "hi"}]}
            """)
            _ = try handler.prepare(request)
            check(false, "max_tokens 0 should throw")
        } catch let adapter as AdapterError {
            check(adapter.code == "invalid_max_tokens", "max_tokens code")
        } catch {
            check(false, "max_tokens threw \(error)")
        }

        // Body-size guard.
        check(contentLengthPermitted(0), "body cap zero")
        check(contentLengthPermitted(maxRequestBodyBytes), "body cap at limit")
        check(!contentLengthPermitted(maxRequestBodyBytes + 1), "body cap over limit")
        check(!contentLengthPermitted(-1), "body cap negative")

        // JSON-Schema translator (needs AFM types: macOS 26+).
        if #available(macOS 26, *) {
            do {
                let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
                {"type": "object",
                 "properties": {
                   "color": {"type": "string", "enum": ["mauve", "chartreuse"]},
                   "tags": {"type": "array", "items": {"type": "string"}}
                 },
                 "required": ["color"]}
                """)
                let description = String(describing: schema)
                check(description.contains("mauve"), "translator enum")
                check(description.contains("tags"), "translator array")
                let shared = try buildGenerationSchema(name: "Response", jsonSchema: """
                {"type": "object",
                 "$defs": {"Pet": {"type": "object",
                   "properties": {"name": {"type": "string"}}, "required": ["name"]}},
                 "properties": {
                   "cat": {"$ref": "#/$defs/Pet"},
                   "dog": {"$ref": "#/$defs/Pet"}},
                 "required": ["cat", "dog"]}
                """)
                check(String(describing: shared).contains("Pet"), "translator shared def")
            } catch {
                check(false, "translator threw \(error)")
            }
            do {
                let open = try buildGenerationSchema(name: "Response", jsonSchema: "{\"type\": \"object\"}")
                check(!String(describing: open).isEmpty, "translator open object")
            } catch {
                check(false, "translator open object threw \(error)")
            }
            do {
                _ = try buildGenerationSchema(name: "Response", jsonSchema: """
                {"type": "object",
                 "properties": {"choice": {"oneOf": [{"type": "string"}]}}}
                """)
                check(false, "translator oneOf should throw")
            } catch {
                check(true, "translator oneOf throws")
            }
        }
    } catch {
        check(false, "recovery/retry checks threw \(error)")
    }

    // Developer role folds into instructions; refusal/usage details encoded.
    do {
        let built = try buildPrompt(messages: [
            ChatMessage(role: "developer", content: .text("Be terse.")),
            ChatMessage(role: "user", content: .text("Hi")),
        ])
        check(built.instructions == "Be terse.", "developer instructions")
        let handler = testHandler()
        let request = try decodeRequest("{\"model\": \"apple.fm.system\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")
        let response = try await handler.complete(request: request, model: request.model)
        let data = try JSONEncoder().encode(response)
        let text = String(data: data, encoding: .utf8)!
        check(text.contains("\"refusal\":null"), "refusal null")
        check(text.contains("prompt_tokens_details"), "usage details")
    } catch {
        check(false, "parity checks threw \(error)")
    }

    // Auth
    do {
        let handler = testHandler(apiKey: "secret")
        try handler.checkAuth(authorization: nil)
        check(false, "auth should throw")
    } catch let adapter as AdapterError {
        check(adapter.code == "invalid_api_key", "auth code")
    } catch {
        check(false, "wrong auth error")
    }

    // response_format
    do {
        let handler = testHandler()
        let request = try decodeRequest("""
        {"model": "apple.fm.system", "messages": [{"role": "user", "content": "hi"}],
         "response_format": {"type": "json_object"}}
        """)
        _ = try handler.prepare(request)
        check(false, "json_object should throw")
    } catch let adapter as AdapterError {
        check(adapter.message.contains("json_schema"), "json_object message")
    } catch {
        check(false, "wrong format error")
    }
}

func decodeResponses(_ json: String) throws -> ResponsesRequest {
    try JSONDecoder().decode(ResponsesRequest.self, from: Data(json.utf8))
}

func runResponsesChecks() async {
    // String input -> message output with mapped usage and resp_/msg_ ids.
    do {
        let handler = testHandler()
        let request = try decodeResponses("{\"model\": \"apple.fm.system\", \"input\": \"hello\"}")
        let response = try await handler.completeResponses(request: request, model: request.model)
        check(response.id.hasPrefix("resp_"), "responses id prefix")
        check(response.status == "completed", "responses status")
        if case .message(let id, let text) = response.output.first {
            check(id.hasPrefix("msg_"), "responses msg id prefix")
            check(text == "hello back", "responses text")
        } else {
            check(false, "responses message output")
        }
        check((response.usage?.totalTokens ?? 0) > 0, "responses usage")
    } catch {
        check(false, "responses basic threw \(error)")
    }

    // Tools + envelope mock -> function_call items, no message item.
    do {
        let handler = testHandler(responseText: "{\"content\": null, \"tool_calls\": [{\"name\": \"noop\", \"arguments\": {\"x\": 1}}]}")
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": "do",
         "tools": [{"type": "function", "name": "noop", "description": "noop"}]}
        """)
        let response = try await handler.completeResponses(request: request, model: request.model)
        check(response.output.count == 1, "responses single function call")
        if case .functionCall(let id, let callId, let name, let args) = response.output.first {
            check(id.hasPrefix("fc_"), "responses fc id prefix")
            check(callId.hasPrefix("call_"), "responses call_id preserved")
            check(name == "noop", "responses fn name")
            check(args.contains("\"x\""), "responses fn args")
        } else {
            check(false, "responses function_call output")
        }
    } catch {
        check(false, "responses tools threw \(error)")
    }

    // History merge preserves call IDs across assistant/output turns.
    do {
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": [
          {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do"}]},
          {"type": "function_call", "call_id": "call_9", "name": "noop", "arguments": "{}"},
          {"type": "function_call_output", "call_id": "call_9", "output": "ok"},
          {"type": "message", "role": "user", "content": "continue"}]}
        """)
        let chat = try request.asChatRequest()
        let assistant = chat.messages.first(where: { $0.role == "assistant" })
        check(assistant?.toolCalls?.first?.id == "call_9", "responses merge call id")
        let tool = chat.messages.first(where: { $0.role == "tool" })
        check(tool?.toolCallID == "call_9", "responses merge output id")
        check(tool?.content.map { (try? messageContentToText($0)) ?? "" } == "ok", "responses merge output text")
    } catch {
        check(false, "responses merge threw \(error)")
    }

    // Image blocks rejected like chat parts.
    do {
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": [
          {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}
        """)
        _ = try request.asChatRequest()
        check(false, "responses image should throw")
    } catch let adapter as AdapterError {
        check(adapter.code == "image_not_supported", "responses image code")
    } catch {
        check(false, "responses image wrong error")
    }

    // Unknown model + tool_choice none.
    do {
        let handler = testHandler()
        let request = try decodeResponses("{\"model\": \"wrong\", \"input\": \"hi\"}")
        _ = try await handler.completeResponses(request: request, model: request.model)
        check(false, "responses unknown model should throw")
    } catch let adapter as AdapterError {
        check(adapter.code == "model_not_found", "responses model_not_found")
    } catch {
        check(false, "responses model wrong error")
    }
    do {
        let handler = testHandler(responseText: "{\"content\": \"plain\", \"tool_calls\": []}")
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": "hi",
         "tools": [{"type": "function", "name": "noop"}],
         "tool_choice": "none"}
        """)
        let response = try await handler.completeResponses(request: request, model: request.model)
        check(response.output.count == 1, "responses tool_choice none single output")
    } catch {
        check(false, "responses tool_choice none threw \(error)")
    }

    // instructions array form + sloppy tools/blocks tolerated.
    do {
        let request = try decodeResponses("""
        {"model": "apple.fm.system", "input": "hi",
         "instructions": [{"type": "input_text", "text": "Be terse."}],
         "tools": [{"name": "noop"}]}
        """)
        check(request.instructions == "Be terse.", "responses instructions blocks")
        check(request.tools?.first?.type == "function", "responses tool type default")
    } catch {
        check(false, "responses tolerance threw \(error)")
    }

    // Stream event sequence: created -> deltas -> completed, shared id, no [DONE].
    do {
        let handler = testHandler()
        let request = try decodeResponses("{\"model\": \"apple.fm.system\", \"input\": \"hi\"}")
        let events = try await handler.responsesEvents(request: request, model: request.model)
        let payload = events.joined()
        check(payload.contains("\"type\":\"response.created\""), "responses stream created")
        check(payload.contains("\"type\":\"response.output_text.delta\""), "responses stream delta")
        check(payload.contains("\"type\":\"response.completed\""), "responses stream completed")
        check(!payload.contains("[DONE]"), "responses stream no DONE")
        // Created and completed share the response id.
        let ids = events.compactMap { line -> String? in
            guard let start = line.range(of: "\"id\":\"resp_") else { return nil }
            let rest = line[start.upperBound...]
            guard let end = rest.firstIndex(of: "\"") else { return nil }
            return "resp_" + String(rest[..<end])
        }
        check(Set(ids).count == 1 && !ids.isEmpty, "responses stream shared id")
    } catch {
        check(false, "responses stream threw \(error)")
    }
}

let semaphore = DispatchSemaphore(value: 0)
Task {
    await runAll()
    await runResponsesChecks()
    if failures == 0 {
        print("verify: all checks passed (incl. responses)")
    } else {
        print("verify: \(failures) failure(s)")
        exit(1)
    }
    semaphore.signal()
}
semaphore.wait()
