import Foundation

/// Orchestrates an OpenAI chat-completions request against an LLMProvider.
/// Pure logic (no sockets) so it can be unit-tested with MockProvider.
public struct ChatHandler: Sendable {
    public var config: Config
    public var provider: any LLMProvider
    public var hybrid: HybridProvider?

    public init(config: Config, provider: any LLMProvider, hybrid: HybridProvider? = nil) {
        self.config = config
        self.provider = provider
        self.hybrid = hybrid
    }

    public func checkAuth(authorization: String?) throws {
        guard let apiKey = config.apiKey, !apiKey.isEmpty else { return }
        guard let authorization, authorization.hasPrefix("Bearer ") else {
            throw AdapterError.invalidAPIKey
        }
        let token = String(authorization.dropFirst("Bearer ".count)).trimmingCharacters(in: .whitespaces)
        guard !token.isEmpty, token == apiKey else {
            throw AdapterError.invalidAPIKey
        }
    }

    public func isKnownModel(_ model: String) -> Bool {
        if model == config.modelID { return true }
        if model == "system" || model == "apple.fm.system" { return true }
        if model == "pcc" || model == "apple.fm.pcc" { return true }
        return false
    }

    public func activeProvider(forModel model: String) async -> any LLMProvider {
        if let hybrid {
            return await hybrid.provider(forModel: model)
        }
        return provider
    }

    // MARK: - Request preparation

    public struct PreparedRequest: Sendable {
        var instructions: String?
        var prompt: String
        var temperature: Double?
        var topP: Double?
        var maxTokens: Int?
        var stop: StopSequences?
        var responseSchemaJSON: String?
        var emulateTools: Bool
        var toolDefs: [ToolDefinition]
        var validToolNames: Set<String>
        var toolChoice: AnyCodable?
    }

    public func prepare(_ request: ChatCompletionsRequest) throws -> PreparedRequest {
        guard isKnownModel(request.model) else {
            throw AdapterError.modelNotFound(request.model)
        }
        if let temperature = request.temperature, temperature < 0 || temperature > 2 {
            throw AdapterError("temperature must be between 0 and 2", statusCode: 400, param: "temperature", code: "invalid_temperature")
        }
        if let topP = request.topP, topP < 0 || topP > 1 {
            throw AdapterError("top_p must be between 0 and 1", statusCode: 400, param: "top_p", code: "invalid_top_p")
        }
        if let maxTokens = request.effectiveMaxTokens, maxTokens < 1 {
            let param = request.maxTokens != nil ? "max_tokens" : "max_completion_tokens"
            throw AdapterError("max tokens must be at least 1", statusCode: 400, param: param, code: "invalid_max_tokens")
        }

        let built = try buildPrompt(messages: request.messages)

        // response_format handling (mirrors `fm serve`: json_object unsupported).
        var responseSchemaJSON: String?
        if let format = request.responseFormat {
            switch format.type {
            case "text":
                break
            case "json_object":
                throw AdapterError(
                    "response_format type 'json_object' is not supported. Use 'json_schema' instead.",
                    statusCode: 400,
                    param: "response_format",
                    code: "invalid_response_format"
                )
            case "json_schema":
                guard let payload = format.jsonSchema,
                      let schema = payload.schema else {
                    throw AdapterError(
                        "response_format.json_schema.schema is required for type=json_schema",
                        statusCode: 400,
                        param: "response_format",
                        code: "invalid_response_format"
                    )
                }
                let object = schema.mapValues { $0.value }
                guard JSONSerialization.isValidJSONObject(object),
                      let data = try? JSONSerialization.data(withJSONObject: object),
                      let text = String(data: data, encoding: .utf8) else {
                    throw AdapterError(
                        "Invalid response_format schema",
                        statusCode: 400,
                        param: "response_format",
                        code: "invalid_response_format"
                    )
                }
                responseSchemaJSON = text
            default:
                throw AdapterError(
                    "Unsupported response_format type.",
                    statusCode: 400,
                    param: "response_format",
                    code: "invalid_response_format"
                )
            }
        }

        // Tool emulation: fold tool definitions into instructions; the model's
        // structured reply is parsed back into OpenAI tool_calls.
        let emulateTools = shouldEmulateTools(request: request)
        if emulateTools, responseSchemaJSON != nil {
            // Contradictory directives: the tool envelope already constrains
            // the shape, so response_format is set aside (and logged).
            logLine("WARN tools and response_format combined; response_format ignored")
            responseSchemaJSON = nil
        }
        var instructions = built.instructions
        var toolDefs: [ToolDefinition] = []
        var validNames: Set<String> = []
        if emulateTools, let tools = request.tools {
            toolDefs = tools
            validNames = Set(tools.map { $0.function.name })
            let toolText = toolInstructions(for: tools, toolChoice: request.toolChoice)
            if let existing = instructions, !existing.isEmpty {
                instructions = existing + "\n\n" + toolText
            } else {
                instructions = toolText
            }
        }

        return PreparedRequest(
            instructions: instructions,
            prompt: built.prompt,
            temperature: request.temperature,
            topP: request.topP,
            maxTokens: request.effectiveMaxTokens,
            stop: request.stop,
            responseSchemaJSON: responseSchemaJSON,
            emulateTools: emulateTools,
            toolDefs: toolDefs,
            validToolNames: validNames,
            toolChoice: request.toolChoice
        )
    }

    /// Stricter instructions for the single retry when tools are required but
    /// the model answered in prose (common with small on-device models).
    public func retryInstructions(_ instructions: String?) -> String? {
        let suffix = "CRITICAL: Reply with ONLY a raw JSON object, no markdown fences, no prose: {\"content\": <string|null>, \"tool_calls\": [{\"name\": ..., \"arguments\": {...}}]}."
        if let instructions, !instructions.isEmpty {
            return instructions + "\n\n" + suffix
        }
        return suffix
    }

    // MARK: - Non-streaming

    public func complete(request: ChatCompletionsRequest, model: String) async throws -> ChatCompletionsResponse {
        let prepared = try prepare(request)
        let active = await activeProvider(forModel: request.model)
        let (available, reason) = await active.isAvailable()
        guard available else { throw AdapterError.providerUnavailable(reason) }

        let generation = GenerationRequest(
            instructions: prepared.instructions,
            prompt: prepared.prompt,
            temperature: prepared.temperature,
            topP: prepared.topP,
            maxTokens: prepared.maxTokens,
            responseSchemaJSON: prepared.emulateTools ? nil : prepared.responseSchemaJSON
        )
        // Streaming + response_format is unsupported (matches previous adapter).
        if request.stream, prepared.responseSchemaJSON != nil {
            throw AdapterError(
                "Streaming with response_format is not supported",
                statusCode: 400,
                param: "response_format",
                code: "response_format_stream_unsupported"
            )
        }

        var result = try await withTimeout(seconds: config.requestTimeoutSeconds) {
            try await active.generate(generation)
        }

        // Post-process: tools, stop sequences, usage.
        if prepared.emulateTools {
            var parsed = parseToolResponse(result.text, validNames: prepared.validToolNames)
            if parsed.toolCalls.isEmpty, toolChoiceRequiresTools(prepared.toolChoice) {
                // Tools demanded but prose returned: one stricter retry.
                logLine("WARN tool envelope missing, retrying with stricter instructions")
                let retry = GenerationRequest(
                    instructions: retryInstructions(prepared.instructions),
                    prompt: generation.prompt,
                    temperature: generation.temperature,
                    topP: generation.topP,
                    maxTokens: generation.maxTokens,
                    responseSchemaJSON: generation.responseSchemaJSON
                )
                result = try await withTimeout(seconds: config.requestTimeoutSeconds) {
                    try await active.generate(retry)
                }
                parsed = parseToolResponse(result.text, validNames: prepared.validToolNames)
            }
            if parsed.recovered {
                logLine("WARN recovered \(parsed.toolCalls.count) tool call(s) from malformed envelope")
            }
            if parsed.toolCalls.isEmpty, !parsed.recovered,
               envelopeStatus(result.text, validNames: prepared.validToolNames) == .unusable {
                logLine("WARN tool calls unusable; raw: \(bodySnippet(result.text, maxChars: 500))")
            }
            let (finalText, _) = applyStopSequences(parsed.content ?? result.text, stop: prepared.stop)
            // Stop-sequence truncation maps to "stop": AFM exposes no separate
            // length signal on this path (max_tokens enforcement is provider-side).
            let finishReason = !parsed.toolCalls.isEmpty ? "tool_calls" : "stop"
            let message = ChatCompletionMessageOut(
                content: parsed.toolCalls.isEmpty ? finalText : parsed.content.map { applyStopSequences($0, stop: prepared.stop).text },
                toolCalls: parsed.toolCalls.isEmpty ? nil : parsed.toolCalls
            )
            return ChatCompletionsResponse(
                id: newCompletionID(),
                created: Int(Date().timeIntervalSince1970),
                model: model,
                choices: [ChatCompletionChoiceOut(index: 0, message: message, finishReason: finishReason)],
                usage: usageFor(prompt: prepared.prompt, completion: result.text, fallback: result)
            )
        }

        let (finalText, _) = applyStopSequences(result.text, stop: prepared.stop)
        let message = ChatCompletionMessageOut(content: finalText, toolCalls: nil)
        return ChatCompletionsResponse(
            id: newCompletionID(),
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [ChatCompletionChoiceOut(index: 0, message: message, finishReason: "stop")],
            usage: usageFor(prompt: prepared.prompt, completion: result.text, fallback: result)
        )
    }

    // MARK: - Streaming core (shared by Chat Completions SSE and Responses events)

    /// Buffered result of a streaming run: snapshots are collected under the
    /// request timeout, then parsed once (tool envelope must not leak raw).
    public struct StreamRun: Sendable {
        public var id: String
        public var created: Int
        public var model: String
        public var prepared: PreparedRequest
        public var snapshots: [StreamSnapshot]
        public var accumulated: String
        public var parsed: ParsedToolResponse
        public var usage: UsageOut?
        public var includeUsage: Bool
    }

    /// Runs generation to completion via the streaming provider path and
    /// returns the structured result. Throws before producing output for
    /// validation/availability/timeout failures.
    public func runStream(request: ChatCompletionsRequest, model: String) async throws -> StreamRun {
        let prepared = try prepare(request)
        if prepared.responseSchemaJSON != nil {
            throw AdapterError(
                "Streaming with response_format is not supported",
                statusCode: 400,
                param: "response_format",
                code: "response_format_stream_unsupported"
            )
        }
        let active = await activeProvider(forModel: request.model)
        let (available, reason) = await active.isAvailable()
        guard available else { throw AdapterError.providerUnavailable(reason) }

        let generation = GenerationRequest(
            instructions: prepared.instructions,
            prompt: prepared.prompt,
            temperature: prepared.temperature,
            topP: prepared.topP,
            maxTokens: prepared.maxTokens,
            responseSchemaJSON: nil
        )

        // Timeout applies to the whole stream.
        var snapshots = try await withTimeout(seconds: config.requestTimeoutSeconds) {
            var collected: [StreamSnapshot] = []
            for try await snapshot in active.streamGenerate(generation) {
                collected.append(snapshot)
            }
            return collected
        }
        var accumulated = snapshots.last?.text ?? ""
        var parsed = prepared.emulateTools
            ? parseToolResponse(accumulated, validNames: prepared.validToolNames)
            : ParsedToolResponse(content: accumulated, toolCalls: [])
        if prepared.emulateTools, parsed.toolCalls.isEmpty, toolChoiceRequiresTools(prepared.toolChoice) {
            logLine("WARN tool envelope missing, retrying with stricter instructions")
            let retry = GenerationRequest(
                instructions: retryInstructions(prepared.instructions),
                prompt: generation.prompt,
                temperature: generation.temperature,
                topP: generation.topP,
                maxTokens: generation.maxTokens,
                responseSchemaJSON: nil
            )
            var recollected: [StreamSnapshot] = []
            for try await snapshot in active.streamGenerate(retry) {
                recollected.append(snapshot)
            }
            snapshots = recollected
            accumulated = snapshots.last?.text ?? ""
            parsed = parseToolResponse(accumulated, validNames: prepared.validToolNames)
        }
        if parsed.recovered {
            logLine("WARN recovered \(parsed.toolCalls.count) tool call(s) from malformed envelope")
        }
        if parsed.toolCalls.isEmpty, !parsed.recovered,
           envelopeStatus(accumulated, validNames: prepared.validToolNames) == .unusable {
            logLine("WARN tool calls unusable; raw: \(bodySnippet(accumulated, maxChars: 500))")
        }
        var usage: UsageOut?
        if let last = snapshots.last {
            usage = usageFor(
                prompt: prepared.prompt, completion: last.text,
                promptTokens: last.promptTokens, completionTokens: last.completionTokens
            )
        }
        return StreamRun(
            id: newCompletionID(),
            created: Int(Date().timeIntervalSince1970),
            model: model,
            prepared: prepared,
            snapshots: snapshots,
            accumulated: accumulated,
            parsed: parsed,
            usage: usage,
            includeUsage: request.streamOptions?.includeUsage == true
        )
    }

    // MARK: - Streaming (SSE payloads)

    /// Returns the sequence of SSE `data:` payload strings. Includes the
    /// OpenAI-standard trailing usage-only chunk (when `include_usage`) and
    /// final [DONE].
    public func completeStream(request: ChatCompletionsRequest, model: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let run = try await runStream(request: request, model: model)
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.sortedKeys]

                    func emit(_ chunk: ChatCompletionsChunk) throws -> String {
                        let data = try encoder.encode(chunk)
                        return "data: " + String(data: data, encoding: .utf8)! + "\n\n"
                    }

                    func usageChunk() throws -> String {
                        try emit(ChatCompletionsChunk(
                            id: run.id, created: run.created, model: run.model,
                            choices: [],
                            usage: run.usage
                        ))
                    }

                    // Role preamble.
                    continuation.yield(try emit(ChatCompletionsChunk(
                        id: run.id, created: run.created, model: run.model,
                        choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: "assistant", content: nil), finishReason: nil)]
                    )))

                    var differ = SnapshotDiffer()
                    for snapshot in run.snapshots {
                        if let delta = differ.delta(for: snapshot.text) {
                            // When emulating tools, don't leak the raw JSON envelope
                            // token-by-token; buffer and emit parsed at the end.
                            if !run.prepared.emulateTools {
                                continuation.yield(try emit(ChatCompletionsChunk(
                                    id: run.id, created: run.created, model: run.model,
                                    choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: nil, content: delta), finishReason: nil)]
                                )))
                            }
                        }
                    }

                    if run.prepared.emulateTools {
                        if !run.parsed.toolCalls.isEmpty {
                            continuation.yield(try emit(ChatCompletionsChunk(
                                id: run.id, created: run.created, model: run.model,
                                choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: nil, content: nil, toolCalls: run.parsed.toolCalls), finishReason: nil)]
                            )))
                            continuation.yield(try emit(ChatCompletionsChunk(
                                id: run.id, created: run.created, model: run.model,
                                choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: nil, content: nil), finishReason: "tool_calls")]
                            )))
                        } else {
                            let (finalText, _) = applyStopSequences(run.parsed.content ?? run.accumulated, stop: run.prepared.stop)
                            // Emit as a single delta (buffered) for envelope mode.
                            if !finalText.isEmpty {
                                continuation.yield(try emit(ChatCompletionsChunk(
                                    id: run.id, created: run.created, model: run.model,
                                    choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: nil, content: finalText), finishReason: nil)]
                                )))
                            }
                            continuation.yield(try emit(ChatCompletionsChunk(
                                id: run.id, created: run.created, model: run.model,
                                choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: nil, content: nil), finishReason: "stop")]
                            )))
                        }
                    } else {
                        let (_, truncated) = applyStopSequences(run.accumulated, stop: run.prepared.stop)
                        _ = truncated
                        continuation.yield(try emit(ChatCompletionsChunk(
                            id: run.id, created: run.created, model: run.model,
                            choices: [ChatCompletionChunkChoice(index: 0, delta: DeltaMessage(role: nil, content: nil), finishReason: "stop")]
                        )))
                    }
                    if run.includeUsage, let _ = run.usage {
                        continuation.yield(try usageChunk())
                    }
                    continuation.yield("data: [DONE]\n\n")
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Usage

    public func usageFor(prompt: String, completion: String, fallback result: GenerationResult) -> UsageOut {
        usageFor(prompt: prompt, completion: completion, promptTokens: result.promptTokens, completionTokens: result.completionTokens)
    }

    public func usageFor(prompt: String, completion: String, promptTokens: Int?, completionTokens: Int?) -> UsageOut {
        let promptCount = promptTokens ?? estimateTokens(prompt)
        let completionCount = completionTokens ?? estimateTokens(completion)
        return UsageOut(promptTokens: promptCount, completionTokens: completionCount, totalTokens: promptCount + completionCount)
    }
}

// MARK: - Helpers

public func newCompletionID() -> String {
    "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
}

/// Timeout helper using a throwing task group.
public func withTimeout<T: Sendable>(seconds: Double, operation: @escaping @Sendable () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw AdapterError.requestTimeout
        }
        guard let result = try await group.next() else {
            throw AdapterError.requestTimeout
        }
        group.cancelAll()
        return result
    }
}
