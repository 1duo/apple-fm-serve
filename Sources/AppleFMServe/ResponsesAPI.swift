import Foundation

// MARK: - OpenAI Responses API (for Codex; Chat Completions was removed Feb 2026)
//
// Stateless translation over the existing Chat Completions pipeline:
// Responses input items -> [ChatMessage] -> prepare/generate -> Responses output.
// Streaming is buffered (same as chat): all events are produced after the AFM
// run completes, so failures surface as HTTP errors instead of broken streams.

public func randomHex(_ bytes: Int = 12) -> String {
    (0..<bytes).map { _ in String(format: "%02x", UInt8.random(in: 0...255)) }.joined()
}

public func newResponseID() -> String { "resp_" + randomHex() }
public func newMessageItemID() -> String { "msg_" + randomHex() }
public func newFunctionCallItemID() -> String { "fc_" + randomHex() }

// MARK: - Request types (tolerant decoding; unknown fields ignored)

public enum ResponsesInput: Decodable, Sendable {
    case text(String)
    case items([ResponsesItem])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .items(try container.decode([ResponsesItem].self))
        }
    }
}

public enum ResponsesMessageContent: Decodable, Sendable {
    case text(String)
    case blocks([ResponsesContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .blocks(try container.decode([ResponsesContentBlock].self))
        }
    }
}

public struct ResponsesContentBlock: Decodable, Sendable {
    public var type: String
    public var text: String?
    public var imageURL: String?

    public enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        text = try? container.decode(String.self, forKey: .text)
        if let url = try? container.decode(String.self, forKey: .imageURL) {
            imageURL = url
        } else {
            imageURL = nil
        }
    }
}

public struct ResponsesItem: Decodable, Sendable {
    public var type: String
    // message fields
    public var id: String?
    public var role: String?
    public var content: ResponsesMessageContent?
    public var status: String?
    // function_call fields
    public var callId: String?
    public var name: String?
    public var arguments: String?
    // function_call_output fields
    public var output: AnyCodable?

    public enum CodingKeys: String, CodingKey {
        case type
        case id
        case role
        case content
        case status
        case callId = "call_id"
        case name
        case arguments
        case output
    }
}

public struct ResponsesTool: Decodable, Sendable {
    public var type: String
    public var name: String?
    public var description: String?
    public var parameters: [String: AnyCodable]?

    public enum CodingKeys: String, CodingKey {
        case type
        case name
        case description
        case parameters
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Only function tools are emulated; unknown kinds are filtered by
        // asChatRequest, so default a missing kind instead of failing.
        type = (try? container.decode(String.self, forKey: .type)) ?? "function"
        name = try container.decodeIfPresent(String.self, forKey: .name)
        description = try? container.decode(String.self, forKey: .description)
        parameters = try? container.decode([String: AnyCodable].self, forKey: .parameters)
    }
}

public struct ResponsesRequest: Decodable, Sendable {
    public var model: String
    public var input: ResponsesInput?
    public var instructions: String?
    public var tools: [ResponsesTool]?
    public var toolChoice: AnyCodable?
    public var stream: Bool
    public var temperature: Double?
    public var topP: Double?
    public var maxOutputTokens: Int?
    public var maxTokens: Int?

    public enum CodingKeys: String, CodingKey {
        case model
        case input
        case instructions
        case tools
        case toolChoice = "tool_choice"
        case stream
        case temperature
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
        case maxTokens = "max_tokens"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        input = try container.decodeIfPresent(ResponsesInput.self, forKey: .input)
        if container.contains(.instructions) {
            if try container.decodeNil(forKey: .instructions) {
                instructions = nil
            } else if let text = try? container.decode(String.self, forKey: .instructions) {
                instructions = text
            } else if let blocks = try? container.decode([ResponsesContentBlock].self, forKey: .instructions) {
                instructions = try extractResponsesText(.blocks(blocks))
            } else {
                instructions = try container.decodeIfPresent(String.self, forKey: .instructions)
            }
        } else {
            instructions = nil
        }
        tools = try container.decodeIfPresent([ResponsesTool].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent(AnyCodable.self, forKey: .toolChoice)
        stream = (try container.decodeIfPresent(Bool.self, forKey: .stream)) ?? false
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        // Note: previous_response_id/store/background are intentionally not
        // decoded: this server is stateless and answers from the full `input`
        // on every turn (codex resends complete context).
    }
}

// MARK: - Input translation

/// Extracts text from Responses message content. Image blocks are rejected,
/// matching the Chat Completions adapter (text-only shim over AFM sessions).
public func extractResponsesText(_ content: ResponsesMessageContent?) throws -> String {
    guard let content else { return "" }
    switch content {
    case .text(let text):
        return text
    case .blocks(let blocks):
        var texts: [String] = []
        for block in blocks {
            switch block.type {
            case "input_text", "output_text":
                if let text = block.text, !text.isEmpty { texts.append(text) }
            case "refusal":
                continue
            case "input_image":
                throw AdapterError(
                    "Image content is not supported by this adapter",
                    statusCode: 400,
                    param: "input",
                    code: "image_not_supported"
                )
            default:
                continue
            }
        }
        return texts.joined(separator: "\n")
    }
}

/// Renders a function_call_output value (string or content-block array) to text.
public func functionOutputText(from output: AnyCodable?) -> String {
    guard let output else { return "" }
    if let string = output.asString { return string }
    if let array = output.value as? [Any] {
        let texts = array.compactMap { ($0 as? [String: Any])?["text"] as? String }
        if !texts.isEmpty { return texts.joined(separator: "\n") }
        if let data = try? JSONSerialization.data(withJSONObject: array, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            return text
        }
        return ""
    }
    if let dict = output.value as? [String: Any],
       let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]),
       let text = String(data: data, encoding: .utf8) {
        return text
    }
    return ""
}

/// Folds Responses input items into chat messages. Assistant `function_call`
/// items replayed in history merge into the preceding assistant message (or a
/// new one), preserving call IDs so outputs stay matched.
public func responsesInputMessages(_ input: ResponsesInput?, instructions: String?) throws -> [ChatMessage] {
    var messages: [ChatMessage] = []
    if let instructions, !instructions.isEmpty {
        messages.append(ChatMessage(role: "system", content: .text(instructions)))
    }
    guard let input else { return messages }
    switch input {
    case .text(let text):
        if !text.isEmpty {
            messages.append(ChatMessage(role: "user", content: .text(text)))
        }
    case .items(let items):
        var lastAssistantIndex: Int?
        for item in items {
            switch item.type {
            case "message":
                let role = item.role ?? "user"
                let text = try extractResponsesText(item.content)
                messages.append(ChatMessage(role: role, content: .text(text)))
                lastAssistantIndex = (role == "assistant") ? messages.count - 1 : nil
            case "function_call":
                guard let name = item.name else { continue }
                let call = ToolCall(
                    id: item.callId ?? newToolCallID(),
                    function: FunctionCall(name: name, arguments: item.arguments ?? "{}")
                )
                if let index = lastAssistantIndex {
                    var merged = messages[index]
                    merged.toolCalls = (merged.toolCalls ?? []) + [call]
                    messages[index] = merged
                } else {
                    messages.append(ChatMessage(role: "assistant", content: .text(""), toolCalls: [call]))
                    lastAssistantIndex = messages.count - 1
                }
            case "function_call_output":
                messages.append(ChatMessage(
                    role: "tool",
                    content: .text(functionOutputText(from: item.output)),
                    toolCallID: item.callId
                ))
                lastAssistantIndex = nil
            default:
                // reasoning items and unknown future types carry no chat content.
                continue
            }
        }
    }
    return messages
}

extension ResponsesRequest {
    /// Funnels through the Chat Completions request shape so validation,
    /// prompt building, and tool emulation stay in one place.
    public func asChatRequest() throws -> ChatCompletionsRequest {
        let messages = try responsesInputMessages(input, instructions: instructions)
        let mappedTools = tools?.filter { $0.type == "function" }.compactMap { tool -> ToolDefinition? in
            guard let name = tool.name else { return nil }
            return ToolDefinition(
                type: "function",
                function: FunctionDefinition(
                    name: name,
                    description: tool.description,
                    parameters: tool.parameters
                )
            )
        }
        return ChatCompletionsRequest(
            model: model,
            messages: messages,
            stream: stream,
            temperature: temperature,
            topP: topP,
            maxTokens: maxOutputTokens ?? maxTokens,
            tools: (mappedTools?.isEmpty == true) ? nil : mappedTools,
            toolChoice: toolChoice
        )
    }
}

// MARK: - Response types

public struct ResponsesUsage: Codable, Sendable {
    public var inputTokens: Int
    public var outputTokens: Int
    public var totalTokens: Int
    public var inputTokensDetails = ResponsesInputTokensDetails()
    public var outputTokensDetails = ResponsesOutputTokensDetails()

    public enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case totalTokens = "total_tokens"
        case inputTokensDetails = "input_tokens_details"
        case outputTokensDetails = "output_tokens_details"
    }
}

public struct ResponsesInputTokensDetails: Codable, Sendable {
    public var cachedTokens: Int = 0

    public enum CodingKeys: String, CodingKey {
        case cachedTokens = "cached_tokens"
    }
}

public struct ResponsesOutputTokensDetails: Codable, Sendable {
    public var reasoningTokens: Int = 0

    public enum CodingKeys: String, CodingKey {
        case reasoningTokens = "reasoning_tokens"
    }
}

public struct ResponsesTextPart: Codable, Sendable {
    public var type: String = "output_text"
    public var text: String
    public var annotations: [AnyCodable] = []
}

public enum ResponsesOutputItem: Encodable, Sendable {
    case message(id: String, text: String)
    case functionCall(id: String, callId: String, name: String, arguments: String)

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicKey.self)
        switch self {
        case .message(let id, let text):
            try container.encode("message", forKey: DynamicKey("type"))
            try container.encode(id, forKey: DynamicKey("id"))
            try container.encode("completed", forKey: DynamicKey("status"))
            try container.encode("assistant", forKey: DynamicKey("role"))
            try container.encode([ResponsesTextPart(text: text)], forKey: DynamicKey("content"))
        case .functionCall(let id, let callId, let name, let arguments):
            try container.encode("function_call", forKey: DynamicKey("type"))
            try container.encode(id, forKey: DynamicKey("id"))
            try container.encode("completed", forKey: DynamicKey("status"))
            try container.encode(callId, forKey: DynamicKey("call_id"))
            try container.encode(name, forKey: DynamicKey("name"))
            try container.encode(arguments, forKey: DynamicKey("arguments"))
        }
    }
}

public struct DynamicKey: CodingKey {
    public var stringValue: String
    public var intValue: Int? { nil }
    public init(_ string: String) { stringValue = string }
    public init?(stringValue: String) { self.stringValue = stringValue }
    public init?(intValue: Int) { nil }
}

public struct ResponsesResponse: Encodable, Sendable {
    public var id: String
    public var object: String = "response"
    public var createdAt: Int
    public var model: String
    public var status: String = "completed"
    public var output: [ResponsesOutputItem]
    public var usage: ResponsesUsage?
    public var parallelToolCalls: Bool = true

    public enum CodingKeys: String, CodingKey {
        case id
        case object
        case createdAt = "created_at"
        case model
        case status
        case output
        case usage
        case parallelToolCalls = "parallel_tool_calls"
    }
}

// MARK: - Handler

extension ChatHandler {
    /// Builds the Responses output object from chat content + tool calls.
    public func responsesObject(
        model: String,
        created: Int,
        content: String?,
        toolCalls: [ToolCall],
        usage: UsageOut?,
        id: String? = nil
    ) -> ResponsesResponse {
        var output: [ResponsesOutputItem] = []
        if let content {
            output.append(.message(id: newMessageItemID(), text: content))
        }
        for call in toolCalls {
            output.append(.functionCall(
                id: newFunctionCallItemID(),
                callId: call.id,
                name: call.function.name,
                arguments: call.function.arguments
            ))
        }
        let responsesUsage = usage.map {
            ResponsesUsage(
                inputTokens: $0.promptTokens,
                outputTokens: $0.completionTokens,
                totalTokens: $0.totalTokens
            )
        }
        return ResponsesResponse(
            id: id ?? newResponseID(),
            createdAt: created,
            model: model,
            output: output,
            usage: responsesUsage
        )
    }

    public func completeResponses(request: ResponsesRequest, model: String) async throws -> ResponsesResponse {
        let chat = try request.asChatRequest()
        let chatResponse = try await complete(request: chat, model: model)
        let message = chatResponse.choices.first?.message
        return responsesObject(
            model: model,
            created: chatResponse.created,
            content: message?.content,
            toolCalls: message?.toolCalls ?? [],
            usage: chatResponse.usage
        )
    }

    /// Produces Responses SSE event payloads (`data: {...}`). Unlike chat SSE,
    /// there is no trailing [DONE] — the stream ends at `response.completed`.
    /// All AFM work happens before the first event, so throws here mean the
    /// HTTP layer can still return a proper error status.
    public func responsesEvents(request: ResponsesRequest, model: String) async throws -> [String] {
        let chat = try request.asChatRequest()
        let run = try await runStream(request: chat, model: model)
        let created = run.created
        let responseID = newResponseID()
        var sequence = 0
        var events: [String] = []

        func skeleton(status: String, output: [[String: Any]]) -> [String: Any] {
            let response: [String: Any] = [
                "id": responseID,
                "object": "response",
                "created_at": created,
                "model": model,
                "status": status,
                "output": output,
                "parallel_tool_calls": true,
            ]
            return ["response": response]
        }

        func emit(_ type: String, _ fields: [String: Any] = [:]) throws {
            sequence += 1
            var payload = fields
            payload["type"] = type
            payload["sequence_number"] = sequence
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            events.append("data: " + String(data: data, encoding: .utf8)! + "\n\n")
        }

        func messageItem(id: String, status: String, text: String?) -> [String: Any] {
            var item: [String: Any] = [
                "id": id,
                "type": "message",
                "status": status,
                "role": "assistant",
                "content": [],
            ]
            if let text {
                item["content"] = [["type": "output_text", "text": text, "annotations": []]]
            }
            return item
        }

        func functionCallItem(id: String, callId: String, name: String, arguments: String) -> [String: Any] {
            [
                "id": id,
                "type": "function_call",
                "status": "completed",
                "call_id": callId,
                "name": name,
                "arguments": arguments,
            ]
        }

        try emit("response.created", skeleton(status: "in_progress", output: []))

        var outputIndex = 0
        if !run.prepared.emulateTools {
            // Plain text: replay snapshot deltas as output_text events.
            let messageID = newMessageItemID()
            try emit("response.output_item.added", [
                "output_index": outputIndex,
                "item": messageItem(id: messageID, status: "in_progress", text: nil),
            ])
            try emit("response.content_part.added", [
                "output_index": outputIndex,
                "item_id": messageID,
                "content_index": 0,
                "part": ["type": "output_text", "text": "", "annotations": []],
            ])
            var differ = SnapshotDiffer()
            for snapshot in run.snapshots {
                if let delta = differ.delta(for: snapshot.text) {
                    try emit("response.output_text.delta", [
                        "item_id": messageID,
                        "output_index": outputIndex,
                        "content_index": 0,
                        "delta": delta,
                    ])
                }
            }
            let (finalText, _) = applyStopSequences(run.accumulated, stop: run.prepared.stop)
            try emit("response.output_text.done", [
                "item_id": messageID,
                "output_index": outputIndex,
                "content_index": 0,
                "text": finalText,
            ])
            try emit("response.content_part.done", [
                "item_id": messageID,
                "output_index": outputIndex,
                "content_index": 0,
                "part": ["type": "output_text", "text": finalText, "annotations": []],
            ])
            try emit("response.output_item.done", [
                "output_index": outputIndex,
                "item": messageItem(id: messageID, status: "completed", text: finalText),
            ])
            outputIndex += 1
        } else if let content = run.parsed.content, !content.isEmpty {
            // Envelope mode with user-facing text: single buffered delta.
            let messageID = newMessageItemID()
            let (finalText, _) = applyStopSequences(content, stop: run.prepared.stop)
            try emit("response.output_item.added", [
                "output_index": outputIndex,
                "item": messageItem(id: messageID, status: "in_progress", text: nil),
            ])
            try emit("response.content_part.added", [
                "output_index": outputIndex,
                "item_id": messageID,
                "content_index": 0,
                "part": ["type": "output_text", "text": "", "annotations": []],
            ])
            try emit("response.output_text.delta", [
                "item_id": messageID,
                "output_index": outputIndex,
                "content_index": 0,
                "delta": finalText,
            ])
            try emit("response.output_text.done", [
                "item_id": messageID,
                "output_index": outputIndex,
                "content_index": 0,
                "text": finalText,
            ])
            try emit("response.content_part.done", [
                "item_id": messageID,
                "output_index": outputIndex,
                "content_index": 0,
                "part": ["type": "output_text", "text": finalText, "annotations": []],
            ])
            try emit("response.output_item.done", [
                "output_index": outputIndex,
                "item": messageItem(id: messageID, status: "completed", text: finalText),
            ])
            outputIndex += 1
        }

        for call in run.parsed.toolCalls {
            let itemID = newFunctionCallItemID()
            try emit("response.output_item.added", [
                "output_index": outputIndex,
                "item": [
                    "id": itemID,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": "",
                ] as [String: Any],
            ])
            try emit("response.function_call_arguments.delta", [
                "item_id": itemID,
                "output_index": outputIndex,
                "delta": call.function.arguments,
            ])
            try emit("response.function_call_arguments.done", [
                "item_id": itemID,
                "output_index": outputIndex,
                "arguments": call.function.arguments,
            ])
            try emit("response.output_item.done", [
                "output_index": outputIndex,
                "item": functionCallItem(
                    id: itemID, callId: call.id,
                    name: call.function.name, arguments: call.function.arguments
                ),
            ])
            outputIndex += 1
        }

        let completed = responsesObject(
            model: model,
            created: created,
            content: run.prepared.emulateTools ? run.parsed.content : run.accumulated,
            toolCalls: run.parsed.toolCalls,
            usage: run.usage,
            id: responseID
        )
        // Reuse the non-stream object so both shapes stay identical.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let completedData = try encoder.encode(completed)
        let completedObj = try JSONSerialization.jsonObject(with: completedData) as! [String: Any]
        try emit("response.completed", ["response": completedObj])
        return events
    }
}
