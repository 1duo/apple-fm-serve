import Foundation

// MARK: - OpenAI Chat Completions types (tolerant decoding for harness compat)

/// A single chat message. `content` may be a string, a list of parts, or null.
/// `tool_calls` / `tool_call_id` support the OpenAI agentic loop used by opencode.
public struct ChatMessage: Codable {
    public var role: String
    public var content: MessageContent?
    public var toolCalls: [ToolCall]?
    public var toolCallID: String?
    public var name: String?

    public enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
        case name
    }

    public init(role: String, content: MessageContent? = nil, toolCalls: [ToolCall]? = nil, toolCallID: String? = nil, name: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        // Content may be absent/null/string/parts. Be liberal.
        if container.contains(.content) {
            if try container.decodeNil(forKey: .content) {
                content = nil
            } else if let text = try? container.decode(String.self, forKey: .content) {
                content = .text(text)
            } else if let parts = try? container.decode([MessagePart].self, forKey: .content) {
                content = .parts(parts)
            } else {
                content = nil
            }
        } else {
            content = nil
        }
        toolCalls = try container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
        toolCallID = try container.decodeIfPresent(String.self, forKey: .toolCallID)
        name = try container.decodeIfPresent(String.self, forKey: .name)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        switch content {
        case .none:
            try container.encodeNil(forKey: .content)
        case .text(let text)?:
            try container.encode(text, forKey: .content)
        case .parts(let parts)?:
            try container.encode(parts, forKey: .content)
        }
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
        try container.encodeIfPresent(toolCallID, forKey: .toolCallID)
        try container.encodeIfPresent(name, forKey: .name)
    }
}

public enum MessageContent {
    case text(String)
    case parts([MessagePart])
}

public struct MessagePart: Codable {
    public var type: String
    public var text: String?
    public var imageURL: [String: String]?

    public enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
    }

    public init(type: String, text: String? = nil, imageURL: [String: String]? = nil) {
        self.type = type
        self.text = text
        self.imageURL = imageURL
    }

    /// Tolerant decode: `image_url` arrives as an object per spec but accept a
    /// bare string too, so image turns still reach the explicit 400 instead of
    /// being silently dropped.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        text = try? container.decode(String.self, forKey: .text)
        if let dict = try? container.decode([String: String].self, forKey: .imageURL) {
            imageURL = dict
        } else if let url = try? container.decode(String.self, forKey: .imageURL) {
            imageURL = ["url": url]
        } else {
            imageURL = nil
        }
    }
}

public struct FunctionCall: Codable, Sendable {
    public var name: String
    public var arguments: String

    public enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }

    /// Tolerant decode: history replays from third parties may omit arguments
    /// or carry them as an object instead of a JSON string.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        if let string = try? container.decode(String.self, forKey: .arguments) {
            arguments = string
        } else if let value = try? container.decode(AnyCodable.self, forKey: .arguments) {
            arguments = FunctionCall.argumentsString(from: value)
        } else {
            arguments = "{}"
        }
    }

    static func argumentsString(from value: AnyCodable) -> String {
        if let string = value.asString { return string }
        if JSONSerialization.isValidJSONObject(value.value),
           let data = try? JSONSerialization.data(withJSONObject: value.value, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            return text
        }
        return "{}"
    }
}

public struct ToolCall: Codable, Sendable {
    public var id: String
    public var type: String
    public var function: FunctionCall

    public enum CodingKeys: String, CodingKey {
        case id
        case type
        case function
    }

    public init(id: String, type: String = "function", function: FunctionCall) {
        self.id = id
        self.type = type
        self.function = function
    }

    /// Tolerant decode: default the kind, mint an id when absent, so replays
    /// with sparse tool_calls still fold into context instead of 400ing.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(String.self, forKey: .id)) ?? newToolCallID()
        type = (try? container.decode(String.self, forKey: .type)) ?? "function"
        function = try container.decode(FunctionCall.self, forKey: .function)
    }
}

public struct FunctionDefinition: Codable, Sendable {
    public var name: String
    public var description: String?
    public var parameters: [String: AnyCodable]?

    public enum CodingKeys: String, CodingKey {
        case name
        case description
        case parameters
    }

    public init(name: String, description: String? = nil, parameters: [String: AnyCodable]? = nil) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    /// Tolerant decode: only `name` is load-bearing (prompt rendering skips
    /// anything malformed rather than failing the turn).
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        description = try? container.decode(String.self, forKey: .description)
        parameters = try? container.decode([String: AnyCodable].self, forKey: .parameters)
    }
}

public struct ToolDefinition: Codable, Sendable {
    public var type: String
    public var function: FunctionDefinition

    public enum CodingKeys: String, CodingKey {
        case type
        case function
    }

    public init(type: String = "function", function: FunctionDefinition) {
        self.type = type
        self.function = function
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = (try? container.decode(String.self, forKey: .type)) ?? "function"
        function = try container.decode(FunctionDefinition.self, forKey: .function)
    }
}

public struct ResponseFormatJSONSchema: Codable {
    public var name: String?
    public var schema: [String: AnyCodable]?
    public var strict: Bool?
}

public struct ResponseFormat: Codable {
    public var type: String
    public var jsonSchema: ResponseFormatJSONSchema?

    public enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }
}

public struct StreamOptions: Codable {
    public var includeUsage: Bool?

    public enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

/// Chat completions request. Unknown fields are ignored so harness
/// extras (logprobs, n, presence_penalty, etc.) don't cause 422s.
public struct ChatCompletionsRequest: Decodable {
    public var model: String
    public var messages: [ChatMessage]
    public var stream: Bool
    public var streamOptions: StreamOptions?
    public var temperature: Double?
    public var topP: Double?
    public var maxTokens: Int?
    public var maxCompletionTokens: Int?
    public var stop: StopSequences?
    public var tools: [ToolDefinition]?
    public var toolChoice: AnyCodable?
    public var responseFormat: ResponseFormat?

    public enum CodingKeys: String, CodingKey {
        case model
        case messages
        case stream
        case streamOptions = "stream_options"
        case temperature
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case stop
        case tools
        case toolChoice = "tool_choice"
        case responseFormat = "response_format"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        messages = try container.decode([ChatMessage].self, forKey: .messages)
        stream = (try container.decodeIfPresent(Bool.self, forKey: .stream)) ?? false
        streamOptions = try container.decodeIfPresent(StreamOptions.self, forKey: .streamOptions)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        maxCompletionTokens = try container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        if let single = try? container.decode(String.self, forKey: .stop) {
            stop = .single(single)
        } else if let list = try? container.decode([String].self, forKey: .stop) {
            stop = .list(list)
        } else {
            stop = nil
        }
        tools = try container.decodeIfPresent([ToolDefinition].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent(AnyCodable.self, forKey: .toolChoice)
        responseFormat = try container.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
    }

    public var effectiveMaxTokens: Int? {
        maxTokens ?? maxCompletionTokens
    }

    public init(
        model: String,
        messages: [ChatMessage],
        stream: Bool = false,
        streamOptions: StreamOptions? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        maxTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        stop: StopSequences? = nil,
        tools: [ToolDefinition]? = nil,
        toolChoice: AnyCodable? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.model = model
        self.messages = messages
        self.stream = stream
        self.streamOptions = streamOptions
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.stop = stop
        self.tools = tools
        self.toolChoice = toolChoice
        self.responseFormat = responseFormat
    }
}

public enum StopSequences: Sendable {
    case single(String)
    case list([String])

    public var values: [String] {
        switch self {
        case .single(let string): return [string]
        case .list(let list): return list
        }
    }
}

// MARK: - Response types

public struct ChatCompletionMessageOut: Codable {
    public var role: String = "assistant"
    public var content: String?
    public var toolCalls: [ToolCall]?
    public var refusal: String?

    public enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case refusal
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encodeIfPresent(content, forKey: .content)
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
        // Match OpenAI/fm serve: always present, null when no refusal.
        if let refusal {
            try container.encode(refusal, forKey: .refusal)
        } else {
            try container.encodeNil(forKey: .refusal)
        }
    }
}

public struct ChatCompletionChoiceOut: Codable {
    public var index: Int
    public var message: ChatCompletionMessageOut
    public var finishReason: String?

    public enum CodingKeys: String, CodingKey {
        case index
        case message
        case finishReason = "finish_reason"
    }
}

public struct PromptTokensDetailsOut: Codable, Sendable {
    public var cachedTokens: Int = 0

    public enum CodingKeys: String, CodingKey {
        case cachedTokens = "cached_tokens"
    }
}

public struct CompletionTokensDetailsOut: Codable, Sendable {
    public var reasoningTokens: Int = 0

    public enum CodingKeys: String, CodingKey {
        case reasoningTokens = "reasoning_tokens"
    }
}

public struct UsageOut: Codable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int
    public var promptTokensDetails = PromptTokensDetailsOut()
    public var completionTokensDetails = CompletionTokensDetailsOut()

    public enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTokensDetails = "completion_tokens_details"
    }
}

public struct ChatCompletionsResponse: Codable {
    public var id: String
    public var object: String = "chat.completion"
    public var created: Int
    public var model: String
    public var choices: [ChatCompletionChoiceOut]
    public var usage: UsageOut?
}

public struct DeltaMessage: Codable {
    public var role: String?
    public var content: String?
    public var toolCalls: [ToolCall]?

    public enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
    }
}

public struct ChatCompletionChunkChoice: Codable {
    public var index: Int
    public var delta: DeltaMessage
    public var finishReason: String?

    public enum CodingKeys: String, CodingKey {
        case index
        case delta
        case finishReason = "finish_reason"
    }
}

public struct ChatCompletionsChunk: Codable {
    public var id: String
    public var object: String = "chat.completion.chunk"
    public var created: Int
    public var model: String
    public var choices: [ChatCompletionChunkChoice]
    public var usage: UsageOut?

    public enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case model
        case choices
        case usage
    }
}

public struct ModelObject: Codable {
    public var id: String
    public var object: String = "model"
    public var created: Int = Int(Date().timeIntervalSince1970)
    public var ownedBy: String = "apple"

    public enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
    }
}

public struct ModelsResponse: Codable {
    public var object: String = "list"
    public var data: [ModelObject]
}

// MARK: - Type-erased JSON value

/// Minimal AnyCodable supporting the subset of JSON used in tool schemas.
public struct AnyCodable: Codable, @unchecked Sendable {
    public var value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            value = NSNull()
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case is NSNull:
            try container.encodeNil()
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        default:
            try container.encode(String(describing: value))
        }
    }
}

public extension AnyCodable {
    var asDictionary: [String: Any]? { value as? [String: Any] }
    var asString: String? { value as? String }
}
