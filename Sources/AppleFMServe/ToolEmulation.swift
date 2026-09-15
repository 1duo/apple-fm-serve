import Foundation

// MARK: - Tool-call emulation for client-executed tools (opencode)

/// AFM native tools auto-execute server-side, but OpenAI clients like opencode
/// execute tools client-side: the server must RETURN tool_calls, not run them.
/// This module translates OpenAI function tools into prompt instructions and
/// parses the model's structured reply back into OpenAI tool_calls.
///
/// Wire format requested from the model (JSON envelope):
///   {"content": string|null, "tool_calls": [{"name": str, "arguments": object|string}]}
public struct ToolEnvelopeCall: Codable, Sendable {
    public var name: String
    public var arguments: ToolEnvelopeArguments?
}

public struct ToolEnvelopeArguments: Codable, Sendable {
    public var raw: String

    public init(_ raw: String) {
        self.raw = raw
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            raw = string
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            let data = try JSONSerialization.data(withJSONObject: dict.mapValues { $0.value }, options: [.sortedKeys])
            raw = String(data: data, encoding: .utf8) ?? "{}"
        } else if container.decodeNil() {
            raw = "{}"
        } else {
            raw = "{}"
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(raw)
    }
}

public struct ToolEnvelope: Codable, Sendable {
    public var content: String?
    public var toolCalls: [ToolEnvelopeCall]?

    public enum CodingKeys: String, CodingKey {
        case content
        case toolCalls = "tool_calls"
    }
}

public struct ParsedToolResponse: Sendable {
    public var content: String?
    public var toolCalls: [ToolCall]
    /// True when calls were salvaged from malformed (non-strict-JSON) output.
    public var recovered: Bool = false
}

/// Describe available functions for the model.
public func toolInstructions(for tools: [ToolDefinition], toolChoice: AnyCodable?) -> String {
    var lines: [String] = []
    lines.append("You have access to the following functions. Use them when they help answer the user's request.")
    lines.append("")
    for tool in tools {
        let fn = tool.function
        var header = "- \(fn.name)"
        if let description = fn.description, !description.isEmpty {
            header += ": \(description)"
        }
        lines.append(header)
        if let parameters = fn.parameters {
            if let data = try? JSONSerialization.data(
                withJSONObject: parameters.mapValues { $0.value },
                options: [.sortedKeys, .prettyPrinted]
            ), let text = String(data: data, encoding: .utf8) {
                lines.append("  Parameters JSON Schema: \(text)")
            }
        }
    }
    lines.append("")
    if let choice = toolChoiceDescription(toolChoice) {
        lines.append(choice)
    } else {
        lines.append("To call functions, reply with ONLY a JSON object of the form:")
        lines.append("{\"content\": <string|null>, \"tool_calls\": [{\"name\": <function-name>, \"arguments\": <object with the function arguments>}]}")
        lines.append("Use the EXACT function names listed above — never invent new function names.")
        lines.append("When no function is needed, use {\"content\": \"<your reply>\", \"tool_calls\": []}.")
        lines.append("When functions are needed, put any user-facing text in \"content\" (or null) and list each call in \"tool_calls\".")
    }
    return lines.joined(separator: "\n")
}

private func toolChoiceDescription(_ toolChoice: AnyCodable?) -> String? {
    guard let toolChoice else { return nil }
    if let string = toolChoice.asString {
        switch string {
        case "none":
            return "Do not call any functions. Reply normally without the JSON envelope."
        case "required", "auto":
            return nil // default envelope instructions apply
        default:
            return nil
        }
    }
    if let dict = toolChoice.asDictionary {
        // {"type": "function", "function": {"name": "x"}} — force that function.
        if let function = dict["function"] as? [String: Any],
           let name = function["name"] as? String, !name.isEmpty {
            return "You MUST call the function \"\(name)\". Reply with ONLY the JSON envelope {\"content\": <string|null>, \"tool_calls\": [{\"name\": \"\(name)\", \"arguments\": {...}}]}."
        }
        if let type = dict["type"] as? String, type == "function" {
            return nil
        }
    }
    return nil
}

public func shouldEmulateTools(request: ChatCompletionsRequest) -> Bool {
    shouldEmulateTools(tools: request.tools, toolChoice: request.toolChoice)
}

public func shouldEmulateTools(tools: [ToolDefinition]?, toolChoice: AnyCodable?) -> Bool {
    guard let tools, !tools.isEmpty else { return false }
    if let choice = toolChoice?.asString, choice == "none" { return false }
    if let dict = toolChoice?.asDictionary,
       let type = dict["type"] as? String, type == "none" { return false }
    return true
}

/// Extract the first balanced {...} JSON object from free text.
public func extractJSONObject(from text: String) -> String? {
    guard let start = text.firstIndex(of: "{") else { return nil }
    var depth = 0
    var inString = false
    var escaped = false
    var index = start
    while index < text.endIndex {
        let char = text[index]
        if inString {
            if escaped {
                escaped = false
            } else if char == "\\" {
                escaped = true
            } else if char == "\"" {
                inString = false
            }
        } else {
            if char == "\"" {
                inString = true
            } else if char == "{" {
                depth += 1
            } else if char == "}" {
                depth -= 1
                if depth == 0 {
                    return String(text[start...index])
                }
            }
        }
        index = text.index(after: index)
    }
    return nil
}

private var toolCallIDCounter = 0
private let toolCallIDLock = NSLock()

/// Deterministic-ish tool_call ids (call_ + 8 hex chars). Uses a counter +
/// randomness so concurrent requests don't collide.
public func newToolCallID() -> String {
    toolCallIDLock.lock()
    toolCallIDCounter += 1
    let counter = toolCallIDCounter
    toolCallIDLock.unlock()
    let random = UInt32.random(in: 0...0xFFFF)
    return String(format: "call_%04x%04x", (counter & 0xFFFF), random)
}

/// Parse a model reply into content + OpenAI tool_calls.
/// Falls back to plain content when no envelope is present (graceful
/// degradation keeps harnesses usable even if the small on-device model
/// ignores the envelope format). When the envelope is present but malformed,
/// a lenient recovery pass salvages calls so the agent loop can continue.
public func parseToolResponse(_ text: String, validNames: Set<String>) -> ParsedToolResponse {
    guard let jsonText = extractJSONObject(from: text),
          let data = jsonText.data(using: .utf8) else {
        return ParsedToolResponse(content: text, toolCalls: [])
    }
    if let envelope = try? JSONDecoder().decode(ToolEnvelope.self, from: data) {
        guard let calls = envelope.toolCalls, !calls.isEmpty else {
            // Valid envelope with no calls: unwrap the content field so clients
            // don't see the raw JSON envelope.
            return ParsedToolResponse(content: envelope.content ?? text, toolCalls: [])
        }
        var toolCalls: [ToolCall] = []
        for call in calls {
            guard validNames.contains(call.name) else { continue }
            let arguments = call.arguments?.raw ?? "{}"
            toolCalls.append(ToolCall(
                id: newToolCallID(),
                function: FunctionCall(name: call.name, arguments: arguments)
            ))
        }
        guard !toolCalls.isEmpty else {
            // Envelope named unknown functions: treat as plain text, unwrapped.
            return ParsedToolResponse(content: envelope.content ?? text, toolCalls: [])
        }
        return ParsedToolResponse(content: envelope.content, toolCalls: toolCalls)
    }
    // Strict decode failed but braces balanced: attempt salvage.
    if let recovered = recoverToolCalls(text) {
        return recovered
    }
    return ParsedToolResponse(content: text, toolCalls: [])
}

/// Coarse classification of a model reply's tool envelope, used to decide
/// whether an operator-visible warning is warranted.
public enum EnvelopeStatus: Equatable {
    /// No tool_calls key / no JSON object at all.
    case none
    /// Valid envelope with zero calls, or no usable envelope (plain prose).
    case empty
    /// Calls were present but none survived (unknown names, or malformed
    /// beyond recovery). Worth a WARN: the turn likely stalls the harness.
    case unusable
}

public func envelopeStatus(_ text: String, validNames: Set<String>) -> EnvelopeStatus {
    guard text.contains("\"tool_calls\"") else { return .none }
    guard let jsonText = extractJSONObject(from: text),
          let data = jsonText.data(using: .utf8) else { return .none }
    guard let envelope = try? JSONDecoder().decode(ToolEnvelope.self, from: data) else {
        // Malformed: recovery already ran upstream; warn only if it failed.
        return recoverToolCalls(text) == nil ? .unusable : .empty
    }
    guard let calls = envelope.toolCalls, !calls.isEmpty else { return .empty }
    return calls.contains(where: { validNames.contains($0.name) }) ? .empty : .unusable
}

/// True when the client demands at least one tool call (OpenAI `required` or
/// a named function choice). Used to decide on a stricter single retry.
public func toolChoiceRequiresTools(_ choice: AnyCodable?) -> Bool {
    if let string = choice?.asString { return string == "required" }
    if let dict = choice?.asDictionary,
       let type = dict["type"] as? String, type == "function" {
        return true
    }
    return false
}

// MARK: - Lenient envelope recovery

/// Scans `text` for the value starting at `index`, returning the substring
/// through the matching close delimiter (`{}`/`[]`), or a bare scalar/string.
func scanJSONValue(in text: String, from index: String.Index) -> (String, String.Index)? {
    var idx = index
    while idx < text.endIndex && text[idx].isWhitespace { idx = text.index(after: idx) }
    guard idx < text.endIndex else { return nil }
    let open = text[idx]
    if open == "{" || open == "[" {
        guard let end = matchDelimiter(in: text, from: idx) else { return nil }
        return (String(text[idx...end]), text.index(after: end))
    }
    if open == "\"" {
        guard let (string, end) = parseJSONString(in: text, from: idx) else { return nil }
        _ = string
        // Return the raw substring (including quotes) for pass-through.
        return (String(text[idx..<end]), end)
    }
    // Bare scalar: read to the next structural character at depth 0.
    var end = idx
    var inString = false
    var escaped = false
    while end < text.endIndex {
        let char = text[end]
        if inString {
            if escaped { escaped = false } else if char == "\\" { escaped = true }
            else if char == "\"" { inString = false }
        } else if char == "\"" {
            inString = true
        } else if ",]}".contains(char) {
            break
        }
        end = text.index(after: end)
    }
    return (String(text[idx..<end]).trimmingCharacters(in: .whitespaces), end)
}

/// Parses a `\uXXXX` escape (plus an optional low-surrogate pair) starting at
/// the `u`. Returns the scalar and the index just past the escape.
func parseUnicodeEscape(in text: String, from uIndex: String.Index) -> (UnicodeScalar, String.Index)? {
    func hex4(from index: String.Index) -> (UInt32, String.Index)? {
        var value: UInt32 = 0
        var idx = index
        for _ in 0..<4 {
            guard idx < text.endIndex, let digit = text[idx].hexDigitValue else { return nil }
            value = value &* 16 &+ UInt32(digit)
            idx = text.index(after: idx)
        }
        return (value, idx)
    }
    var cursor = text.index(after: uIndex) // skip 'u'
    guard let (high, afterHigh) = hex4(from: cursor) else { return nil }
    cursor = afterHigh
    // Surrogate pair: high followed by \uYYYY low.
    if (0xD800...0xDBFF).contains(high),
       cursor < text.endIndex, text[cursor] == "\\" {
        let next = text.index(after: cursor)
        if next < text.endIndex, text[next] == "u",
           let (low, afterLow) = hex4(from: text.index(after: next)),
           (0xDC00...0xDFFF).contains(low) {
            let composed = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
            guard let scalar = UnicodeScalar(composed) else { return nil }
            return (scalar, afterLow)
        }
    }
    guard let scalar = UnicodeScalar(high) else { return nil }
    return (scalar, cursor)
}

/// Matches the close delimiter for the `{`/`[` at `from`, honouring strings
/// and escapes. Returns the index of the close delimiter.
func matchDelimiter(in text: String, from: String.Index) -> String.Index? {
    let open = text[from]
    let close: Character = (open == "{") ? "}" : "]"
    var depth = 0
    var inString = false
    var escaped = false
    var idx = from
    while idx < text.endIndex {
        let char = text[idx]
        if inString {
            if escaped { escaped = false } else if char == "\\" { escaped = true }
            else if char == "\"" { inString = false }
        } else if char == "\"" {
            inString = true
        } else if char == "{" || char == "[" {
            depth += 1
        } else if char == "}" || char == "]" {
            depth -= 1
            if depth == 0 {
                // Accept any close type at depth 0 (models mix them up).
                _ = close
                return idx
            }
        }
        idx = text.index(after: idx)
    }
    return nil
}

/// Parses a JSON string literal starting at the opening quote, returning the
/// unescaped value and the index just past the closing quote.
func parseJSONString(in text: String, from: String.Index) -> (String, String.Index)? {
    var result = ""
    var idx = text.index(after: from) // skip opening quote
    while idx < text.endIndex {
        let char = text[idx]
        if char == "\\" {
            let next = text.index(after: idx)
            guard next < text.endIndex else { return nil }
            switch text[next] {
            case "\"": result.append("\"")
            case "\\": result.append("\\")
            case "/": result.append("/")
            case "n": result.append("\n")
            case "t": result.append("\t")
            case "r": result.append("\r")
            case "b": result.append("\u{08}")
            case "f": result.append("\u{0C}")
            case "u":
                guard let (scalar, after) = parseUnicodeEscape(in: text, from: next) else { return nil }
                result.unicodeScalars.append(scalar)
                idx = after
                continue
            default: result.append(text[next])
            }
            idx = text.index(after: next)
        } else if char == "\"" {
            return (result, text.index(after: idx))
        } else {
            result.append(char)
            idx = text.index(after: idx)
        }
    }
    return nil
}

func skipWhitespaceAndColon(in text: String, from index: String.Index) -> String.Index? {
    var idx = index
    while idx < text.endIndex && (text[idx].isWhitespace || text[idx] == ":") {
        idx = text.index(after: idx)
    }
    return idx < text.endIndex ? idx : nil
}

/// Leniently extracts `{"name", "arguments"}` pairs from a malformed envelope.
/// Unlike the strict path, names are NOT filtered against the known list:
/// the output is already off-spec, and emitting the call lets the harness
/// report an unknown-tool error back to the model (self-healing loop)
/// instead of stalling on a dead text turn. Returns nil when nothing usable
/// is found.
public func recoverToolCalls(_ text: String) -> ParsedToolResponse? {
    guard let keyRange = text.range(of: "\"tool_calls\"") else { return nil }
    guard let colon = text[keyRange.upperBound...].firstIndex(of: ":") else { return nil }
    guard let arrayStart = skipWhitespaceAndColon(in: text, from: text.index(after: colon)),
          arrayStart < text.endIndex, text[arrayStart] == "[",
          let arrayEnd = matchDelimiter(in: text, from: arrayStart) else { return nil }
    let content = recoverJSONString(forKey: "content", in: text)
    var calls: [ToolCall] = []
    var cursor = text.index(after: arrayStart)
    var iterations = 0
    while cursor < arrayEnd, iterations < 64 {
        iterations += 1
        guard let nameKey = text[cursor..<arrayEnd].range(of: "\"name\"") else { break }
        var after = nameKey.upperBound
        guard let valueStart = skipWhitespaceAndColon(in: text, from: after) else { break }
        after = valueStart
        guard after < text.endIndex, text[after] == "\"",
              let (name, nameEnd) = parseJSONString(in: text, from: after) else {
            cursor = text.index(after: nameKey.upperBound)
            continue
        }
        after = nameEnd
        var arguments = "{}"
        // Bound the arguments search to this call: stop at the next "name"
        // so a missing arguments key can't steal a later call's value.
        let callEnd = text[after..<arrayEnd].range(of: "\"name\"")?.lowerBound ?? arrayEnd
        if let argsKey = text[after..<callEnd].range(of: "\"arguments\""),
           let argsStart = skipWhitespaceAndColon(in: text, from: argsKey.upperBound),
           let (raw, argsEnd) = scanJSONValue(in: text, from: argsStart) {
            arguments = raw
            after = argsEnd
        }
        if name.isEmpty {
            cursor = after
            continue
        }
        calls.append(ToolCall(id: newToolCallID(), function: FunctionCall(name: name, arguments: arguments)))
        cursor = after
        if cursor <= nameKey.upperBound { break }
    }
    guard !calls.isEmpty else { return nil }
    return ParsedToolResponse(content: content, toolCalls: calls, recovered: true)
}

/// Best-effort extraction of a top-level string field (unescaped).
func recoverJSONString(forKey key: String, in text: String) -> String? {
    guard let keyRange = text.range(of: "\"\(key)\""),
          let colon = text[keyRange.upperBound...].firstIndex(of: ":"),
          let valueStart = skipWhitespaceAndColon(in: text, from: text.index(after: colon)),
          valueStart < text.endIndex else { return nil }
    if text[valueStart...].hasPrefix("null") { return nil }
    guard text[valueStart] == "\"",
          let (string, _) = parseJSONString(in: text, from: valueStart) else { return nil }
    return string
}
