import Foundation

/// Result of folding OpenAI messages into AFM inputs.
/// `instructions` maps to LanguageModelSession instructions;
/// `prompt` is the single-turn prompt carrying full conversation fidelity
/// (including tool calls/results) for stateless HTTP serving.
public struct PromptBuildResult: Sendable {
    public var instructions: String?
    public var prompt: String
}

/// Extract plain text from message content. Rejects images like the
/// previous Python adapter (AFM text sessions don't accept image parts
/// via this shim; image-capable tools are out of scope for v1).
public func messageContentToText(_ content: MessageContent?) throws -> String {
    guard let content else { return "" }
    switch content {
    case .text(let text):
        return text
    case .parts(let parts):
        var texts: [String] = []
        for part in parts {
            if part.type == "image_url" {
                throw AdapterError(
                    "Image content is not supported by this adapter",
                    statusCode: 400,
                    param: "messages",
                    code: "image_not_supported"
                )
            }
            guard part.type == "text", let text = part.text, !text.isEmpty else { continue }
            texts.append(text)
        }
        return texts.joined(separator: "\n")
    }
}

/// Format a tool call for transcript fidelity.
public func formatAssistantToolCalls(_ toolCalls: [ToolCall]?) -> String? {
    guard let toolCalls, !toolCalls.isEmpty else { return nil }
    let parts = toolCalls.map { call in
        "call \(call.id) \(call.function.name)(\(call.function.arguments))"
    }
    return "Tool calls: " + parts.joined(separator: "; ")
}

public func buildPrompt(messages: [ChatMessage]) throws -> PromptBuildResult {
    guard !messages.isEmpty else {
        throw AdapterError(
            "messages must contain at least one item",
            statusCode: 400,
            param: "messages",
            code: "messages_required"
        )
    }

    var instructionParts: [String] = []
    var turns: [String] = []

    for message in messages {
        let text = try messageContentToText(message.content)
        switch message.role {
        case "system", "developer":
            // pi and newer OpenAI clients send `developer` instead of `system`
            // for harness instructions; both map to AFM instructions.
            if !text.isEmpty { instructionParts.append(text) }
        case "user":
            turns.append("User: \(text)")
        case "assistant":
            if let toolText = formatAssistantToolCalls(message.toolCalls) {
                if text.isEmpty {
                    turns.append("Assistant: [\(toolText)]")
                } else {
                    turns.append("Assistant: \(text)\n[\(toolText)]")
                }
            } else {
                turns.append("Assistant: \(text)")
            }
        case "tool":
            let prefix: String
            if let callID = message.toolCallID, !callID.isEmpty {
                prefix = "Tool[\(callID)]"
            } else {
                prefix = "Tool"
            }
            turns.append("\(prefix): \(text)")
        case "function":
            // Legacy OpenAI role; fold into tool turn like `fm serve` rejects
            // bare 'function' without tool_call_id, but be liberal here.
            let prefix: String
            if let callID = message.toolCallID, !callID.isEmpty {
                prefix = "Tool[\(callID)]"
            } else if let name = message.name, !name.isEmpty {
                prefix = "Tool[\(name)]"
            } else {
                prefix = "Tool"
            }
            turns.append("\(prefix): \(text)")
        default:
            // Unknown roles: preserve as user turns rather than dropping context.
            turns.append("User: \(text)")
        }
    }

    guard !turns.isEmpty else {
        throw AdapterError(
            "At least one non-system message is required",
            statusCode: 400,
            param: "messages",
            code: "no_turns"
        )
    }

    let prompt = "You are answering the final user request in the following conversation.\n"
        + "Return only the assistant response.\n\n"
        + turns.joined(separator: "\n")
        + "\nAssistant:"
    let instructions = instructionParts.filter { !$0.isEmpty }.joined(separator: "\n\n")
    return PromptBuildResult(
        instructions: instructions.isEmpty ? nil : instructions,
        prompt: prompt
    )
}

// MARK: - Streaming delta helper

/// AFM streams full-text snapshots (complete response so far), not deltas.
/// Compute the newly appended suffix, falling back to the whole snapshot
/// when the stream diverges (edits/replacements).
public func textDeltasFromSnapshots(_ snapshots: [String]) -> [String] {
    var previous = ""
    var deltas: [String] = []
    for snapshot in snapshots {
        var delta: String
        if snapshot.hasPrefix(previous) {
            delta = String(snapshot.dropFirst(previous.count))
        } else {
            delta = snapshot
        }
        previous = snapshot
        if !delta.isEmpty { deltas.append(delta) }
    }
    return deltas
}

/// Incremental form used by the live streaming path.
public struct SnapshotDiffer {
    private var previous = ""

    mutating func delta(for snapshot: String) -> String? {
        let delta: String
        if snapshot.hasPrefix(previous) {
            delta = String(snapshot.dropFirst(previous.count))
        } else {
            delta = snapshot
        }
        previous = snapshot
        return delta.isEmpty ? nil : delta
    }

    mutating func reset() {
        previous = ""
    }
}

// MARK: - Token estimation fallback

/// Rough fallback when the native tokenizer/usage APIs are unavailable.
/// Matches the previous Python adapter (len // 4) so usage never goes missing.
public func estimateTokens(_ text: String) -> Int {
    guard !text.isEmpty else { return 0 }
    return max(1, text.count / 4)
}

// MARK: - Stop-sequence emulation

/// AFM has no server-side `stop` parameter; truncate at the first stop string.
public func applyStopSequences(_ text: String, stop: StopSequences?) -> (text: String, truncated: Bool) {
    guard let stop else { return (text, false) }
    var earliest: Range<String.Index>?
    for sequence in stop.values where !sequence.isEmpty {
        if let range = text.range(of: sequence), earliest == nil || range.lowerBound < earliest!.lowerBound {
            earliest = range
        }
    }
    guard let range = earliest else { return (text, false) }
    return (String(text[..<range.lowerBound]), true)
}
