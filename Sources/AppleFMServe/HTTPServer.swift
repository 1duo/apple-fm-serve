import Foundation
import Network

// MARK: - Minimal HTTP/1.1 server (stdlib + Network.framework, no deps)

public struct HTTPRequest: Sendable {
    public var method: String
    public var path: String
    public var query: String?
    public var headers: [String: String]
    public var body: Data
}

public struct HTTPResponse {
    public var statusCode: Int
    public var statusText: String
    public var headers: [String: String]
    public var body: Data?

    static func json(_ value: some Encodable, status: Int = 200) -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(value)) ?? Data("{\"error\":{\"message\":\"encoding failure\"}}".utf8)
        return HTTPResponse(
            statusCode: status,
            statusText: statusText(for: status),
            headers: [
                "Content-Type": "application/json",
                "Content-Length": "\(data.count)",
                "Connection": "close",
                "Access-Control-Allow-Origin": "*",
            ],
            body: data
        )
    }

    static func error(_ error: AdapterError) -> HTTPResponse {
        var detail: [String: String] = [
            "message": error.message,
            "type": error.type,
        ]
        if let param = error.param { detail["param"] = param }
        if let code = error.code { detail["code"] = code }
        return .json(["error": detail], status: error.statusCode)
    }

    static func statusText(for code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 413: return "Payload Too Large"
        case 429: return "Too Many Requests"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        case 504: return "Gateway Timeout"
        default: return "OK"
        }
    }
}

public actor ConcurrencyLimiter {
    private var permits: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(permits: Int) {
        self.permits = permits
    }

    public func acquire() async {
        if permits > 0 {
            permits -= 1
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    public func release() {
        if !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            waiter.resume()
        } else {
            permits += 1
        }
    }
}

public final class HTTPServer: Sendable {
    private let config: Config
    private let handler: ChatHandler
    private let limiter: ConcurrencyLimiter

    public init(config: Config, handler: ChatHandler) {
        self.config = config
        self.handler = handler
        self.limiter = ConcurrencyLimiter(permits: config.maxConcurrency)
    }

    public func run() async throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: UInt16(config.port)) else {
            throw AdapterError("Invalid port \(config.port)", statusCode: 500, type: "server_error", code: "invalid_port")
        }
        // NOTE: NWListener binds to all interfaces for the port. APPLE_FM_HOST
        // controls the advertised address; firewall + localhost-only opencode
        // usage keeps this safe for v1. Precise IP binding would need BSD sockets.
        let listener = try NWListener(using: parameters, on: port)
        print("apple-fm-serve listening on \(config.host):\(config.port) model=\(config.modelID)")

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    continuation.resume()
                case .failed(let error):
                    continuation.resume(throwing: error)
                default:
                    break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                connection.start(queue: .global())
                Task { await self.handleConnection(connection) }
            }
            listener.start(queue: .global())
        }

        // Keep running until cancelled.
        while true {
            try await Task.sleep(nanoseconds: 3_600_000_000_000)
        }
    }

    private func handleConnection(_ connection: NWConnection) async {
        defer { connection.cancel() }
        let request: HTTPRequest
        switch await readRequest(connection) {
        case .request(let parsed):
            request = parsed
        case .tooLarge(let length):
            logLine("WARN body too large (\(length)B), rejecting with 413")
            await sendResponse(connection, .error(AdapterError(
                "Request body too large",
                statusCode: 413,
                code: "body_too_large"
            )))
            return
        case .none:
            logLine("WARN no-request (parse failed or empty)")
            return
        }
        logLine("\(request.method) \(request.path) body=\(request.body.count)B")
        await limiter.acquire()
        defer { Task { await limiter.release() } }
        do {
            // Preflight support for browser-based harnesses.
            if request.method == "OPTIONS" {
                await sendResponse(connection, HTTPResponse(
                    statusCode: 200, statusText: "OK",
                    headers: [
                        "Content-Length": "0",
                        "Connection": "close",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    ],
                    body: nil
                ))
                return
            }
            let path = request.path
            if request.method == "GET" && (path == "/healthz" || path == "/health") {
                await sendResponse(connection, .json(["status": "ok"]))
                return
            }
            if request.method == "GET" && path == "/readyz" {
                let (available, reason) = await handler.provider.isAvailable()
                if available {
                    await sendResponse(connection, .json(["status": "ready"]))
                } else {
                    await sendResponse(connection, .error(.providerUnavailable(reason)))
                }
                return
            }
            if request.method == "GET" && path == "/v1/models" {
                guard await checkAuthOrRespond(connection, headers: request.headers) else { return }
                let (available, reason) = await handler.provider.isAvailable()
                guard available else {
                    await sendResponse(connection, .error(.providerUnavailable(reason)))
                    return
                }
                let models = await handler.provider.listModels()
                await sendResponse(connection, .json(ModelsResponse(data: models.map { ModelObject(id: $0) })))
                return
            }
            if request.method == "POST" && path == "/v1/chat/completions" {
                guard await checkAuthOrRespond(connection, headers: request.headers) else { return }
                let chatRequest: ChatCompletionsRequest
                do {
                    chatRequest = try JSONDecoder().decode(ChatCompletionsRequest.self, from: request.body)
                    let chatToolNames = (chatRequest.tools ?? []).map { $0.function.name }.joined(separator: ",")
                    logLine("chat model=\(chatRequest.model) msgs=\(chatRequest.messages.count) tools=\(chatRequest.tools?.count ?? 0)[\(chatToolNames.prefix(160))] stream=\(chatRequest.stream) toolChoice=\(toolChoiceSummary(chatRequest.toolChoice))")
                    if handler.config.logBodies {
                        logLine("body: \(bodySnippet(request.body))")
                    }
                } catch {
                    logLine("WARN bad chat body: \(bodySnippet(request.body))")
                    await sendResponse(connection, .error(AdapterError(
                        "Invalid request body: \(error.localizedDescription)",
                        statusCode: 400,
                        code: "invalid_request"
                    )))
                    return
                }
                if chatRequest.stream {
                    await handleStream(connection, request: chatRequest)
                } else {
                    do {
                        let response = try await handler.complete(request: chatRequest, model: chatRequest.model)
                        let choice = response.choices.first
                        logLine("-> 200 chat finish=\(choice?.finishReason ?? "-") toolcalls=\(choice?.message.toolCalls?.count ?? 0) usage=\(response.usage.map { "\($0.promptTokens)/\($0.completionTokens)" } ?? "-")")
                        await sendResponse(connection, .json(response))
                    } catch let adapter as AdapterError {
                        logLine("-> \(adapter.statusCode) chat error=\(adapter.code ?? "-")")
                        await sendResponse(connection, .error(adapter))
                    } catch {
                        await sendResponse(connection, .error(AdapterError(
                            "Provider error: \(error.localizedDescription)",
                            statusCode: 500, type: "server_error", code: "provider_error"
                        )))
                    }
                }
                return
            }
            if request.method == "POST" && path == "/v1/responses" {
                guard await checkAuthOrRespond(connection, headers: request.headers) else { return }
                let responsesRequest: ResponsesRequest
                do {
                    responsesRequest = try JSONDecoder().decode(ResponsesRequest.self, from: request.body)
                    let toolNames = (responsesRequest.tools ?? []).map { $0.name ?? $0.type }.joined(separator: ",")
                    logLine("responses model=\(responsesRequest.model) stream=\(responsesRequest.stream) tools=\(responsesRequest.tools?.count ?? 0)[\(toolNames.prefix(160))] choice=\(toolChoiceSummary(responsesRequest.toolChoice))")
                    if handler.config.logBodies {
                        logLine("body: \(bodySnippet(request.body, maxChars: 1200))")
                    }
                } catch {
                    logLine("WARN bad responses body: \(bodySnippet(request.body))")
                    await sendResponse(connection, .error(AdapterError(
                        "Invalid request body: \(error.localizedDescription)",
                        statusCode: 400,
                        code: "invalid_request"
                    )))
                    return
                }
                if responsesRequest.stream {
                    // Buffered: failures still return HTTP error statuses.
                    do {
                        let events = try await handler.responsesEvents(request: responsesRequest, model: responsesRequest.model)
                        let headerLines = [
                            "HTTP/1.1 200 OK",
                            "Content-Type: text/event-stream",
                            "Cache-Control: no-cache",
                            "Connection: close",
                            "Access-Control-Allow-Origin: *",
                            "",
                            "",
                        ].joined(separator: "\r\n")
                        await sendRaw(connection, Data(headerLines.utf8))
                        for event in events {
                            await sendRaw(connection, Data(event.utf8))
                        }
                        logLine("-> 200 responses-stream events=\(events.count)")
                    } catch let adapter as AdapterError {
                        logLine("-> \(adapter.statusCode) responses-stream error=\(adapter.code ?? "-")")
                        await sendResponse(connection, .error(adapter))
                    } catch {
                        await sendResponse(connection, .error(AdapterError(
                            "Provider error: \(error.localizedDescription)",
                            statusCode: 500, type: "server_error", code: "provider_error"
                        )))
                    }
                } else {
                    do {
                        let response = try await handler.completeResponses(request: responsesRequest, model: responsesRequest.model)
                        logLine("-> 200 responses items=\(response.output.count) usage=\(response.usage.map { "\($0.inputTokens)/\($0.outputTokens)" } ?? "-")")
                        await sendResponse(connection, .json(response))
                    } catch let adapter as AdapterError {
                        logLine("-> \(adapter.statusCode) responses error=\(adapter.code ?? "-")")
                        await sendResponse(connection, .error(adapter))
                    } catch {
                        await sendResponse(connection, .error(AdapterError(
                            "Provider error: \(error.localizedDescription)",
                            statusCode: 500, type: "server_error", code: "provider_error"
                        )))
                    }
                }
                return
            }
            await sendResponse(connection, .error(AdapterError(
                "Not found", statusCode: 404, code: "not_found"
            )))
        }
    }

    /// Shared bearer-auth gate. Returns false after responding when rejected.
    private func checkAuthOrRespond(_ connection: NWConnection, headers: [String: String]) async -> Bool {
        do {
            try handler.checkAuth(authorization: headers["authorization"])
            return true
        } catch let adapter as AdapterError {
            await sendResponse(connection, .error(adapter))
            return false
        } catch {
            await sendResponse(connection, .error(AdapterError(
                "Auth check failed", statusCode: 401, type: "authentication_error", code: "invalid_api_key"
            )))
            return false
        }
    }

    private func handleStream(_ connection: NWConnection, request: ChatCompletionsRequest) async {
        // Send SSE headers immediately, then stream chunks.
        let headerLines = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream",
            "Cache-Control: no-cache",
            "Connection: close",
            "Access-Control-Allow-Origin: *",
            "",
            "",
        ].joined(separator: "\r\n")
        await sendRaw(connection, Data(headerLines.utf8))
        do {
            var chunks = 0
            for try await payload in handler.completeStream(request: request, model: request.model) {
                chunks += 1
                await sendRaw(connection, Data(payload.utf8))
            }
            logLine("-> 200 chat-stream chunks=\(chunks)")
        } catch let adapter as AdapterError {
            logLine("-> stream error=\(adapter.code ?? "-")")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            if let data = try? encoder.encode(["error": [
                "message": adapter.message,
                "type": adapter.type,
                "code": adapter.code ?? "",
            ]]),
               let text = String(data: data, encoding: .utf8) {
                await sendRaw(connection, Data("data: \(text)\n\n".utf8))
            }
        } catch {
            // Close stream on unexpected errors (client sees truncated stream).
        }
    }

    // MARK: - Socket IO

    enum RequestRead: Sendable {
        case request(HTTPRequest)
        case tooLarge(Int)
        case none
    }

    private func readRequest(_ connection: NWConnection) async -> RequestRead {
        var buffer = Data()
        // Read until end of headers.
        while true {
            guard let chunk = await receive(connection, maxLength: 65536) else { return .none }
            if chunk.isEmpty { return .none }
            buffer.append(chunk)
            if let range = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let headerData = buffer[..<range.lowerBound]
                guard let headerText = String(data: headerData, encoding: .utf8) else { return .none }
                let lines = headerText.components(separatedBy: "\r\n")
                guard let requestLine = lines.first else { return .none }
                let parts = requestLine.split(separator: " ")
                guard parts.count >= 2 else { return .none }
                let method = String(parts[0])
                let target = String(parts[1])
                var headers: [String: String] = [:]
                for line in lines.dropFirst() {
                    guard let colon = line.firstIndex(of: ":") else { continue }
                    let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                    let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                    headers[name] = value
                }
                let contentLength = Int(headers["content-length"] ?? "0") ?? 0
                guard contentLengthPermitted(contentLength) else { return .tooLarge(contentLength) }
                var body = buffer[range.upperBound...]
                var remaining = contentLength - body.count
                while remaining > 0 {
                    guard let chunk = await receive(connection, maxLength: min(65536, remaining)) else { return .none }
                    if chunk.isEmpty { break }
                    body.append(chunk)
                    remaining -= chunk.count
                }
                let path: String
                let query: String?
                if let question = target.firstIndex(of: "?") {
                    path = String(target[..<question])
                    query = String(target[target.index(after: question)...])
                } else {
                    path = target
                    query = nil
                }
                return .request(HTTPRequest(method: method, path: path, query: query, headers: headers, body: Data(body)))
            }
            if buffer.count > 1_000_000 { return .none }
        }
    }

    private func receive(_ connection: NWConnection, maxLength: Int) async -> Data? {
        await withCheckedContinuation { continuation in
            connection.receive(minimumIncompleteLength: 1, maximumLength: maxLength) { data, _, isComplete, error in
                if error != nil {
                    continuation.resume(returning: nil)
                    return
                }
                if let data, !data.isEmpty {
                    continuation.resume(returning: data)
                    return
                }
                continuation.resume(returning: isComplete ? Data() : nil)
            }
        }
    }

    private func sendResponse(_ connection: NWConnection, _ response: HTTPResponse) async {
        logLine("-> \(response.statusCode) \(response.statusText)")
        var text = "HTTP/1.1 \(response.statusCode) \(response.statusText)\r\n"
        for (key, value) in response.headers {
            text += "\(key): \(value)\r\n"
        }
        text += "\r\n"
        var data = Data(text.utf8)
        if let body = response.body { data.append(body) }
        await sendRaw(connection, data)
    }

    private func sendRaw(_ connection: NWConnection, _ data: Data) async {
        guard !data.isEmpty else { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            connection.send(content: data, completion: .contentProcessed { _ in
                continuation.resume()
            })
        }
    }
}

/// Unbuffered stderr logging (visible even when stdout is redirected).
public func logLine(_ message: String) {
    let line = "[apple-fm-serve] \(message)\n"
    if let data = line.data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

/// Maximum accepted request body (16MB). AFM's 8K-token window makes larger
/// payloads pointless; the cap bounds memory per connection.
public let maxRequestBodyBytes = 16 * 1024 * 1024

public func contentLengthPermitted(_ length: Int, limit: Int = maxRequestBodyBytes) -> Bool {
    length >= 0 && length <= limit
}

/// One-word summary of a tool_choice value for request logs.
public func toolChoiceSummary(_ choice: AnyCodable?) -> String {
    guard let choice else { return "-" }
    if let string = choice.asString { return string }
    if let dict = choice.asDictionary {
        if let function = dict["function"] as? [String: Any],
           let name = function["name"] as? String {
            return "fn:\(name)"
        }
        return "object"
    }
    return "?"
}

/// Truncated request-body snippet for compat debugging (never logs bodies by
/// default; see APPLE_FM_LOG_BODIES).
public func bodySnippet(_ body: Data, maxChars: Int = 300) -> String {
    bodySnippet(String(data: body, encoding: .utf8) ?? "<non-utf8 \(body.count)B>", maxChars: maxChars)
}

public func bodySnippet(_ text: String, maxChars: Int = 300) -> String {
    guard text.count > maxChars else { return text }
    return String(text.prefix(maxChars)) + "…<truncated>"
}
