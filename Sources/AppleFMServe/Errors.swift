import Foundation

/// OpenAI-compatible error envelope.
public struct OpenAIErrorBody: Codable {
    public var message: String
    public var type: String
    public var param: String?
    public var code: String?
}

/// Adapter error carrying an HTTP status code.
public struct AdapterError: Error {
    public var message: String
    public var statusCode: Int
    public var type: String
    public var param: String?
    public var code: String?

    public init(
        _ message: String,
        statusCode: Int = 400,
        type: String = "invalid_request_error",
        param: String? = nil,
        code: String? = nil
    ) {
        self.message = message
        self.statusCode = statusCode
        self.type = type
        self.param = param
        self.code = code
    }

    public var body: OpenAIErrorBody {
        OpenAIErrorBody(message: message, type: type, param: param, code: code)
    }

    static func modelNotFound(_ model: String) -> AdapterError {
        AdapterError(
            "Model '\(model)' not found",
            statusCode: 404,
            param: "model",
            code: "model_not_found"
        )
    }

    static var invalidAPIKey: AdapterError {
        AdapterError(
            "Invalid API key",
            statusCode: 401,
            type: "authentication_error",
            code: "invalid_api_key"
        )
    }

    static func providerUnavailable(_ reason: String?) -> AdapterError {
        let message = reason.map { "Model provider unavailable: \($0)" } ?? "Model provider unavailable"
        return AdapterError(message, statusCode: 503, type: "server_error", code: "provider_unavailable")
    }

    static var requestTimeout: AdapterError {
        AdapterError(
            "Request timed out",
            statusCode: 504,
            type: "timeout_error",
            code: "request_timeout"
        )
    }
}
