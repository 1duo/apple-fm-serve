import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Generation request handed to an AFM provider.
public struct GenerationRequest: Sendable {
    public var instructions: String?
    public var prompt: String
    public var temperature: Double?
    public var topP: Double?
    public var maxTokens: Int?
    /// Raw JSON schema string for guided generation (response_format json_schema).
    public var responseSchemaJSON: String?
}

/// Generation result with real usage when available.
public struct GenerationResult: Sendable {
    public var text: String
    public var promptTokens: Int?
    public var completionTokens: Int?
}

/// Streaming snapshot from a provider (full text so far + optional usage).
public struct StreamSnapshot: Sendable {
    public var text: String
    public var promptTokens: Int?
    public var completionTokens: Int?
}

public protocol LLMProvider: Sendable {
    func isAvailable() async -> (Bool, String?)
    func listModels() async -> [String]
    func generate(_ request: GenerationRequest) async throws -> GenerationResult
    func streamGenerate(_ request: GenerationRequest) -> AsyncThrowingStream<StreamSnapshot, Error>
}

// MARK: - Error mapping

#if canImport(FoundationModels)
@available(macOS 26, *)
public func mapAFMError(_ error: Error) -> AdapterError {
    let description = String(describing: error)
    let lowered = description.lowercased()
    // GenerationError cases surface via localized description / type name.
    // Match liberally: live AFM errors observed as "Exceeded model context
    // window size" (localizedDescription), not just the case name.
    if lowered.contains("exceededcontextwindowsize") || lowered.contains("contextsizeexceeded")
        || lowered.contains("context window") || lowered.contains("context_length_exceeded") {
        return AdapterError(
            "Context window exceeded",
            statusCode: 400,
            param: "messages",
            code: "context_length_exceeded"
        )
    }
    if description.contains("assetsUnavailable") || description.contains("AssetsUnavailable") {
        return AdapterError(
            "Model assets unavailable",
            statusCode: 503,
            type: "server_error",
            code: "assets_unavailable"
        )
    }
    if description.contains("rateLimited") || description.contains("RateLimited") {
        return AdapterError(
            "Rate limited by provider",
            statusCode: 429,
            type: "rate_limit_error",
            code: "rate_limited"
        )
    }
    if description.contains("guardrailViolation") || description.contains("GuardrailViolation")
        || description.contains("refusal") || description.contains("Refusal") {
        return AdapterError(description, statusCode: 400, code: "guardrail_violation")
    }
    if description.contains("unsupportedGuide") || description.contains("UnsupportedGuide")
        || description.contains("unsupportedCapability") {
        return AdapterError(description, statusCode: 400, code: "unsupported_guide")
    }
    return AdapterError(
        "Provider error: \(error.localizedDescription)",
        statusCode: 500,
        type: "server_error",
        code: "provider_error"
    )
}
#endif

// MARK: - System (on-device) provider

/// On-device AFM via SystemLanguageModel.default.
/// The system auto-selects the most capable variant for the hardware:
/// AFM 3 Core Advanced (20B sparse) on M3+ with 12GB+, else AFM 3 Core (3B).
public final class SystemProvider: LLMProvider, Sendable {
    private let modelID: String
    private let useCase: String
    private let guardrails: String

    public init(modelID: String, useCase: String = "general", guardrails: String = "default") {
        self.modelID = modelID
        self.useCase = useCase
        self.guardrails = guardrails
    }

    public func isAvailable() async -> (Bool, String?) {
#if canImport(FoundationModels)
        if #available(macOS 26, *) {
            // Check the configured model, not .default: useCase/guardrails
            // variants can differ in readiness.
            let model = makeModel()
            switch model.availability {
            case .available:
                return (true, nil)
            case .unavailable(let reason):
                return (false, String(describing: reason))
            @unknown default:
                return (false, "unknown")
            }
        } else {
            return (false, "requires macOS 26+")
        }
#else
        return (false, "FoundationModels unavailable in this build")
#endif
    }

    public func listModels() async -> [String] {
        [modelID]
    }

    public func generate(_ request: GenerationRequest) async throws -> GenerationResult {
#if canImport(FoundationModels)
        if #available(macOS 26, *) {
            do {
                return try await performGenerate(request)
            } catch let adapter as AdapterError {
                throw adapter
            } catch {
                throw mapAFMError(error)
            }
        } else {
            throw AdapterError("Requires macOS 26+", statusCode: 500, type: "server_error", code: "unsupported_os")
        }
#else
        throw AdapterError("FoundationModels unavailable", statusCode: 500, type: "server_error", code: "provider_unavailable")
#endif
    }

    public func streamGenerate(_ request: GenerationRequest) -> AsyncThrowingStream<StreamSnapshot, Error> {
        AsyncThrowingStream { continuation in
            Task {
#if canImport(FoundationModels)
                if #available(macOS 26, *) {
                    do {
                        for try await snapshot in try await performStream(request) {
                            continuation.yield(snapshot)
                        }
                        continuation.finish()
                    } catch let adapter as AdapterError {
                        continuation.finish(throwing: adapter)
                    } catch {
                        continuation.finish(throwing: mapAFMError(error))
                    }
                } else {
                    continuation.finish(throwing: AdapterError(
                        "Requires macOS 26+", statusCode: 500, type: "server_error", code: "unsupported_os"
                    ))
                }
#else
                continuation.finish(throwing: AdapterError(
                    "FoundationModels unavailable", statusCode: 500, type: "server_error", code: "provider_unavailable"
                ))
#endif
            }
        }
    }

#if canImport(FoundationModels)
    @available(macOS 26, *)
    private func makeModel() -> SystemLanguageModel {
        // useCase/guardrails select specialized behavior; default (.general,
        // .default) is the most capable general coding configuration.
        let useCaseValue: SystemLanguageModel.UseCase =
            (useCase == "content-tagging" || useCase == "contentTagging") ? .contentTagging : .general
        let guardrailsValue: SystemLanguageModel.Guardrails =
            (guardrails == "permissive-content-transformations" || guardrails == "permissive") ? .permissiveContentTransformations : .default
        return SystemLanguageModel(useCase: useCaseValue, guardrails: guardrailsValue)
    }

    @available(macOS 26, *)
    private func makeOptions(_ request: GenerationRequest) -> GenerationOptions {
        var options = GenerationOptions()
        if let temperature = request.temperature {
            options.temperature = temperature
        }
        if let maxTokens = request.maxTokens {
            options.maximumResponseTokens = maxTokens
        }
        if let topP = request.topP {
            options.samplingMode = .random(probabilityThreshold: topP)
        }
        return options
    }

    @available(macOS 26, *)
    private func performGenerate(_ request: GenerationRequest) async throws -> GenerationResult {
        let model = makeModel()
        let session = LanguageModelSession(model: model, instructions: request.instructions)
        let options = makeOptions(request)
        // Guided generation when the caller supplied a JSON schema: native
        // enforcement via a translated GenerationSchema, else a prompt hint.
        if let schemaJSON = request.responseSchemaJSON {
            if let schema = try? buildGenerationSchema(name: "Response", jsonSchema: schemaJSON) {
                let response = try await session.respond(
                    to: request.prompt, schema: schema,
                    includeSchemaInPrompt: true, options: options
                )
                let text = response.rawContent.jsonString
                if #available(macOS 27, *) {
                    return GenerationResult(
                        text: text,
                        promptTokens: response.usage.input.totalTokenCount,
                        completionTokens: response.usage.output.totalTokenCount
                    )
                } else {
                    return GenerationResult(text: text, promptTokens: nil, completionTokens: nil)
                }
            }
            logLine("WARN response_format schema beyond translator subset; using prompt hint")
            let hinted = request.prompt + "\n\nRespond with JSON matching this schema:\n" + schemaJSON
            let response = try await session.respond(to: hinted, options: options)
            if #available(macOS 27, *) {
                return GenerationResult(
                    text: response.content,
                    promptTokens: response.usage.input.totalTokenCount,
                    completionTokens: response.usage.output.totalTokenCount
                )
            } else {
                return GenerationResult(text: response.content, promptTokens: nil, completionTokens: nil)
            }
        }
        let response = try await session.respond(to: request.prompt, options: options)
        if #available(macOS 27, *) {
            return GenerationResult(
                text: response.content,
                promptTokens: response.usage.input.totalTokenCount,
                completionTokens: response.usage.output.totalTokenCount
            )
        } else {
            return GenerationResult(text: response.content, promptTokens: nil, completionTokens: nil)
        }
    }

    @available(macOS 26, *)
    private func performStream(_ request: GenerationRequest) async throws -> AsyncThrowingStream<StreamSnapshot, Error> {
        let model = makeModel()
        let session = LanguageModelSession(model: model, instructions: request.instructions)
        let options = makeOptions(request)
        let stream = session.streamResponse(to: request.prompt, options: options)
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await snapshot in stream {
                        let text = snapshot.content
                        if #available(macOS 27, *) {
                            continuation.yield(StreamSnapshot(
                                text: text,
                                promptTokens: snapshot.usage.input.totalTokenCount,
                                completionTokens: snapshot.usage.output.totalTokenCount
                            ))
                        } else {
                            continuation.yield(StreamSnapshot(text: text, promptTokens: nil, completionTokens: nil))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
#endif
}

// MARK: - PCC (Private Cloud Compute) provider, optional

/// Server-side frontier model (32K context, reasoning). Requires the
/// `com.apple.developer.private-cloud-compute` managed entitlement and an
/// eligible account; without it requests fail (ModelManagerError 1046).
/// Only used when APPLE_FM_PREFER_PCC=1 and the model reports available.
public final class PCCProvider: LLMProvider, Sendable {
    private let modelID: String

    public init(modelID: String = "apple.fm.pcc") {
        self.modelID = modelID
    }

    public func isAvailable() async -> (Bool, String?) {
#if canImport(FoundationModels)
        if #available(macOS 27, *) {
            let model = PrivateCloudComputeLanguageModel()
            switch model.availability {
            case .available:
                return (true, nil)
            case .unavailable(let reason):
                return (false, String(describing: reason))
            @unknown default:
                return (false, "unknown")
            }
        } else {
            return (false, "requires macOS 27+")
        }
#else
        return (false, "FoundationModels unavailable")
#endif
    }

    public func listModels() async -> [String] {
        [modelID]
    }

    public func generate(_ request: GenerationRequest) async throws -> GenerationResult {
#if canImport(FoundationModels)
        if #available(macOS 27, *) {
            do {
                let model = PrivateCloudComputeLanguageModel()
                let session = LanguageModelSession(model: model, instructions: request.instructions)
                var options = GenerationOptions()
                if let temperature = request.temperature { options.temperature = temperature }
                if let maxTokens = request.maxTokens { options.maximumResponseTokens = maxTokens }
                let response = try await session.respond(to: request.prompt, options: options)
                return GenerationResult(
                    text: response.content,
                    promptTokens: response.usage.input.totalTokenCount,
                    completionTokens: response.usage.output.totalTokenCount
                )
            } catch let adapter as AdapterError {
                throw adapter
            } catch {
                throw mapAFMError(error)
            }
        } else {
            throw AdapterError("PCC requires macOS 27+", statusCode: 500, type: "server_error", code: "unsupported_os")
        }
#else
        throw AdapterError("FoundationModels unavailable", statusCode: 500, type: "server_error", code: "provider_unavailable")
#endif
    }

    public func streamGenerate(_ request: GenerationRequest) -> AsyncThrowingStream<StreamSnapshot, Error> {
        AsyncThrowingStream { continuation in
            Task {
#if canImport(FoundationModels)
                if #available(macOS 27, *) {
                    do {
                        let model = PrivateCloudComputeLanguageModel()
                        let session = LanguageModelSession(model: model, instructions: request.instructions)
                        var options = GenerationOptions()
                        if let temperature = request.temperature { options.temperature = temperature }
                        if let maxTokens = request.maxTokens { options.maximumResponseTokens = maxTokens }
                        let stream = session.streamResponse(to: request.prompt, options: options)
                        for try await snapshot in stream {
                            continuation.yield(StreamSnapshot(
                                text: snapshot.content,
                                promptTokens: snapshot.usage.input.totalTokenCount,
                                completionTokens: snapshot.usage.output.totalTokenCount
                            ))
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: mapAFMError(error))
                    }
                } else {
                    continuation.finish(throwing: AdapterError("PCC requires macOS 27+", statusCode: 500, type: "server_error", code: "unsupported_os"))
                }
#else
                continuation.finish(throwing: AdapterError("FoundationModels unavailable", statusCode: 500, type: "server_error", code: "provider_unavailable"))
#endif
            }
        }
    }
}

// MARK: - Hybrid router (system primary, PCC optional fallback)

/// Prefers on-device System (offline, unlimited). Optionally falls back to PCC
/// when the system model is unavailable and PCC is available + entitled.
public final class HybridProvider: LLMProvider, Sendable {
    private let system: SystemProvider
    private let pcc: PCCProvider
    private let preferPCC: Bool

    public init(systemModelID: String, useCase: String, guardrails: String, preferPCC: Bool) {
        self.system = SystemProvider(modelID: systemModelID, useCase: useCase, guardrails: guardrails)
        self.pcc = PCCProvider()
        self.preferPCC = preferPCC
    }

    public func isAvailable() async -> (Bool, String?) {
        let (systemOK, systemReason) = await system.isAvailable()
        if systemOK { return (true, nil) }
        if preferPCC {
            let (pccOK, pccReason) = await pcc.isAvailable()
            if pccOK { return (true, nil) }
            return (false, systemReason ?? pccReason)
        }
        return (false, systemReason)
    }

    public func listModels() async -> [String] {
        var models = await system.listModels()
        if preferPCC {
            let (pccOK, _) = await pcc.isAvailable()
            if pccOK {
                models += await pcc.listModels()
            }
        }
        return models
    }

    /// Route a request to PCC only when explicitly asked via model id or when
    /// preferPCC is set and system is unavailable.
    public func provider(forModel model: String) async -> any LLMProvider {
        if model == "apple.fm.pcc" || model == "pcc" {
            return pcc
        }
        if preferPCC {
            let (systemOK, _) = await system.isAvailable()
            if !systemOK {
                let (pccOK, _) = await pcc.isAvailable()
                if pccOK { return pcc }
            }
        }
        return system
    }

    public func generate(_ request: GenerationRequest) async throws -> GenerationResult {
        // Default path uses system; explicit PCC routing happens in the handler
        // via provider(forModel:). This keeps the protocol simple for tests.
        try await system.generate(request)
    }

    public func streamGenerate(_ request: GenerationRequest) -> AsyncThrowingStream<StreamSnapshot, Error> {
        system.streamGenerate(request)
    }

}

// MARK: - Mock provider (tests + graceful dev without model assets)

public final class MockProvider: LLMProvider, @unchecked Sendable {
    private let modelID: String
    private let available: Bool
    private let unavailableReason: String?
    private let responseText: String
    private let snapshots: [String]
    /// Per-call response texts; when non-empty, call N uses texts[min(N, end)].
    private let responseTexts: [String]
    private let lock = NSLock()
    private var callCount = 0

    public init(
        modelID: String = "apple.fm.system",
        available: Bool = true,
        unavailableReason: String? = nil,
        responseText: String = "ok",
        responseTexts: [String] = [],
        snapshots: [String] = ["o", "ok"]
    ) {
        self.modelID = modelID
        self.available = available
        self.unavailableReason = unavailableReason
        self.responseText = responseText
        self.responseTexts = responseTexts
        self.snapshots = snapshots
    }

    private func nextText() -> String {
        lock.lock()
        defer {
            callCount += 1
            lock.unlock()
        }
        guard !responseTexts.isEmpty else { return responseText }
        return responseTexts[min(callCount, responseTexts.count - 1)]
    }

    public var generateCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return callCount
    }

    public func isAvailable() async -> (Bool, String?) {
        (available, unavailableReason)
    }

    public func listModels() async -> [String] {
        [modelID]
    }

    public func generate(_ request: GenerationRequest) async throws -> GenerationResult {
        GenerationResult(text: nextText(), promptTokens: nil, completionTokens: nil)
    }

    public func streamGenerate(_ request: GenerationRequest) -> AsyncThrowingStream<StreamSnapshot, Error> {
        if !responseTexts.isEmpty {
            let text = nextText()
            return AsyncThrowingStream { continuation in
                continuation.yield(StreamSnapshot(text: text, promptTokens: nil, completionTokens: nil))
                continuation.finish()
            }
        }
        let snapshots = snapshots
        return AsyncThrowingStream { continuation in
            for snapshot in snapshots {
                continuation.yield(StreamSnapshot(text: snapshot, promptTokens: nil, completionTokens: nil))
            }
            continuation.finish()
        }
    }
}
