import AppleFMServe
import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

@main
struct AppleFMServe {
    static func main() async {
        let config = Config.fromEnvironment()

        // Require macOS 26+ for FoundationModels at runtime.
        #if canImport(FoundationModels)
        if #available(macOS 26, *) {
            // Runtime check only; providers gate individual calls.
        } else {
            fputs("apple-fm-serve requires macOS 26 or later\n", stderr)
            exit(1)
        }
        #endif

        let useMock = ProcessInfo.processInfo.environment["APPLE_FM_USE_MOCK"] == "1"
        let provider: any LLMProvider
        let hybrid: HybridProvider?
        if useMock {
            let mock = MockProvider(modelID: config.modelID)
            provider = mock
            hybrid = nil
        } else {
            let router = HybridProvider(
                systemModelID: config.modelID,
                useCase: config.useCase,
                guardrails: config.guardrails,
                preferPCC: config.preferPCC
            )
            hybrid = router
            provider = router
        }

        let handler = ChatHandler(config: config, provider: provider, hybrid: hybrid)
        let server = HTTPServer(config: config, handler: handler)

        // Log model availability at startup (mirrors /readyz). Stderr: stdout
        // block-buffers under redirection and hides startup lines in logs.
        let (available, reason) = await provider.isAvailable()
        if available {
            let models = await provider.listModels()
            logLine("models: \(models.joined(separator: ", "))")
            #if canImport(FoundationModels)
            if #available(macOS 27, *) {
                let variant = SystemLanguageModel.default.variant.displayName
                logLine("on-device variant: \(variant)")
            }
            #endif
        } else {
            logLine("model not ready: \(reason ?? "unknown") (serving 503 until ready)")
        }

        do {
            try await server.run()
        } catch {
            fputs("apple-fm-serve failed: \(error)\n", stderr)
            exit(1)
        }
    }
}
