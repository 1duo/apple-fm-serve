import Foundation

/// Server configuration, sourced from environment (APPLE_FM_*).
/// Mirrors the previous Python implementation's env contract for easy migration.
public struct Config: Sendable {
    public var host: String
    public var port: Int
    public var modelID: String
    public var apiKey: String?
    public var maxConcurrency: Int
    public var requestTimeoutSeconds: Double
    public var preferPCC: Bool
    public var useCase: String
    public var guardrails: String
    public var logBodies: Bool = false

    public static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) -> Config {
        func intValue(_ key: String, default defaultValue: Int) -> Int {
            guard let raw = env[key], let value = Int(raw) else { return defaultValue }
            return value
        }
        func doubleValue(_ key: String, default defaultValue: Double) -> Double {
            guard let raw = env[key], let value = Double(raw) else { return defaultValue }
            return value
        }
        func boolValue(_ key: String, default defaultValue: Bool) -> Bool {
            guard let raw = env[key]?.lowercased() else { return defaultValue }
            return raw == "1" || raw == "true" || raw == "yes"
        }
        let apiKey = env["APPLE_FM_API_KEY"].flatMap { $0.isEmpty ? nil : $0 }
        return Config(
            host: env["APPLE_FM_HOST"] ?? "127.0.0.1",
            port: intValue("APPLE_FM_PORT", default: 8000),
            modelID: env["APPLE_FM_MODEL_ID"] ?? "apple.fm.system",
            apiKey: apiKey,
            maxConcurrency: min(max(intValue("APPLE_FM_MAX_CONCURRENCY", default: 4), 1), 128),
            requestTimeoutSeconds: min(max(doubleValue("APPLE_FM_REQUEST_TIMEOUT_S", default: 120.0), 1.0), 600.0),
            preferPCC: boolValue("APPLE_FM_PREFER_PCC", default: false),
            useCase: env["APPLE_FM_USE_CASE"] ?? "general",
            guardrails: env["APPLE_FM_GUARDRAILS"] ?? "default",
            logBodies: boolValue("APPLE_FM_LOG_BODIES", default: false)
        )
    }

    /// Canonical model IDs served by this server.
    /// "system" is the on-device AFM (auto-selects Core Advanced on capable
    /// hardware, falls back to Core). "pcc" is the Private Cloud Compute model
    /// (requires managed entitlement; only advertised when available).
    public var servedModelAliases: [String] {
        [modelID, "system", "apple.fm.system"]
    }
}
