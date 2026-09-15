import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Errors for the JSON-Schema-subset translator.
struct SchemaTranslationError: Error {
    var message: String
}

#if canImport(FoundationModels)
/// Translates a JSON Schema dictionary (OpenAI `response_format.json_schema`
/// dialect) into a native AFM `GenerationSchema` for enforced guided
/// generation.
///
/// Supported subset (covers harness-emitted schemas, e.g. zodoutil):
/// objects (nested, recursive-safe via `$defs`), arrays, primitives,
/// string enums, `required`/`optional`, `description`, `#/$defs/` + `#/definitions/`
/// `$ref`s, nullable `type` arrays. Anything else (e.g. `oneOf`, `not`,
/// numeric ranges) throws and the caller falls back to a prompt hint.
@available(macOS 26, *)
struct SchemaTranslator {
    var definitions: [String: [String: Any]] = [:]
    var dependencies: [DynamicGenerationSchema] = []
    var building: Set<String> = []
    var builtDefinitions: Set<String> = []

    mutating func translateRoot(name: String, json: [String: Any]) throws -> GenerationSchema {
        if let defs = json["$defs"] as? [String: [String: Any]] {
            for (key, value) in defs { definitions[key] = value }
        }
        if let defs = json["definitions"] as? [String: [String: Any]] {
            for (key, value) in defs { definitions[key] = value }
        }
        let root = try node(name: sanitize(name), json: json)
        return try GenerationSchema(root: root, dependencies: dependencies)
    }

    mutating func node(name: String, json: [String: Any]) throws -> DynamicGenerationSchema {
        if let ref = json["$ref"] as? String {
            return try reference(ref: ref)
        }
        // Nullable `type` arrays like ["string", "null"].
        var typeName = json["type"] as? String
        var optional = false
        if typeName == nil, let types = json["type"] as? [String] {
            let nonNull = types.filter { $0 != "null" }
            if nonNull.count == 1, types.count == 2 {
                typeName = nonNull.first
                optional = true
            }
        }
        guard let type = typeName else {
            // No explicit type: infer object (properties) or fall back to string.
            if json["properties"] != nil {
                return try objectNode(name: name, json: json, forceOptional: optional)
            }
            if json["enum"] != nil {
                return try enumNode(name: name, json: json)
            }
            throw SchemaTranslationError(message: "Schema node '\(name)' has no usable type")
        }
        switch type {
        case "string":
            if let choices = json["enum"] as? [String], !choices.isEmpty {
                return DynamicGenerationSchema(name: name, description: stringField(json, "description"), anyOf: choices)
            }
            if json["enum"] != nil {
                throw SchemaTranslationError(message: "Non-string enum unsupported for '\(name)'")
            }
            return DynamicGenerationSchema(type: String.self)
        case "integer", "int":
            return DynamicGenerationSchema(type: Int.self)
        case "number", "float", "double":
            return DynamicGenerationSchema(type: Double.self)
        case "boolean", "bool":
            return DynamicGenerationSchema(type: Bool.self)
        case "array":
            return try arrayNode(name: name, json: json)
        case "object":
            return try objectNode(name: name, json: json, forceOptional: optional)
        case "null":
            throw SchemaTranslationError(message: "Null root/property unsupported for '\(name)'")
        default:
            throw SchemaTranslationError(message: "Unsupported type '\(type)' for '\(name)'")
        }
    }

    // MARK: - Nodes

    mutating func objectNode(name: String, json: [String: Any], forceOptional: Bool) throws -> DynamicGenerationSchema {
        _ = forceOptional
        // Missing/empty properties = open object: constrain nothing.
        let properties = json["properties"] as? [String: [String: Any]] ?? [:]
        let required = Set(json["required"] as? [String] ?? [])
        var built: [DynamicGenerationSchema.Property] = []
        for key in properties.keys.sorted() {
            guard let propJSON = properties[key] else { continue }
            let childName = sanitize("\(name)_\(key)")
            let child: DynamicGenerationSchema
            if (propJSON["type"] as? String) == "object" || propJSON["properties"] != nil {
                // Named nested object goes to dependencies; property references it.
                if building.contains(childName) {
                    throw SchemaTranslationError(message: "Recursive schema '\(childName)' unsupported")
                }
                building.insert(childName)
                let nested = try objectNode(name: childName, json: propJSON, forceOptional: false)
                building.remove(childName)
                dependencies.append(nested)
                child = DynamicGenerationSchema(referenceTo: childName)
            } else if let ref = propJSON["$ref"] as? String {
                child = try reference(ref: ref)
            } else {
                child = try node(name: childName, json: propJSON)
            }
            let isOptional = !required.contains(key) || isNullable(propJSON)
            built.append(DynamicGenerationSchema.Property(
                name: key,
                description: stringField(propJSON, "description"),
                schema: child,
                isOptional: isOptional
            ))
        }
        return DynamicGenerationSchema(
            name: name,
            description: stringField(json, "description"),
            properties: built
        )
    }

    mutating func arrayNode(name: String, json: [String: Any]) throws -> DynamicGenerationSchema {
        let minItems = json["minItems"] as? Int
        let maxItems = json["maxItems"] as? Int
        guard let items = json["items"] as? [String: Any] else {
            throw SchemaTranslationError(message: "Array '\(name)' has no items")
        }
        if (items["type"] as? String) == "object" || items["properties"] != nil {
            let childName = sanitize("\(name)_item")
            if building.contains(childName) {
                throw SchemaTranslationError(message: "Recursive schema '\(childName)' unsupported")
            }
            building.insert(childName)
            let nested = try objectNode(name: childName, json: items, forceOptional: false)
            building.remove(childName)
            dependencies.append(nested)
            return DynamicGenerationSchema(
                arrayOf: DynamicGenerationSchema(referenceTo: childName),
                minimumElements: minItems,
                maximumElements: maxItems
            )
        }
        if let ref = items["$ref"] as? String {
            return DynamicGenerationSchema(
                arrayOf: try reference(ref: ref),
                minimumElements: minItems,
                maximumElements: maxItems
            )
        }
        let item = try node(name: sanitize("\(name)_item"), json: items)
        return DynamicGenerationSchema(
            arrayOf: item, minimumElements: minItems, maximumElements: maxItems
        )
    }

    mutating func enumNode(name: String, json: [String: Any]) throws -> DynamicGenerationSchema {
        if let choices = json["enum"] as? [String], !choices.isEmpty {
            return DynamicGenerationSchema(name: name, description: stringField(json, "description"), anyOf: choices)
        }
        throw SchemaTranslationError(message: "Non-string enum unsupported for '\(name)'")
    }

    mutating func reference(ref: String) throws -> DynamicGenerationSchema {
        let prefixDefs = "#/$defs/"
        let prefixDefinitions = "#/definitions/"
        let key: String?
        if ref.hasPrefix(prefixDefs) {
            key = String(ref.dropFirst(prefixDefs.count))
        } else if ref.hasPrefix(prefixDefinitions) {
            key = String(ref.dropFirst(prefixDefinitions.count))
        } else {
            key = nil
        }
        guard let key, let target = definitions[key] else {
            throw SchemaTranslationError(message: "Unresolvable $ref '\(ref)'")
        }
        if building.contains(key) {
            // Recursive reference: point at the in-progress definition.
            return DynamicGenerationSchema(referenceTo: key)
        }
        if builtDefinitions.contains(key) {
            // Already materialized by an earlier property: reuse it instead
            // of appending a duplicate dependency.
            return DynamicGenerationSchema(referenceTo: key)
        }
        building.insert(key)
        let nested: DynamicGenerationSchema
        if (target["type"] as? String) == "object" || target["properties"] != nil {
            nested = try objectNode(name: key, json: target, forceOptional: false)
        } else {
            nested = try node(name: key, json: target)
        }
        building.remove(key)
        dependencies.append(nested)
        builtDefinitions.insert(key)
        return DynamicGenerationSchema(referenceTo: key)
    }

    // MARK: - Helpers

    func stringField(_ json: [String: Any], _ key: String) -> String? {
        json[key] as? String
    }

    func isNullable(_ json: [String: Any]) -> Bool {
        if let types = json["type"] as? [String] { return types.contains("null") }
        return false
    }

    func sanitize(_ name: String) -> String {
        let cleaned = name.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : "_" }
        let text = String(cleaned)
        return text.isEmpty ? "Node" : text
    }
}

/// Builds a native AFM `GenerationSchema` from a `response_format.json_schema`
/// JSON string. Throws `SchemaTranslationError` for constructs outside the
/// supported subset (caller falls back to a prompt hint).
@available(macOS 26, *)
public func buildGenerationSchema(name: String, jsonSchema: String) throws -> GenerationSchema {
    guard let data = jsonSchema.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw SchemaTranslationError(message: "Schema is not a JSON object")
    }
    var translator = SchemaTranslator()
    return try translator.translateRoot(name: name, json: json)
}
#endif
