import XCTest
@testable import AppleFMServe

final class GenerationSchemaTests: XCTestCase {
    func testFlatObjectBuilds() throws {
        guard #available(macOS 26, *) else { return }
        let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object",
         "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
         "required": ["name", "age"]}
        """)
        let description = String(describing: schema)
        XCTAssertTrue(description.contains("name"))
        XCTAssertTrue(description.contains("age"))
    }

    func testStringEnumNestedAndArrayBuild() throws {
        guard #available(macOS 26, *) else { return }
        let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object",
         "properties": {
           "color": {"type": "string", "enum": ["mauve", "chartreuse"]},
           "address": {"type": "object",
             "properties": {"street": {"type": "string"}},
             "required": ["street"]},
           "tags": {"type": "array", "items": {"type": "string"}},
           "nickname": {"type": ["string", "null"]}
         },
         "required": ["color"]}
        """)
        let description = String(describing: schema)
        XCTAssertTrue(description.contains("mauve"))
        XCTAssertTrue(description.contains("address"))
    }

    func testDefsRefBuilds() throws {
        guard #available(macOS 26, *) else { return }
        let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object",
         "$defs": {"Pet": {"type": "object",
           "properties": {"name": {"type": "string"}}, "required": ["name"]}},
         "properties": {"pet": {"$ref": "#/$defs/Pet"}},
         "required": ["pet"]}
        """)
        XCTAssertTrue(String(describing: schema).contains("Pet"))
    }

    func testRecursiveRefBuilds() throws {
        guard #available(macOS 26, *) else { return }
        // Must terminate (cycle resolves to a reference, not infinite nesting).
        let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object",
         "$defs": {"Node": {"type": "object",
           "properties": {"child": {"$ref": "#/$defs/Node"}}}},
         "properties": {"root": {"$ref": "#/$defs/Node"}},
         "required": ["root"]}
        """)
        XCTAssertTrue(String(describing: schema).contains("Node"))
    }

    func testSharedDefReferencedTwiceBuilds() throws {
        guard #available(macOS 26, *) else { return }
        // Two properties sharing one $def must not duplicate dependencies.
        let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object",
         "$defs": {"Pet": {"type": "object",
           "properties": {"name": {"type": "string"}}, "required": ["name"]}},
         "properties": {
           "cat": {"$ref": "#/$defs/Pet"},
           "dog": {"$ref": "#/$defs/Pet"}},
         "required": ["cat", "dog"]}
        """)
        XCTAssertTrue(String(describing: schema).contains("Pet"))
    }

    func testOpenObjectBuilds() throws {
        guard #available(macOS 26, *) else { return }
        let schema = try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object"}
        """)
        XCTAssertFalse(String(describing: schema).isEmpty)
    }

    func testUnsupportedConstructThrows() {
        guard #available(macOS 26, *) else { return }
        XCTAssertThrowsError(try buildGenerationSchema(name: "Response", jsonSchema: """
        {"type": "object",
         "properties": {"choice": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}}
        """))
        XCTAssertThrowsError(try buildGenerationSchema(name: "Response", jsonSchema: "\"just a string\""))
    }
}
