// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "apple-fm-serve",
    platforms: [
        .macOS(.v14),
    ],
    targets: [
        .target(
            name: "AppleFMServe",
            path: "Sources/AppleFMServe"
        ),
        .executableTarget(
            name: "apple-fm-serve",
            dependencies: ["AppleFMServe"],
            path: "Sources/AppleFMExe"
        ),
        .executableTarget(
            name: "apple-fm-verify",
            dependencies: ["AppleFMServe"],
            path: "Sources/AppleFMVerify"
        ),
        .testTarget(
            name: "AppleFMServeTests",
            dependencies: ["AppleFMServe"],
            path: "Tests/AppleFMServeTests"
        ),
    ]
)
